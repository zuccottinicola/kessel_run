import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

side = 44


def prepare_data(path, n_points, device):
    half = n_points - 32
    if 2 * half != side * side:
        raise ValueError(f"2*(n_points-32)={2*half} deve essere {side*side}")
    fcols = [f"flux_{i:04d}" for i in range(half)]
    ecols = [f"flux_err_{i:04d}" for i in range(half)]
    df = pd.read_csv(path, usecols=["label"] + fcols + ecols)
    df = df[df.label == 1]
    f = df[fcols].to_numpy(np.float32)
    e = df[ecols].to_numpy(np.float32)

    med = np.nanmedian(f, axis=1, keepdims=True)
    x = f / med - 1.0
    ex = e / med
    sx = float(np.nanpercentile(np.abs(x), 99.5))
    se = float(np.nanpercentile(ex, 99.5))
    x = np.clip(np.nan_to_num(x / sx, nan=0.0), -1, 1)
    ex = np.clip(np.nan_to_num(ex / se, nan=0.0), 0, 1) * 2 - 1

    z = np.concatenate([x, ex], axis=1)
    X = torch.tensor(z).view(-1, 1, side, side).to(device)
    return X, (sx, se), half


class Generator(nn.Module):
    def __init__(self, zdim, c=352):
        super().__init__()
        self.c = c
        self.fc = nn.Linear(zdim, c * 11 * 11)
        self.net = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.ConvTranspose2d(c, c // 2, 4, 2, 1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c // 2, 1, 4, 2, 1),
            nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, self.c, 11, 11))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 88, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(88, 352, 4, 2, 1),
            nn.BatchNorm2d(352),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(352 * 11 * 11, 1))

    def forward(self, x):
        return self.net(x)


def to_curves(g, n, zdim, scales, half, device):
    sx, se = scales
    with torch.no_grad():  # niente eval(): BatchNorm usa le statistiche del batch
        out = g(torch.randn(max(n, 128), zdim, device=device))[:n]
    out = out.cpu().view(n, -1).numpy()
    flux = 1.0 + out[:, :half] * sx
    err = (out[:, half:] + 1) / 2 * se
    return flux, err


def plot_curves(curves, path):
    flux, err = curves
    n = len(flux)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows), sharex=True, squeeze=False)
    x = np.arange(flux.shape[1])
    for a, f, e in zip(ax.ravel(), flux, err):
        a.plot(x, f, lw=0.8)
        a.fill_between(x, f - e, f + e, alpha=0.3, lw=0)
    fig.supxlabel("Sample")
    fig.supylabel("Normalized flux")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def dream(cfg):
    data_path = cfg["data_path"]
    n_points = cfg.get("n_points", 1000)
    epochs = cfg.get("epochs", 2000)
    bs = cfg.get("batch_size", 88)
    zdim = cfg.get("z_dim", 100)
    lr = cfg.get("lr", 2e-4)
    every = cfg.get("sample_every", 200)
    n_samples = cfg.get("n_samples", 8)
    out_dir = cfg.get("output_dir", "dream_output")
    seed = cfg.get("seed", 0)
    ema_decay = cfg.get("ema_decay", 0.999)

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, scales, half = prepare_data(data_path, n_points, dev)
    print(f"curve di training: {len(X)} (device: {dev})")

    g, d = Generator(zdim).to(dev), Discriminator().to(dev)
    g_ema = copy.deepcopy(g)
    og = torch.optim.Adam(g.parameters(), lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr / 2, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    hist = []

    for ep in range(epochs):
        idx = torch.randperm(len(X), device=dev)
        sld, slg, nb = 0.0, 0.0, 0
        for i in range(0, len(X) - bs + 1, bs):
            real = X[idx[i:i + bs]]
            fake = g(torch.randn(bs, zdim, device=dev))
            ld = bce(d(real), torch.full((bs, 1), 0.9, device=dev)) + bce(d(fake.detach()), torch.zeros(bs, 1, device=dev))
            od.zero_grad()
            ld.backward()
            od.step()
            lg = bce(d(fake), torch.ones(bs, 1, device=dev))
            og.zero_grad()
            lg.backward()
            og.step()

            with torch.no_grad():
                for pe, p in zip(g_ema.parameters(), g.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
                for be, b in zip(g_ema.buffers(), g.buffers()):
                    be.copy_(b)

            sld += ld.item()
            slg += lg.item()
            nb += 1
        hist.append((sld / nb, slg / nb))
        if ep % every == 0:
            print(f"ep {ep} D {hist[-1][0]:.3f} G {hist[-1][1]:.3f}")
            c = to_curves(g_ema, n_samples, zdim, scales, half, dev)
            plot_curves(c, os.path.join(out_dir, f"dream_ep{ep:05d}.png"))

    curves = to_curves(g_ema, n_samples, zdim, scales, half, dev)
    plot_curves(curves, os.path.join(out_dir, "dream_final.png"))
    np.savez(os.path.join(out_dir, "generated_curves.npz"), flux=curves[0], err=curves[1])
    torch.save(g_ema.state_dict(), os.path.join(out_dir, "generator.pt"))

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h[:, 0], label="discriminator")
    ax.plot(h[:, 1], label="generator")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150)
    plt.close(fig)
    print(f"Output salvati in {out_dir}/")