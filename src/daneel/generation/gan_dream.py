import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

side = 32  


def prepare_data(path, n_points, device):

    if n_points > side * side:
        raise ValueError(f"n_points={n_points} supera {side * side}: serve troncare")
    
    cols = ["label"] + [f"flux_{i:04d}" for i in range(n_points)]
    df = pd.read_csv(path, usecols=cols)
    f = df[df.label == 1].drop(columns="label").to_numpy(np.float32)
    med = np.nanmedian(f, axis=1, keepdims=True)
    x = f / med - 1.0
    scale = float(np.nanpercentile(np.abs(x), 99))
    x = np.clip(np.nan_to_num(x / scale, nan=0.0), -1, 1)
    left = (side * side - n_points) // 2
    right = side * side - n_points - left
    x = np.pad(x, ((0, 0), (left, right)))
    X = torch.tensor(x).view(-1, 1, side, side).to(device)
    return X, scale, left


class Generator(nn.Module):
    def __init__(self, zdim):
        super().__init__()
        self.fc = nn.Linear(zdim, 256 * 4 * 4)
        up = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.net = nn.Sequential(*up(256, 128), *up(128, 64), nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 4, 4))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(256 * 16, 1))

    def forward(self, x):
        return self.net(x)


def to_curves(g, n, zdim, scale, left, n_points, device):
    g.eval()
    with torch.no_grad():
        out = g(torch.randn(n, zdim, device=device)).cpu().view(n, -1)
    g.train()
    return 1.0 + out[:, left:left + n_points].numpy() * scale


def plot_curves(curves, path):
    n = len(curves)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.5 * rows), sharex=True, squeeze=False)
    for a, c in zip(ax.ravel(), curves):
        a.plot(c, lw=0.8)
    fig.supxlabel("Sample")
    fig.supylabel("Normalized flux")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def dream(cfg):
    data_path = cfg["data_path"]
    n_points = cfg.get("n_points", 1000)
    epochs = cfg.get("epochs", 2000)
    bs = cfg.get("batch_size", 64)
    zdim = cfg.get("z_dim", 100)
    lr = cfg.get("lr", 2e-4)
    every = cfg.get("sample_every", 200)
    n_samples = cfg.get("n_samples", 8)
    out_dir = cfg.get("output_dir", "dream_output")
    seed = cfg.get("seed", 0)

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X, scale, left = prepare_data(data_path, n_points, dev)
    print(f"curve di training: {len(X)} (device: {dev})")

    g, d = Generator(zdim).to(dev), Discriminator().to(dev)
    og = torch.optim.Adam(g.parameters(), lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    hist = []

    for ep in range(epochs):
        idx = torch.randperm(len(X), device=dev)
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
        hist.append((ld.item(), lg.item()))
        if ep % every == 0:
            print(f"ep {ep} D {ld.item():.3f} G {lg.item():.3f}")
            c = to_curves(g, n_samples, zdim, scale, left, n_points, dev)
            plot_curves(c, os.path.join(out_dir, f"dream_ep{ep:05d}.png"))

    curves = to_curves(g, n_samples, zdim, scale, left, n_points, dev)
    plot_curves(curves, os.path.join(out_dir, "dream_final.png"))
    np.save(os.path.join(out_dir, "generated_curves.npy"), curves)
    torch.save(g.state_dict(), os.path.join(out_dir, "generator.pt"))

    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(h[:, 0], label="discriminatore"); ax.plot(h[:, 1], label="generatore")
    ax.set_xlabel("epoca"); ax.set_ylabel("loss"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss.png"), dpi=150)
    plt.close(fig)
    print(f"Output salvati in {out_dir}/")