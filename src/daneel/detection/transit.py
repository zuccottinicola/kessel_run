"""import batman
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('ldc_result.csv', sep=r'\s+')  
dd=df.head()
print(dd)
u1 = df['c1'].mean()
u2 = df['c2'].mean()
print(u1,u2)
name = "Kepler-10 b"
p = batman.TransitParams()
p.t0 = 0.0 # relative time, days
p.per = 0.83749 # orbital period, days
p.rp = 0.012581 # Rp / Rstar
p.a = 3.6 # a / Rstar
p.inc = 89.38 # degrees
p.ecc = 0.0
p.w = 90.0
p.limb_dark = "quadratic"
p.u = [u1, u2]
t = np.linspace(-0.075, 0.075, 1000)
flux = batman.TransitModel(p,t).light_curve(p)


plt.plot(t,flux, color='red')
plt.ticklabel_format(useOffset=False, axis='y')
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
plt.title("Kepler_10b transit")
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig('kepler-10b_assignment1_taskF.png')
plt.show()
"""
import batman
import matplotlib.pyplot as plt
import numpy as np



def transit(params, output):
    """Compute and plot the transit light curve of a planet.

    params : (t0, per, rp, a, inc, ecc, w, u, limb_dark; optional: name)
    output : path of the PNG to save
    """
    planets = params.keys()
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in planets:
        p = batman.TransitParams()
        p.t0 = params[n].get("t0", 0.0)
        p.per = params[n]["per"]
        p.rp = params[n]["rp"]
        p.a = params[n]["a"]
        p.inc = params[n]["inc"]
        p.ecc = params[n].get("ecc", 0.0)
        p.w = params[n].get("w", 90.0)
        p.limb_dark = params[n].get("limb_dark", "quadratic")
        p.u = params[n].get("u", [0.1, 0.3])

        t = np.linspace(p.t0-0.25, p.t0+0.25, 1000)
        flux = batman.TransitModel(p, t).light_curve(p)
        
        ax.plot(t, flux, label = n)
    
    ax.ticklabel_format(useOffset=False, axis="y")
    ax.set(xlabel="Time from central transit (days)", ylabel="Relative flux", title="transit light curves")
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.show()
    plt.close(fig)
    return t, flux