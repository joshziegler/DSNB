import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("plot_style.mplstyle")

z = np.linspace(0, 4, 100)
L = np.logspace(8.0, 14.0, 120)

plt.plot(
    z,
    integ.simps(sn.Phi_Spiral(L, z), x=L, axis=1),
    color="C0",
    linewidth=1,
    label="Spiral",
)
plt.plot(
    z,
    integ.simps(sn.Phi_Spiral(L[:40], z), x=L[:40], axis=1),
    ls=":",
    color="C0",
    linewidth=1,
)
plt.plot(
    z,
    integ.simps(sn.Phi_Spiral(L[40:80], z), x=L[40:80], axis=1),
    ls="-.",
    color="C0",
    linewidth=1,
)
plt.plot(
    z,
    integ.simps(sn.Phi_Spiral(L[80:], z), x=L[80:], axis=1),
    ls="--",
    color="C0",
    linewidth=1,
)

plt.plot(
    z,
    integ.simps(sn.Phi_Starburst(L, z), x=L, axis=1),
    color="C1",
    linewidth=1,
    label="Spheroidal",
)
plt.plot(
    z,
    integ.simps(sn.Phi_Starburst(L[:40], z), x=L[:40], axis=1),
    ls=":",
    color="C1",
    linewidth=1,
)
plt.plot(
    z,
    integ.simps(sn.Phi_Starburst(L[40:80], z), x=L[40:80], axis=1),
    ls="-.",
    color="C1",
    linewidth=1,
)
plt.plot(
    z,
    integ.simps(sn.Phi_Starburst(L[80:], z), x=L[80:], axis=1),
    ls="--",
    color="C1",
    linewidth=1,
)
plt.plot(
    z,
    integ.simps(sn.Phi_Spiral(L, z), x=L, axis=1)
    + integ.simps(sn.Phi_Starburst(L, z), x=L, axis=1),
    color="k",
    linewidth=2,
    label="Sum",
)

plt.plot([0, 1], [-1, -1], label=r"$\log(L)$ = 8-10", color="k", ls=":", linewidth=1)
plt.plot([0, 1], [-1, -1], label=r"$\log(L)$ = 10-12", color="k", ls="-.", linewidth=1)
plt.plot([0, 1], [-1, -1], label=r"$\log(L)$ = 12-14", color="k", ls="--", linewidth=1)
plt.plot([0, 1], [-1, -1], label=r"$\log(L)$ = 8-14", color="k", ls="-", linewidth=1)

plt.yscale("log")
plt.legend(frameon=True, fontsize=14, loc="lower right")
plt.xlim(0, 4.0)
plt.ylim(1e5, 2e9)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$\int \frac{dN_g}{d\log L}dL$ $\mathrm{[L_\odot \,Mpc^{-3}]}$")
plt.savefig("../Plots/Luminosity_funcs.pdf", bbox_inches="tight")
