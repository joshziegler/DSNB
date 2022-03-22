import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

Hubble_high = np.loadtxt(
    "../Data/SN_constraints/Strogler2015_high.txt", usecols=(1,), delimiter=","
)
z_hubble, Hubble_mid = np.loadtxt(
    "../Data/SN_constraints/Strogler2015_mid.txt", unpack=True, delimiter=","
)
Hubble_low = np.loadtxt(
    "../Data/SN_constraints/Strogler2015_low.txt", usecols=(1,), delimiter=","
)

Hubble_high *= 1e-4
Hubble_mid *= 1e-4
Hubble_low *= 1e-4

NOT_high = np.loadtxt(
    "../Data/SN_constraints/Petrushevska2016_high.txt", usecols=(1,), delimiter=","
)
z_NOT, NOT_mid = np.loadtxt(
    "../Data/SN_constraints/Petrushevska2016_mid.txt", unpack=True, delimiter=","
)
NOT_low = np.loadtxt(
    "../Data/SN_constraints/Petrushevska2016_low.txt", usecols=(1,), delimiter=","
)

NOT_high *= 1e-4
NOT_mid *= 1e-4
NOT_low *= 1e-4
NOT_low[-2:] *= 1e-10

z_dahl, low_dahl, mid_dahl, hi_dahl = np.loadtxt(
    "../Data/SN_constraints/Dahlen12.txt", unpack=True
)
low_dahl *= 1e-4
mid_dahl *= 1e-4
hi_dahl *= 1e-4

z = np.linspace(0, 4, 100)

rhoCC_spiral = sn.RCC_density(z, gtype="spiral")
rhoCC_starburst = sn.RCC_density(z, gtype="starburst")
rhoCC_AGNspiral = sn.RCC_density(z, gtype="AGN spiral")
rhoCC_AGNstarburst = sn.RCC_density(z, gtype="AGN starburst")
rhoCC = rhoCC_spiral + rhoCC_starburst + rhoCC_AGNspiral + rhoCC_AGNstarburst

rhoCC_spiral_sal = sn.RCC_density(z, gtype="spiral", salpeter=True)
rhoCC_starburst_sal = sn.RCC_density(z, gtype="starburst", salpeter=True)
rhoCC_AGNspiral_sal = sn.RCC_density(z, gtype="AGN spiral", salpeter=True)
rhoCC_AGNstarburst_sal = sn.RCC_density(z, gtype="AGN starburst", salpeter=True)
rhoCC_sal = (
    rhoCC_spiral_sal
    + rhoCC_starburst_sal
    + rhoCC_AGNspiral_sal
    + rhoCC_AGNstarburst_sal
)


plt.figure(figsize=(7, 5))
fig, ax = plt.subplots()

plt.errorbar(
    z_NOT,
    NOT_mid,
    yerr=[NOT_mid - NOT_low, NOT_high - NOT_mid],
    color="C4",
    marker="o",
    label="Petrushevska et al. (2016)",
    ls="none",
)
plt.errorbar(
    z_hubble,
    Hubble_mid,
    (Hubble_mid - Hubble_low, Hubble_high - Hubble_mid),
    color="C1",
    marker="x",
    label="Strogler et al. (2015)",
    ls="none",
)
plt.errorbar(
    z_dahl,
    mid_dahl,
    (mid_dahl - low_dahl, hi_dahl - mid_dahl),
    color="C3",
    marker="+",
    label="Dahlen et al. (2012)",
    ls="none",
)
plt.errorbar(
    0.01,
    1.5e-4,
    [[0.3e-4], [0.4e-4]],
    color="C9",
    marker="H",
    label="Mattila et al. (2018)",
    ls="none",
)

plt.plot(z, rhoCC, "-", color="C0", label="Varying IMF (Total)")
plt.plot(z, rhoCC_sal, "--", color="C2", label="Salpeter IMF (Total)")

plt.yscale("log")
plt.xlim(0.0, 3.0)
plt.ylim(4e-5, 8e-2)
plt.legend(loc="upper left", fontsize=12, ncol=1, frameon=True)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$R_{\mathrm{CCSN}} \, \mathrm{[yr^{-1}\,Mpc^{-3}]}$")
plt.savefig("../plots/R_cc_total.pdf", bbox_inches="tight")

z = np.linspace(0, 4, 100)

plt.figure(figsize=(7, 5))
fig, ax = plt.subplots()

plt.plot(z, rhoCC_spiral, "-", color="C0", label="Spiral")
plt.plot(z, rhoCC_starburst, "-", color="C2", label="Spheroidal")

plt.plot(z, rhoCC_spiral_sal, "--", color="C0")
plt.plot(z, rhoCC_starburst_sal, "--", color="C2")

plt.plot([-2, -1], [0, 0], "-", color="k", label="Varying IMF")
plt.plot([-2, -1], [0, 0], "--", color="k", label="Salpeter IMF")

plt.yscale("log")
plt.xlim(0, 3)
plt.ylim(4e-5, 8e-2)
plt.legend(loc="upper right", fontsize=15, ncol=2, frameon=True)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$R_{\mathrm{CCSN}} \, \mathrm{[yr^{-1}\,Mpc^{-3}]}$")
plt.savefig("../plots/R_cc_split.pdf", bbox_inches="tight")
