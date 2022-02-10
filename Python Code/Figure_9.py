import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

E = np.linspace(1, 50, 100) * sn.MeV

DSNB_spiral = sn.DSNB_galaxy(E, gtype="spiral", SFR2alpha=2.6)
DSNB_starburst = sn.DSNB_galaxy(E, gtype="starburst", SFR2alpha=2.6)
DSNB_AGNspiral = sn.DSNB_galaxy(E, gtype="AGN spiral", SFR2alpha=2.6)
DSNB_AGNstarburst = sn.DSNB_galaxy(E, gtype="AGN starburst", SFR2alpha=2.6)
DSNB = DSNB_spiral + DSNB_starburst + DSNB_AGNspiral + DSNB_AGNstarburst

DSNB_spiral_sal = sn.DSNB_galaxy(E, gtype="spiral", salpeter=True)
DSNB_starburst_sal = sn.DSNB_galaxy(E, gtype="starburst", salpeter=True)
DSNB_AGNspiral_sal = sn.DSNB_galaxy(E, gtype="AGN spiral", salpeter=True)
DSNB_AGNstarburst_sal = sn.DSNB_galaxy(E, gtype="AGN starburst", salpeter=True)
DSNB_salpeter = (
    DSNB_spiral_sal + DSNB_starburst_sal + DSNB_AGNspiral_sal + DSNB_AGNstarburst_sal
)

KamLAND_E, KamLAND_lim = np.loadtxt("../data/KamLAND.txt", unpack=True)
SK_E, SK_lim = np.loadtxt("../data/SK-I_II_III.txt", unpack=True)
SK4_E, SK4_lim = np.loadtxt("../data/SK-IV.txt", unpack=True)

plt.figure(figsize=(7, 5))
fig, ax = plt.subplots()

plt.scatter(SK_E, SK_lim, label=r"SK-I/II/III", marker="x", color="k")
plt.scatter(SK4_E, SK4_lim, label=r"SK-IV", marker="o", color="k")
plt.axvspan(9.3, 31.3, alpha=0.2, color="C3", label="SK Signal Region")

plt.plot(E, DSNB, "-", color="C0", label="Varying IMF")
plt.plot(E, DSNB_salpeter, "--", color="C2", label="Salpeter IMF")

plt.yscale("log")
plt.xlim(0, 50)
plt.ylim(1e-3, 100e0)
plt.legend(loc="upper right", frameon=True, fontsize=15)
plt.xlabel(r"$E \, \mathrm{[MeV]}$")
plt.ylabel(r"$\Phi_{\bar{\nu_e}} \, \mathrm{[cm^{-2} \, s^{-1} \, MeV^{-1}]} $")
plt.savefig("../plots/DSNB.pdf", bbox_inches="tight")

E = np.linspace(1, 50, 100) * sn.MeV

DSNB_spiral0 = sn.DSNB_galaxy(E, gtype="spiral", SFR2alpha=2.6, zmin=0.0, zmax=0.2)
DSNB_starburst0 = sn.DSNB_galaxy(
    E, gtype="starburst", SFR2alpha=2.6, zmin=0.0, zmax=0.2
)
DSNB_AGNspiral0 = sn.DSNB_galaxy(
    E, gtype="AGN spiral", SFR2alpha=2.6, zmin=0.0, zmax=0.2
)
DSNB_AGNstarburst0 = sn.DSNB_galaxy(
    E, gtype="AGN starburst", SFR2alpha=2.6, zmin=0.0, zmax=0.2
)
DSNB0 = DSNB_spiral0 + DSNB_starburst0 + DSNB_AGNspiral0 + DSNB_AGNstarburst0

DSNB_spiral1 = sn.DSNB_galaxy(E, gtype="spiral", SFR2alpha=2.6, zmin=0.2, zmax=0.5)
DSNB_starburst1 = sn.DSNB_galaxy(
    E, gtype="starburst", SFR2alpha=2.6, zmin=0.2, zmax=0.5
)
DSNB_AGNspiral1 = sn.DSNB_galaxy(
    E, gtype="AGN spiral", SFR2alpha=2.6, zmin=0.2, zmax=0.5
)
DSNB_AGNstarburst1 = sn.DSNB_galaxy(
    E, gtype="AGN starburst", SFR2alpha=2.6, zmin=0.2, zmax=0.5
)
DSNB1 = DSNB_spiral1 + DSNB_starburst1 + DSNB_AGNspiral1 + DSNB_AGNstarburst1

DSNB_spiral2 = sn.DSNB_galaxy(E, gtype="spiral", SFR2alpha=2.6, zmin=0.5, zmax=1.0)
DSNB_starburst2 = sn.DSNB_galaxy(
    E, gtype="starburst", SFR2alpha=2.6, zmin=0.5, zmax=1.0
)
DSNB_AGNspiral2 = sn.DSNB_galaxy(
    E, gtype="AGN spiral", SFR2alpha=2.6, zmin=0.5, zmax=1.0
)
DSNB_AGNstarburst2 = sn.DSNB_galaxy(
    E, gtype="AGN starburst", SFR2alpha=2.6, zmin=0.5, zmax=1.0
)
DSNB2 = DSNB_spiral2 + DSNB_starburst2 + DSNB_AGNspiral2 + DSNB_AGNstarburst2

DSNB_spiral3 = sn.DSNB_galaxy(E, gtype="spiral", SFR2alpha=2.6, zmin=1.0, zmax=5.0)
DSNB_starburst3 = sn.DSNB_galaxy(
    E, gtype="starburst", SFR2alpha=2.6, zmin=1.0, zmax=5.0
)
DSNB_AGNspiral3 = sn.DSNB_galaxy(
    E, gtype="AGN spiral", SFR2alpha=2.6, zmin=1.0, zmax=5.0
)
DSNB_AGNstarburst3 = sn.DSNB_galaxy(
    E, gtype="AGN starburst", SFR2alpha=2.6, zmin=1.0, zmax=5.0
)
DSNB3 = DSNB_spiral3 + DSNB_starburst3 + DSNB_AGNspiral3 + DSNB_AGNstarburst3

plt.figure(figsize=(7, 5))
fig, ax = plt.subplots()

plt.axvspan(9.3, 31.3, alpha=0.2, color="C3", label="SK Signal Region")

plasma = matplotlib.cm.get_cmap("plasma")

plt.plot(E, DSNB0, "-", color=plasma(0.8), label="z=0-0.2")
plt.plot(E, DSNB1, "-", color=plasma(0.6), label="z=0.2-0.5")
plt.plot(E, DSNB2, "-", color=plasma(0.4), label="z=0.5-1.0")
plt.plot(E, DSNB3, "-", color=plasma(0.2), label="z=1.0-5.0")

plt.yscale("log")
plt.xlim(0, 50)
plt.ylim(1e-3, 10e0)
plt.legend(loc="upper right", frameon=True)
plt.xlabel(r"$E \, \mathrm{[MeV]}$")
plt.ylabel(r"$\Phi_{\bar{\nu_e}} \, \mathrm{[cm^{-2} \, s^{-1} \, MeV^{-1}]} $")
plt.savefig("../plots/DSNB_z.pdf", bbox_inches="tight")
