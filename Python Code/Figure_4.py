import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

z = np.linspace(0, 4, 100)

rhoSF_spiral = sn.RSF_density(z, gtype="spiral")
rhoSF_starburst = sn.RSF_density(z, gtype="starburst")
rhoSF_AGNspiral = sn.RSF_density(z, gtype="AGN spiral")
rhoSF_AGNstarburst = sn.RSF_density(z, gtype="AGN starburst")
rhoSF = rhoSF_spiral + rhoSF_starburst + rhoSF_AGNspiral + rhoSF_AGNstarburst

rhoSF_spiral_sal = sn.RSF_density(z, gtype="spiral", usesalpeter=True)
rhoSF_starburst_sal = sn.RSF_density(z, gtype="starburst", usesalpeter=True)
rhoSF_AGNspiral_sal = sn.RSF_density(z, gtype="AGN spiral", usesalpeter=True)
rhoSF_AGNstarburst_sal = sn.RSF_density(z, gtype="AGN starburst", usesalpeter=True)
rhoSF_sal = (
    rhoSF_spiral_sal
    + rhoSF_starburst_sal
    + rhoSF_AGNspiral_sal
    + rhoSF_AGNstarburst_sal
)

m11x, m11xer, m11y, m11yerlo, m11yerhi = np.loadtxt(
    "../Data/SN_constraints/Magnelli_2011.txt", unpack=True
)
m13x, m13xer, m13y, m13yerlo, m13yerhi = np.loadtxt(
    "../Data/SN_constraints/Magnelli_2013.txt", unpack=True
)
gx, gxer, gy, gyerlo, gyerhi = np.loadtxt(
    "../Data/SN_constraints/Gruppioni_2013.txt", unpack=True
)


def Magnelli2013LF(L, z):
    z = z.reshape(-1, 1)
    Phiknee = lambda z: np.where(
        z < 1.0, 10 ** -2.57 * (1 + z) ** -1.5, 10 ** -2.03 * (1 + z) ** -3.0
    )
    Lknee = lambda z: np.where(
        z < 1.0, 10 ** 10.48 * (1 + z) ** 3.8, 10 ** 10.31 * (1 + z) ** 4.2
    )
    return np.where(
        L < Lknee(z),
        Phiknee(z) * (L / Lknee(z)) ** -0.6,
        Phiknee(z) * (L / Lknee(z)) ** -2.2,
    )


def Magnelli_RSF_density(
    z,
    gtype="spiral",
    usesalpeter=False,
    logLmin=8.0,
    logLmax=14.0,
    logLsteps=200,
    Mmin=8.0,
    Mmax=125.0,
    Msteps=52,
):
    # set the arrays of luminosity and mass.
    logL = np.linspace(logLmin, logLmax, logLsteps)
    L = 10.0 ** logL * sn.Lsun
    M = np.linspace(Mmin, Mmax, Msteps) * sn.Msun

    # calculate the star formation rate in an individual galaxy with luminosity L
    R = sn.RSF(L, usesalpeter=usesalpeter, gtype="spiral")

    # Integrate the star formation rate over a collection of galaxies of different luminosities, weighting according to the appropriate luminosity function Phi.
    rhoSF = integ.simps(R * Magnelli2013LF(L, z), x=logL)

    return rhoSF / (1.0 / sn.yr / sn.Mpc ** 3)


plt.figure(figsize=(7, 5))
fig, ax = plt.subplots()

plt.errorbar(
    m11x,
    10 ** m11y,
    xerr=m11xer,
    yerr=[10 ** m11y - 10 ** (m11y + m11yerlo), 10 ** (m11y + m11yerhi) - 10 ** m11y],
    label="Magnelli et al. (2011)",
    color="red",
    marker="o",
    ls="none",
)
plt.errorbar(
    m13x,
    10 ** m13y,
    xerr=m13xer,
    yerr=[10 ** m13y - 10 ** (m13y + m13yerlo), 10 ** (m13y + m13yerhi) - 10 ** m13y],
    label="Magnelli et al. (2013)",
    color="red",
    marker="s",
    ls="none",
)
plt.errorbar(
    gx,
    10 ** gy,
    xerr=gxer,
    yerr=[10 ** gy - 10 ** (gy + gyerlo), 10 ** (gy + gyerhi) - 10 ** gy],
    label="Gruppioni et al. (2013)",
    color="darkred",
    marker="p",
    ls="none",
)
# print(z)
# print((rhoSF_sal - rhoSF) / rhoSF_sal)

magsal0 = Magnelli_RSF_density(
    m13x[0], usesalpeter=True, logLmin=9.6, gtype="starburst"
)
mag0 = Magnelli_RSF_density(m13x[0], usesalpeter=False, logLmin=9.6, gtype="starburst")
magsal1 = Magnelli_RSF_density(
    m13x[1], usesalpeter=True, logLmin=10.5, gtype="starburst"
)
mag1 = Magnelli_RSF_density(m13x[1], usesalpeter=False, logLmin=10.5, gtype="starburst")
magsal2 = Magnelli_RSF_density(
    m13x[2], usesalpeter=True, logLmin=10.9, gtype="starburst"
)
mag2 = Magnelli_RSF_density(m13x[2], usesalpeter=False, logLmin=10.9, gtype="starburst")
magsal3 = Magnelli_RSF_density(
    m13x[3], usesalpeter=True, logLmin=11.3, gtype="starburst"
)
mag3 = Magnelli_RSF_density(m13x[3], usesalpeter=False, logLmin=11.3, gtype="starburst")
magsal4 = Magnelli_RSF_density(
    m13x[4], usesalpeter=True, logLmin=11.6, gtype="starburst"
)
mag4 = Magnelli_RSF_density(m13x[4], usesalpeter=False, logLmin=11.6, gtype="starburst")
plt.scatter(
    m13x[0],
    mag0 * (10 ** m13y[0] / magsal0),
    color="blue",
    marker="s",
)
plt.scatter(m13x[1], mag1 * (10 ** m13y[1] / magsal1), color="blue", marker="s")
plt.scatter(m13x[2], mag2 * (10 ** m13y[2] / magsal2), color="blue", marker="s")
plt.scatter(m13x[3], mag3 * (10 ** m13y[3] / magsal3), color="blue", marker="s")
plt.scatter(m13x[4], mag4 * (10 ** m13y[4] / magsal4), color="blue", marker="s")

# plt.arrow(
#     x=m13x[4],
#     y=10 ** m13y[4],
#     dx=0.0,
#     dy=-(10 ** m13y[4] - mag4 * (10 ** m13y[4] / magsal4))[0],
#     length_includes_head=True,
#     head_width=0.05,
#     head_length=0.01,
#     alpha=0.3,
#     ls="--",
#     color="blue",
# )

plt.plot(z, rhoSF, "-", color="C0", label="Varying IMF")
plt.plot(z, rhoSF_sal, "--", color="C2", label="Canonical IMF")

# plt.scatter(
#     m11x,
#     10 ** m11y
#     * [rhoSF[z < m11x[i]][-1] / rhoSF_sal[z < m11x[i]][-1] for i in range(len(m11x))],
#     color="C0",
# )
# plt.scatter(
#     m13x,
#     10 ** m13y
#     * [rhoSF[z < m13x[i]][-1] / rhoSF_sal[z < m13x[i]][-1] for i in range(len(m13x))],
#     color="C0",
# )
# plt.scatter(
#     gx,
#     10 ** gy
#     * [rhoSF[z < gx[i]][-1] / rhoSF_sal[z < gx[i]][-1] for i in range(len(gx))],
#     color="C0",
# )

# plt.arrow(1.75, 0.085, 0, -0.04, length_includes_head=True, color = 'firebrick', width = 0.1, head_width = 0.2, head_length = 0.02, alpha = 0.4)
# plt.text(1.87, 0.048, r'\noindent Observed $R_\mathrm{SF}$ expected\\ to change assuming\\ non-Salpeter IMF', horizontalalignment='left')
plt.text(
    0.08,
    0.3,
    r"\noindent Note that all red data points assume a Salpeter-like IMF. Blue points \\ are scaled using a varying IMF assumption.",
    horizontalalignment="left",
    fontsize=12,
)

plt.yscale("log")
plt.xlim(0, 3.0)
plt.ylim(1e-2, 3.8e-1)
plt.legend(loc="lower right", fontsize=12, frameon=True, columnspacing=1)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$R_{\mathrm{SF}} \, \mathrm{[M_\odot\,yr^{-1}\,Mpc^{-3}]}$")
plt.savefig("../plots/R_SF.pdf", bbox_inches="tight")
