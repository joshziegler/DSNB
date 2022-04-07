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

plt.plot(z, rhoSF, "-", color="C0", label="Varying IMF")
plt.plot(z, rhoSF_sal, "--", color="C2", label="Salpeter-like IMF")

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
    r"\noindent Note that all data points assume a Salpeter-like IMF. Changes to \\ this assumption would therefore change the inferred SFRD.",
    horizontalalignment="left",
    fontsize=12,
)

plt.yscale("log")
plt.xlim(0, 3.0)
plt.ylim(1e-2, 3.8e-1)
plt.legend(loc="lower right", fontsize=12, frameon=True)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$R_{\mathrm{SF}} \, \mathrm{[M_\odot\,yr^{-1}\,Mpc^{-3}]}$")
plt.savefig("../plots/R_SF.pdf", bbox_inches="tight")
