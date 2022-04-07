import SN_rates as sn
import scipy.integrate as integ
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

z = np.linspace(0, 4, 100)

BHfrac_tot = (
    sn.RCC_density(z, gtype="spiral")
    + sn.RCC_density(z, gtype="starburst")
    + sn.RCC_density(z, gtype="AGN spiral")
    + sn.RCC_density(z, gtype="AGN starburst")
)
BHfrac_sal_tot = (
    sn.RCC_density(z, gtype="spiral", salpeter=True)
    + sn.RCC_density(z, gtype="starburst", salpeter=True)
    + sn.RCC_density(z, gtype="AGN spiral", salpeter=True)
    + sn.RCC_density(z, gtype="AGN starburst", salpeter=True)
)

BHfrac_spiral = sn.RCC_density(
    z, gtype="spiral", Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="spiral", Mmin=27.0)
BHfrac_starburst = sn.RCC_density(
    z, gtype="starburst", Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="starburst", Mmin=27.0)
BHfrac_AGNspiral = sn.RCC_density(
    z, gtype="AGN spiral", Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="AGN spiral", Mmin=27.0)
BHfrac_AGNstarburst = sn.RCC_density(
    z, gtype="AGN starburst", Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="AGN starburst", Mmin=27.0)
BHfrac = (
    BHfrac_spiral + BHfrac_starburst + BHfrac_AGNspiral + BHfrac_AGNstarburst
) / BHfrac_tot

BHfrac_spiral_sal = sn.RCC_density(
    z, gtype="spiral", salpeter=True, Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="spiral", salpeter=True, Mmin=27.0)
BHfrac_starburst_sal = sn.RCC_density(
    z, gtype="starburst", salpeter=True, Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="starburst", salpeter=True, Mmin=27.0)
BHfrac_AGNspiral_sal = sn.RCC_density(
    z, gtype="AGN spiral", salpeter=True, Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="AGN spiral", salpeter=True, Mmin=27.0)
BHfrac_AGNstarburst_sal = sn.RCC_density(
    z, gtype="AGN starburst", salpeter=True, Mmin=22.0, Mmax=25.0
) + sn.RCC_density(z, gtype="AGN starburst", salpeter=True, Mmin=27.0)
BHfrac_sal = (
    BHfrac_spiral_sal
    + BHfrac_starburst_sal
    + BHfrac_AGNspiral_sal
    + BHfrac_AGNstarburst_sal
) / BHfrac_sal_tot

# plt.figure(figsize=(7, 5))

plt.plot(z, BHfrac, "-", color="C0", label="Varying IMF (Total)")
plt.plot(z, BHfrac_sal, "--", color="C2", label="Salpeter-like IMF (Total)")

plt.xlim(0, 3.0)
plt.ylim(0.2, 0.4)
plt.legend(loc="upper left", frameon=True)
plt.xlabel(r"Redshift, $z$")
plt.ylabel("Black Hole Fraction")
plt.savefig("../plots/BHfrac.pdf", bbox_inches="tight")
