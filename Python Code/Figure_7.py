import SN_rates as sn
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("plot_style.mplstyle")

z = np.linspace(0.00000001, 4, 1000)

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

h0 = 0.678 * 1.019e-12
OmegaM = 0.308
OmegaL = 1 - OmegaM
t = (
    2
    / (3 * np.sqrt(OmegaL) * 100 * h0)
    * (
        np.arctanh((OmegaM / OmegaL + 1) ** (-1 / 2))
        - np.arctanh((OmegaM / OmegaL * (z + 1) ** 3 + 1) ** (-1 / 2))
    )
)

fftconv = np.fft.irfft(np.fft.rfft(rhoSF) * np.fft.rfft(t ** -1 / 10))
fftconvsal = np.fft.irfft(np.fft.rfft(rhoSF_sal) * np.fft.rfft(t ** -1 / 10))

sn1az, sn1azerr, sn1arate, sn1arateerrminus, sn1arateerrplus = np.loadtxt(
    "../Data/sn1a.txt", unpack=True
)
pz, prate, prateerrminus, prateerrplus = np.loadtxt(
    "../Data/SN_constraints/Perrett_2012.txt", unpack=True
)
cz, czerrlo, czerrhi, crate, crateerr = np.loadtxt(
    "../Data/SN_constraints/Cappellaro_2015.txt", unpack=True
)


plt.figure(figsize=(7, 5))
plt.plot(z, fftconv * 1e4, color="C0", label="Varied IMF")
plt.plot(z, fftconvsal * 1e4, color="C2", ls="--", label="Salpeter-like IMF")
plt.plot(z, fftconv * 2 * 1e4, color="C0", ls="-.", label=r"Varied IMF-DTD$\times$2")
plt.errorbar(
    sn1az,
    sn1arate,
    xerr=sn1azerr,
    yerr=(sn1arateerrminus, sn1arateerrplus),
    color="firebrick",
    ls="none",
    label="Strolger et al. (2020)",
    marker="o",
)
plt.errorbar(
    pz,
    prate,
    xerr=0.05,
    yerr=(prateerrminus, prateerrplus),
    color="darkred",
    ls="none",
    label="Perrett et al. (2012)",
    marker="s",
)
plt.errorbar(
    cz,
    crate,
    xerr=(czerrlo, czerrhi),
    yerr=crateerr,
    color="red",
    ls="none",
    label="Cappellaro et al. (2015)",
    marker="p",
)

plt.xlim(0, 3.0)
plt.ylim(0.0, 1.0)
plt.xlabel(r"Redshift, $z$")
plt.ylabel(r"$R_{\rm SN1a} \, \mathrm{[10^{-4} \, yr^{-1} \, Mpc^{-3}]} $")
plt.legend(loc="lower right", fontsize=14, frameon=True)
plt.savefig("../plots/SN1a.pdf", bbox_inches="tight")
