import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

M = np.linspace(0.1, 120.0, 150)  # Msun

lum = np.where(
    M < 2.0,
    M ** 4.8,
    np.where(
        M < 20.0,
        2 ** (1.3) * M ** 3.5,
        2 ** (201.0 / 50.0) * 5 ** (34.0 / 25.0) * M ** 2.14,
    ),
)  # Lsun

rad = np.where(M < 10.0, M ** 0.8, 10 ** (9.0 / 20.0) * M ** 0.35)  # Rsun

temp = 5777 * np.where(
    M < 2.0,
    M ** 0.8,
    np.where(
        M < 10.0,
        2 ** (13.0 / 40.0) * M ** 0.475,
        np.where(
            M < 20.0,
            2 ** 0.1 * 5 ** (-9.0 / 40.0) * M ** 0.7,
            2 ** (39.0 / 50.0) * 5 ** (23.0 / 200.0) * M ** 0.36,
        ),
    ),
)  # K

npho = (
    lambda energy, T: 1
    / (np.pi ** 2 * 197.327 ** 3)
    * energy ** 2
    / (np.exp(energy / (8.617e-5 * T)) - 1)
)  # eV^-1 nm^-3

dNdtpho = (
    lambda energy, T, R: np.pi * (R * 6.957e17) ** 2 * 3e17 * npho(energy, T)
)  # eV^-1 s^-1

H0 = 100.0 * 0.678 * sn.km / sn.sec / sn.Mpc
tstar = 11 * M / lum  # Gyr
yr2sec = 3.154e16
Omega_m = 0.308

zd = (
    lambda z: (
        (
            np.tanh(
                (3.0 / 2.0) * H0 * tstar * yr2sec * np.sqrt(1 - Omega_m)
                + np.arctanh((1 + Omega_m / (1 - Omega_m) * (1 + z) ** 3) ** -0.5)
            )
        )
        ** -2
        - 1
    )
    ** (1.0 / 3.0)
    - 1
)

z = np.linspace(0, 5, 50)
E = np.logspace(-1, 2, 100)
eprime = E * (1 + z).reshape(-1, 1)  # eV

spec = [sn.Hinv(z) * dNdtpho(eprime, temp[i], rad[i]).T for i in range(len(M))]

totstarlum = np.zeros((len(M), len(z), len(E)))
for j in range(len(M)):
    for i in range(len(z)):
        zdi = zd(z[i])[j]
        totstarlum[j, i, :] = integ.simps(
            spec[j][:, (z <= z[i]) & (z >= zdi)], x=z[(z <= z[i]) & (z >= zdi)], axis=1
        )

logL = np.linspace(8, 14, 200)
L = 10 ** logL

sfrimf = integ.simps(
    [
        (sn.RSF(L) * sn.IMF(M, L).T)[i, :]
        * (sn.Phi_Spiral(L, z) + sn.Phi_Starburst(L, z) + sn.Phi_SFAGN(L, z))
        for i in range(len(M))
    ],
    x=logL,
    axis=2,
)

sfrimfsal = integ.simps(
    [
        (sn.RSF(L, usesalpeter=True) * sn.IMF(M, L, usesalpeter=True).T)[i, :]
        * (sn.Phi_Spiral(L, z) + sn.Phi_Starburst(L, z) + sn.Phi_SFAGN(L, z))
        for i in range(len(M))
    ],
    x=logL,
    axis=2,
)

integrand = integ.simps(
    [sfrimf * totstarlum[:, :, i] for i in range(len(E))], x=M, axis=1
)
integrandsal = integ.simps(
    [sfrimfsal * totstarlum[:, :, i] for i in range(len(E))], x=M, axis=1
)

ssrd = integ.simps(sn.Hinv(z) * integrand, x=z, axis=1)
ssrdsal = integ.simps(sn.Hinv(z) * integrandsal, x=z, axis=1)

ebl = (
    E ** 2 * sn.c / (4 * np.pi) * 1.60217662e-12 / 3.086e22 ** 2 * ssrd / (2 * np.pi)
)  # erg s^-1 m^-2 sr^-1
eblsal = (
    E ** 2 * sn.c / (4 * np.pi) * 1.60217662e-12 / 3.086e22 ** 2 * ssrdsal / (2 * np.pi)
)  # erg s^-1 m^-2 sr^-1


plt.loglog(1240 / E, ebl, label="Varying IMF", color="C0")
plt.loglog(1240 / E, eblsal, label="Salpeter-like IMF", color="C2", ls="--")

(
    BW_lambda,
    BW_flux,
    BW_lambda_error_minus,
    BW_lambda_error_plus,
    BW_flux_error,
) = np.loadtxt("../data/BW2015.txt", unpack=True)
plt.errorbar(
    BW_lambda,
    BW_flux,
    xerr=[BW_lambda_error_minus, BW_lambda_error_plus],
    yerr=BW_flux_error,
    ls="none",
    label="Biteau, Williams 2015",
    color="C3",
)

plt.xlabel("Wavelength [nm]")
plt.ylabel("$\lambda I_\lambda$ $[\mathrm{erg\,sec^{-1}\,m^{-2}\, sr^{-1}}]$")
plt.axvspan(2700, 7000, alpha=0.2, color="C3")

plt.text(
    (7000 + 2700) / 2,
    10,
    r"\noindent Region where dust",
    color="C3",
    ha="center",
)
plt.text(
    (7000 + 2700) / 2,
    4,
    r"\noindent effects are small",
    color="C3",
    ha="center",
)

plt.legend(loc="lower left", fontsize=14, frameon=True)
plt.xlim(1e2, 1e5)

plt.savefig("../plots/EBL-2.pdf", bbox_inches="tight")
