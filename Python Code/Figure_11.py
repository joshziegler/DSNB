import SN_rates as sn
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("plot_style.mplstyle")

Energies = np.linspace(1, 80, 159)
lowns = sn.FermiDiracfitting("../Data/OSCILLATED_FLUXES/flux_z9_6_ls220.dat")
hins = sn.FermiDiracfitting("../Data/OSCILLATED_FLUXES/flux_s27_ls220.dat")
bh = sn.FermiDiracfitting("../Data/OSCILLATED_FLUXES/flux_BH_s40c.dat")

fig, ax = plt.subplots()
plt.semilogy(Energies, lowns(Energies), label=r"NS ($10\mathrm{\,M}_{\odot}$)")
plt.semilogy(Energies, hins(Energies), ls="--", label=r"NS ($27\mathrm{\,M}_{\odot}$)")
plt.semilogy(Energies, bh(Energies), ls=":", label=r"BH ($40\mathrm{\,M}_{\odot}$)")

plt.legend(frameon=True)
plt.xlabel(r"Neutrino Energy, E [MeV]")
plt.ylabel(r"$\frac{\mathrm{d}N_{\bar{\nu}_{e}}}{\mathrm{d}E}$ $[\mathrm{MeV}^{-1}]$")
plt.xlim(1, 80)
plt.ylim(1e53, 4e56)
plt.savefig("../Plots/Fitting_spectrum.pdf", bbox_inches="tight")
