import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['cmr']})
rc('font',**{'family':'serif','serif':['cmr']})
rc('font', size=18)

Energies = np.linspace(1, 80, 159)
lowns = sn.FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_z9_6_ls220.dat')
hins = sn.FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_s27_ls220.dat')
bh = sn.FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_BH_s40c.dat')

plt.figure(figsize=(7,5))
fig, ax = plt.subplots()
plt.semilogy(Energies, lowns(Energies), label=r'NS ($10\mathrm{\,M}_{\odot}$)')
plt.semilogy(Energies, hins(Energies), ls='--', label=r'NS ($27\mathrm{\,M}_{\odot}$)')
plt.semilogy(Energies, bh(Energies), ls=':', label=r'BH ($40\mathrm{\,M}_{\odot}$)')

plt.legend(frameon=False)
plt.xlabel(r"Energy [MeV]")
ax.yaxis.set_ticks_position('both')
ax.yaxis.set_tick_params(direction='in', which='both')
plt.ylabel(r"$\mathrm{d}N_{\nu_{\bar{e}}}/\mathrm{d}E, \, [\mathrm{MeV}^{-1}]$")
plt.xlim(1,80)
plt.ylim(1e53,1e57)
plt.savefig('../Plots/Fitting_spectrum.pdf', bbox_inches='tight')