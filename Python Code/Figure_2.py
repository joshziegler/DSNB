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

age, calibsp1, calibsp2, calibsp3, calibsp4, calibsp5 = np.loadtxt('../Data/Calibration/SFcalibration_spiral.dat', unpack=True, skiprows=1)
age, calibsb1, calibsb2, calibsb3, calibsb4, calibsb5 = np.loadtxt('../Data/Calibration/SFcalibration_starburst.dat', unpack=True, skiprows=1)

plt.figure(figsize=(7,5))

plasma = matplotlib.cm.get_cmap('plasma')

plt.semilogx(age, calibsp5, label=r'$\alpha = -2.35$', color = plasma(0.1))
plt.semilogx(age, calibsp4, label=r'$\alpha = -2.3$', color = plasma(0.2))
plt.semilogx(age, calibsp3, label=r'$\alpha = -2.2$', color = plasma(0.4))
plt.semilogx(age, calibsp2, label=r'$\alpha = -2.1$', color = plasma(0.6))
plt.semilogx(age, calibsp1, label=r'$\alpha = -2.0$', color = plasma(0.8))

plt.semilogx(age, calibsb1, ls = '--', color = plasma(0.8))
plt.semilogx(age, calibsb2, ls = '--', color = plasma(0.6))
plt.semilogx(age, calibsb3, ls = '--', color = plasma(0.4))
plt.semilogx(age, calibsb4, ls = '--', color = plasma(0.2))
plt.semilogx(age, calibsb5, ls = '--', color = plasma(0.1))

plt.hlines(4.5e-44, 9, 1000, color = 'black')

plt.semilogx([0,1], [-1,-1], label='Starburst', color='k', ls='--')
plt.semilogx([0,1], [-1,-1], label='Spiral', color='k')

plt.axvline(x=100, color='gray', ls=':')

plt.xlim(9,1000)
plt.ylim(0.15e-43,0.6e-43)
plt.xlabel(r'Age [Myr]')
plt.ylabel(r'Calibration $[\mathrm{M_{\odot}\,yr^{-1}\,erg^{-1}\,s}]$')
plt.legend(fontsize=13, ncol=2, loc='upper right')
plt.savefig('../plots/Calib_alpha.pdf', bbox_inches='tight')

age, calibspa, calibspb, calibspc, calibspd = np.loadtxt('../Data/Calibration/SFcalibration_spiral_2.35.dat', unpack=True, skiprows=1)
age, calibsba, calibsbb, calibsbc, calibsbd = np.loadtxt('../Data/Calibration/SFcalibration_starburst_2.35.dat', unpack=True, skiprows=1)
plt.figure(figsize=(7,5))
plt.semilogx(age, calibspa, label = 'low Z', color = 'C0')
plt.semilogx(age, calibspb, label = 'high Z', color = 'C1')
plt.semilogx(age, calibspc, label = 'dust ZDA', color = 'C2')
plt.semilogx(age, calibspd, label = 'dust LWD', color = 'C3')
plt.semilogx(age, calibsba, ls = '--', color = 'C0')
plt.semilogx(age, calibsbb, ls = '--', color = 'C1')
plt.semilogx(age, calibsbc, ls = '--', color = 'C2')
plt.semilogx(age, calibsbd, ls = '--', color = 'C2')

plt.hlines(4.5e-44, 9, 1000, color = 'black')

plt.semilogx([0,1], [-1,-1], label='Starburst', color='k', ls='--')
plt.semilogx([0,1], [-1,-1], label='Spiral', color='k')

plt.axvline(x=100, color='gray', ls=':')

plt.xlim(9,1000)
plt.ylim(0.15e-43,0.6e-43)
plt.xlabel('Star Formation Duration [Myr]')
plt.ylabel('Calibration $[\mathrm{M_{\odot}\,yr^{-1}\,erg^{-1}\,s}]$')
plt.legend(fontsize=15, ncol=3, loc='lower left')
plt.savefig('../plots/Calib_range.pdf', bbox_inches='tight')