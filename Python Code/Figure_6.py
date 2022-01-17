import SN_rates as sn
import scipy.integrate as integ
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['cmr']})
rc('font',**{'family':'serif','serif':['cmr']})
rc('font', size=18)

z = np.linspace(0.00000001,4, 100)

rhoSF_spiral = sn.RSF_density(z,gtype='spiral')
rhoSF_starburst = sn.RSF_density(z,gtype='starburst')
rhoSF_AGNspiral = sn.RSF_density(z,gtype='AGN spiral')
rhoSF_AGNstarburst = sn.RSF_density(z,gtype='AGN starburst')
rhoSF = (rhoSF_spiral+rhoSF_starburst+rhoSF_AGNspiral+rhoSF_AGNstarburst)

rhoSF_spiral_sal = sn.RSF_density(z,gtype='spiral',usesalpeter=True)
rhoSF_starburst_sal = sn.RSF_density(z,gtype='starburst',usesalpeter=True)
rhoSF_AGNspiral_sal = sn.RSF_density(z,gtype='AGN spiral',usesalpeter=True)
rhoSF_AGNstarburst_sal = sn.RSF_density(z,gtype='AGN starburst',usesalpeter=True)
rhoSF_sal = (rhoSF_spiral_sal+rhoSF_starburst_sal+rhoSF_AGNspiral_sal+rhoSF_AGNstarburst_sal)

h0 = 0.678*1.019e-12
OmegaM = 0.308
OmegaL = 1-OmegaM
t = 2/(3*np.sqrt(OmegaL)*100*h0) * (np.arctanh((OmegaM/OmegaL + 1)**(-1/2)) - np.arctanh((OmegaM/OmegaL * (z+1)**3 + 1)**(-1/2)))

fftconv = np.fft.irfft(np.fft.rfft(rhoSF) * np.fft.rfft(t**-1/10))
fftconvsal = np.fft.irfft(np.fft.rfft(rhoSF_sal) * np.fft.rfft(t**-1/10))

sn1az, sn1azerr, sn1arate, sn1arateerrminus, sn1arateerrplus = np.loadtxt('../Data/sn1a.txt', unpack=True)

plt.figure(figsize = (7,5))
plt.plot(z, fftconv*1e4, color = 'blue', label='Varied IMF')
plt.plot(z, fftconvsal*1e4, color = 'green', label='Salpeter IMF')
plt.plot(z, fftconv*2*1e4, color = 'blue', ls = '--', label=r'Varied IMF-DTD$\times$2')
plt.errorbar(sn1az, sn1arate, xerr=sn1azerr, yerr = (sn1arateerrminus, sn1arateerrplus), color = 'black', ls='none', label='Strolger et al. (2020)')

plt.xlabel(r'z')
plt.ylabel(r'$R_{SN1a} \, \mathrm{[10^-4 \, yr^{-1} \, Mpc^{-3}]} $')
plt.legend(loc='lower right', fontsize=13)
plt.savefig('../plots/SN1a.pdf')