import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import quad as quad, dblquad
from scipy import interpolate

plt.style.use("plot_style.mplstyle")

sec2yr = 365*24*60*60 

def SK_year_bin(flux, time=1, Ee_min=10, Ee_max=30, bin_size=2, eff=0.67, SK=1.5e33):
    Ee = np.linspace(Ee_min-1, Ee_max+1, 200)
    Rate = np.asanyarray([cross_section_v1_Ee(Ee[i], SK, flux) for i in range(len(Ee))])*sec2yr*time*eff
    rint = interpolate.interp1d(Ee, Rate)
    Ebins = np.arange(Ee_min, Ee_max+0.01, bin_size)
    binned_rate = np.asanyarray([quad(rint, Ebins[i], Ebins[i+1])[0] for i in range(len(Ebins)-1)])
    return Ebins, binned_rate


def cross_section_v1(E_nu, mp=938.272, me=0.510, mn=939.565, Delta=1.293):

    Ee = E_nu - Delta
    x = np.log(E_nu)
    pe = np.sqrt(Ee**2 - me**2)
    K = np.exp(-0.07056 * x + 0.02018 * x * x - 0.001953 * x * x * x * x) * 10**(-43) #cm2
    return K * Ee * pe

#Ee - array, Ev - array, flux - interpolation_function
def cross_section_v1_Ee(Ee, Np, flux, mp=938.272, me=0.510, mn=939.565, alpha=1./137., Delta=1.293):

    E_nu = (Ee + Delta) / (1 - Ee/mp)
    Jv = (1. + E_nu/mp)**2 / (1 + Delta/mp)
    return Np * flux(E_nu) * cross_section_v1(E_nu) * Jv 

x, y, z = np.loadtxt("../Data/DSNB.txt", unpack=True)

BGS = np.loadtxt("../Data/BG_article.dat")
E_BGS = np.arange(10, 31, 2)

BINS, RATE_SALPETER = SK_year_bin(interpolate.interp1d(x, z))
BINS, RATE_VAR = SK_year_bin(interpolate.interp1d(x, y))

TOTAL_BG = BGS[0]*10+BGS[1]*10+BGS[2]*10+BGS[3]*10
SK_scale = (22.5/374.)

#fonon=24
#plt.rcParams["figure.subplot.left"] = 0.14
#plt.rcParams["figure.subplot.right"] = 0.99
#plt.rcParams["figure.subplot.bottom"] = 0.14
#plt.rcParams["figure.subplot.top"] = 0.92
#plt.rcParams['axes.labelsize'] = fonon
#plt.rcParams['xtick.labelsize'] = fonon+2
#plt.rcParams['ytick.labelsize'] = fonon+2
#plt.rcParams["figure.figsize"] = (6.8, 6.5)
#fig, ax = plt.subplots(1, 1, sharey=True)

plt.step(BINS[1:-2], TOTAL_BG[1:-2]*SK_scale + RATE_VAR[1:-1]*10, where="post", linestyle='-', label="SIG+BG: Varying IMF")
plt.step(BINS[1:-2], TOTAL_BG[1:-2]*SK_scale + RATE_SALPETER[1:-1]*10, where="post", linestyle='--', color="green", label="SIG+BG: Salpeter IMF")
plt.fill_between(E_BGS, TOTAL_BG*SK_scale, color="gray", linestyle='-', alpha=0.25, step="post", label="BG")

#plt.legend(fontsize=18)
plt.legend(fontsize=15, frameon=True)
plt.errorbar(BINS[1:-3]+1, TOTAL_BG[1:-3]*SK_scale + RATE_VAR[1:-2]*10, np.sqrt(TOTAL_BG[1:-3]*SK_scale + RATE_SALPETER[1:-2]*10), fmt='none', marker='o', color="black", mfc='tab:blue', mec='tab:blue', ms=10, capsize=4, mew=1, linewidth=1)

plt.xlim(10, 30)
plt.ylim(0, 10)
#plt.suptitle("SK with Gd", fontsize=26)
plt.ylabel(r"Event rate [(2 MeV)${}^{-1}$ (10 yr)${}^{-1}$]",labelpad=15)
plt.xlabel(r"$E_{e^+}$ [MeV]")

plt.ylim(0, 10)
plt.xlim(10, 30)
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig("SK_10.pdf")

#fonon=24
#plt.rcParams["figure.subplot.left"] = 0.14
#plt.rcParams["figure.subplot.right"] = 0.99
#plt.rcParams["figure.subplot.bottom"] = 0.14
#plt.rcParams["figure.subplot.top"] = 0.92
#plt.rcParams['axes.labelsize'] = fonon
#plt.rcParams['xtick.labelsize'] = fonon+2
#plt.rcParams['ytick.labelsize'] = fonon+2
#plt.rcParams["figure.figsize"] = (6.8, 6.5)
#fig, ax = plt.subplots(1, 1, sharey=True)

plt.step(BINS[1:-2], TOTAL_BG[1:-2] + RATE_VAR[1:-1]*10*(22.5/374.)**(-1), where="post", linestyle='-', label="SIG+BG: Varying IMF")
plt.step(BINS[1:-2], TOTAL_BG[1:-2] + RATE_SALPETER[1:-1]*10*(22.5/374.)**(-1), where="post", linestyle='--', color="green", label="SIG+BG: Salpeter IMF")
plt.fill_between(E_BGS, TOTAL_BG, color="gray", linestyle='-', alpha=0.25, step="post", label="BG")

#plt.legend(fontsize=18)
plt.legend(fontsize=15, frameon=True)
plt.errorbar(BINS[1:-3]+1, TOTAL_BG[1:-3] + RATE_VAR[1:-2]*10*(22.5/374.)**(-1), np.sqrt(TOTAL_BG[1:-3] + RATE_SALPETER[1:-2]*10*(22.5/374.)**(-1)), fmt='none', marker='o', color="black", mfc='tab:blue', mec='tab:blue', ms=10, capsize=4, mew=1, linewidth=1)

plt.xlim(10, 30)
plt.ylim(0, 10)
#plt.suptitle("HK with Gd", fontsize=26)
plt.ylabel(r"Event rate [(2 MeV)${}^{-1}$ (10 yr)${}^{-1}$]",labelpad=15)
plt.xlabel(r"$E_{e^+}$ [MeV]")

plt.ylim(0, 125)
plt.xlim(10, 30)
#ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
plt.savefig("HK_10.pdf")
