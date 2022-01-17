import numpy as np
import matplotlib.pyplot as plt
from random import *
from scipy.integrate import quad, simps
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os

##################################################################
#
# Here we are going to re write the signal calculation relaxing both
# assumnptions: universal IMF and universal spectrum
#
##################################################################

#Units:
Mpc = 1.
Lsun = 1.
cm = Mpc/3.08567758e24
km = 1.e5*cm
MeV = 1.
Msun = 1.
sec = 1.
yr = 3.154e7*sec
c = 2.9979e10*cm/sec

#Array lengths:
N_L = 200           #length of luminosity array
N_z = 51            #length of redshift array
N_M = 52            #length of mass array



###################################
def calibfactorfunc(calibdata, IMF_slopes = [-2.0, -2.1, -2.2, -2.3, -2.35], calibration_age = 100.0):
    #calibfactorfunc reads calibration data from file calibdata, and produces an interpolation of calibration factors from the slope of the IMF, calibration_factor(alpha). calibdata is a file containing calibration factor data on star forming galaxies. Column 0 consists of the age of a galaxy after the onset of star formation, and each subsequent column contains the calibration factor star_formation_rate/luminosity at those times, in units of (Msun/yr)/(erg/s). IMF_slopes is a list of the exponents of the initial mass function of stars in a star forming galaxy, used to generate the calibration factor data. The Salpeter IMF would correspond to IMF exponent of -2.35. calibration_age is the age at which the interpolation function will calculate the calibration factor for a given IMF exponent, defaulting to 100 Myr.
    
    Solar_L = 3.82800e33 # erg/s
    
    #Load calibration factor data from file. File must contain age of star forming galaxy in column 0, and calibration factors (star formation rate/ luminosity) in subsequent columns. Number of columns of calibration factors must match length of 'IMF_slopes' array.
    sp = np.genfromtxt(calibdata, skip_header = 1, autostrip=True, unpack = False)
    
    #Calculate array of calibration factors for each IMF
    if len(np.where(sp[:,0] == calibration_age))==1:
        #If the ages array contains the age at which to calculate calibration factor
        calib = sp[sp[:,0]==calibration_age, 1:][0]
    elif len(np.where(sp[:,0]== calibration_age)) < 1:
        #If the ages array does not include the age at which to calculate the calibration factor, but straddles it
        low_age = sp[:,0][sp[:,0]<calibration_age][-1]
        hi_age = sp[:,0][sp[:,0]>calibration_age][0]
        x = (calibration_age - low_age)/(hi_age - low_age)
        calib = x*sp[sp[:,0]==low_age, 1:][0] + (1-x)*sp[sp[:,0]==hi_age, 1:][0]
    else:
        print('Calibration data file must contain no more than one calibration factor per age')
        return None
    
    calib *= Solar_L #now in M_sun/yr/L_sun

    #Generate the interpolation function
    if len(IMF_slopes) == len(calib):
        return interp1d(IMF_slopes, calib, bounds_error=False, fill_value='extrapolate')
    else:
        print('Length of IMF_slopes must be number of columns in calibdata - 1')
        return None
        

def Hinv(z, h=0.678, omegaM=0.308, omegaL=1-0.308):
    # Hinv calculates the inverse of the Hubble constant (H^-1) at redshift z
    # h is the reduced Hubble constant H_0/(100 km s^-1 Mpc^-1)
    # OmegaM is the matter energy density
    # OmegaL is the cosmological constant energy density
    # Hinv gives accurate results only in cosmological constant and matter dominated phases of the universe, not radiation dominated
    
    H0 = 100.*h*km/sec/Mpc # convert H0 to s^-1 i.e. converting km to Mpc
    H = H0*(omegaM*(1+z)**3 + omegaL)**0.5
    return 1/H

def FermiDirac(E, Norm, a, Eav):
    #FermiDirac defines a pinched Fermi-Dirac spectrum (dN/dE (E)).
    
    return Norm*(1+a)**(1+a)*E**a*np.exp(-1*(1+a)*E/Eav)/gamma(1+a)/(Eav**(2+a))

def FermiDiracfitting(fluxdata, initvals = [1e56, 10, 4]):
    #FermiDiracfitting fits data in file fluxdata to a pinched Fermi-Dirac spectrum (FermiDirac). The data in fluxdata consists of two columns: column 0 consists of energy in units of MeV, and column 1 consists of corresponding values of dN/dE in units of MeV^-1. initvals is a 3-element list that provides the initial guess of the parameters in the FermiDirac function used in the curve-fitting process. Returns a lambda function, which can take an energy in MeV as an argument and return dN/dE in MeV^-1.
    
    Energies, nu_ebar = np.loadtxt(fluxdata, unpack=True)
    params, cov = curve_fit(FermiDirac, Energies, nu_ebar, p0=initvals)
    return lambda E: FermiDirac(E, params[0], params[1], params[2])
    
#######################################################
    
def lum2alpha(LIR, SFR2alpha=2.6):
    #lum2alpha calculates the exponent of the IMF in a galaxy with infrared luminosity LIR, assuming that alpha is a function of star forming rate (and consequently luminosity) as described in [1104.2379]. The factor SFR2alpha is a fitting parameter from 1104.2379. 
    
    alpha = 0.36*np.log10(LIR) - SFR2alpha - 3.512588
    #The factor -3.512588 is 0.36*log10(calibration factor) where the calibration factor is the value of (star formation rate/infrared luminosity) determined in Kennicutt98.
    alpha[alpha < -2.35] = -2.35
    alpha[alpha > -1.8] = -1.8
    return alpha

def IMF(M, LIR, usesalpeter=False, SFR2alpha=2.6, usekinkednorm=True, Mkink=0.5, alphalow=-1.3, Mhi=125, Mlo=0.1):
    #IMF calculates the probability distribution function of stars as they are created onto the main sequence as a function of stellar mass M, normalized to the total number of stars formed between Mlo and Mhi Msun. M and LIR may be scalars or 1D arrays. LIR is the infrared luminosity of the galaxy and SFR2alpha a fitting parameter used to calculate the exponent of the IMF (as per lum2alpha). If usesalpeter is set to True, the exponent of the IMF is fixed at -2.35, regardless of the galaxy luminosity. To calculate the normalization, either a kinked (if usekinkednorm is True) or unkinked (if usekinkednorm is False). In the kinked case, the IMF used for the normalization is fixed at alphalow in the range Mlo to Mkink Msun, and behaves as described above in the range Mkink to Mhi Msun. Returns an array with dimensions (len(LIR), len(M)).
    
    #Calculation of IMF exponent
    if usesalpeter:
        a = -2.35 * np.ones_like(LIR).reshape(-1,1)
    else:
        a = lum2alpha(LIR, SFR2alpha).reshape(-1,1)
    
    #Calculation of normalization factor: integral of M^(1+a) from Mlo to Mhi Msun
    if usekinkednorm:
        IMFint = ((Mhi**(2+a) - Mkink**(2+a))/(2+a)  + (Mkink**(2+alphalow) - Mlo**(2+alphalow))/(2+alphalow)) * Msun
    else:
        IMFint = ((Mhi**(2+a) - Mlow**(2+a))/(2+a)) * Msun
    
    #Calculation of unnormalized IMF
    IMF = (M/Msun)**a * Msun**-1
    
    return IMF/IMFint

def RSF(LIR, usesalpeter=False, SFR2alpha=2.6, gtype='spiral'):
    #RSF calculates the star formation rate of a galaxy with infrared luminosity LIR. If usesalpeter is True, a fixed Salpeter IMF (IMF proportional to M^-2.35) is used, otherwise the exponent in the IMF will depend on the luminosity of the galaxy. See [1104.2379] for more information. SFR2alpha is a parameter used in lum2alpha to calculate the IMF exponent from the luminosity. gtype describes the galaxy morphology, and may be one of the following: 'spiral', 'starburst', 'AGN starburst', or 'AGN spiral'.
    # LIR may be a scalar or an array, and has units of Lsun
    # The relationship between luminosity and star formation rate is taken from eqn 3 of https://arxiv.org/pdf/astro-ph/9712213.pdf, but with a calibration factor that is calculated using Pegase.3.0.1 galaxy simulations.
    # returns in Msun yr^-1
    
    #Calculate the conversion functions from IMF exponent to calibration factor
    IMF_to_calibration = calibfactorfunc('../Data/Calibration/SFcalibration_spiral.dat')
    IMF_to_calibration_SB = calibfactorfunc('../Data/Calibration/SFcalibration_starburst.dat')

    #Assuming fixed Salpeter IMF: IMF \propto M^{-2.35}
    if usesalpeter:
        #By default, calibration factors are assumed to be different between starburst and spiral galaxy morphologies.
        if gtype == 'starburst' or gtype == 'AGN starburst':
            calib = IMF_to_calibration_SB(-2.35)
        else:
            calib = IMF_to_calibration(-2.35)
        return LIR*(calib/Lsun)*Msun/yr 
    
    #Assuming varying IMF
    else:
        a = lum2alpha(LIR, SFR2alpha) #calculate IMF exponent from luminosity
        #By default calibration factors are assumed to be different between starburst and spiral galaxy morphologies.
        if gtype == 'starburst' or gtype == 'AGN starburst':
            calib = IMF_to_calibration_SB(a)
        else:
            calib = IMF_to_calibration(a)
        return LIR*(calib/Lsun)*Msun/yr
    
def RSF_density(z, gtype='spiral', usesalpeter=False, logLmin=8.0, logLmax=14.0, logLsteps=200, Mmin=8.0, Mmax=125.0, Msteps=52):
    #RSF_density calculates the star formation rate density in a region of the universe at redshift z, potentially including multiple galaxies. gtype describes the galaxy morphology, which can be one of the following: 'spiral', 'starburst', 'AGN starburst', or 'AGN spiral'. Alternatively, gtype can take on the value 'total' to describe an average local sampling of galaxy morphologies. Setting usesalpeter to True fixes the IMF in all galaxies to be the Salpeter IMF (IMF proportional to M^-2.35). Otherwise, the exponent of the IMF is allowed to depend on luminosity, according to the relationship in [1104.2379]. logLmin, logLmax, and logLsteps describe the range of luminosities to be considered. Likewise, Mmin, Mmax, and Msteps describe the range of stellar masses to consider. The default values are set to those stars expected to undergo core collapse supernovae.
    
    #Selecting the appropriate luminosity functions to use based on galaxy morphology
    if gtype == 'spiral':
        Phi = Phi_Spiral
    elif gtype == 'starburst':
        Phi = Phi_Starburst
    elif gtype == 'AGN starburst':
        Phi = lambda Ltemp, ztemp: fSBAGN(ztemp).reshape(-1,1)*Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'AGN spiral':
        Phi = lambda Ltemp, ztemp: (1-fSBAGN(ztemp)).reshape(-1,1)*Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'total':
        Phi = Phi_tot
    else:
        print('Please enter correct galaxy type - spiral, starburst, AGN spiral, or AGN starburst')

    #set the arrays of luminosity and mass.
    logL = np.linspace(logLmin, logLmax, logLsteps)
    L = 10.**logL*Lsun
    M = np.linspace(Mmin, Mmax, Msteps)*Msun
    
    #calculate the star formation rate in an individual galaxy with luminosity L
    R = RSF(L,usesalpeter=usesalpeter, gtype=gtype)
    
    #Integrate the star formation rate over a collection of galaxies of different luminosities, weighting according to the appropriate luminosity function Phi. 
    rhoSF = simps(R*Phi(L,z),x=logL)
    
    return rhoSF/(1./yr/Mpc**3)

def RCC_density(z, gtype='spiral', salpeter=False, logLmin=8.0, logLmax=14.0, logLsteps=200, Mmin=8.0, Mmax=125.0, Msteps=52):
    #RCC_density calculates the core collapse rate density in a region of the universe at redshift z, potentially including multiple galaxies. gtype describes the galaxy morphology, which can be one of the following: 'spiral', 'starburst', 'AGN starburst', or 'AGN spiral'. Alternatively, gtype can take on the value 'total' to describe an average local sampling of galaxy morphologies. Setting usesalpeter to True fixes the IMF in all galaxies to be the Salpeter IMF (IMF proportional to M^-2.35). Otherwise, the exponent of the IMF is allowed to depend on luminosity, according to the relationship in [1104.2379]. logLmin, logLmax, and logLsteps describe the range of luminosities to be considered. Likewise, Mmin, Mmax, and Msteps describe the range of stellar masses to consider. The default values are set to those stars expected to undergo core collapse supenovae.
    
    #Selecting the appropriate luminosity functions to use based on galaxy morphology
    if gtype == 'spiral':
        Phi = Phi_Spiral
    elif gtype == 'starburst':
        Phi = Phi_Starburst
    elif gtype == 'AGN starburst':
        Phi = lambda Ltemp, ztemp: fSBAGN(ztemp).reshape(-1,1)*Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'AGN spiral':
        Phi = lambda Ltemp, ztemp: (1-fSBAGN(ztemp)).reshape(-1,1)*Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'total':
        Phi = Phi_tot
    else:
        print('Please enter correct galaxy type - spiral, starburst, AGN spiral, or AGN starburst')

    #set the arrays of luminosity and mass.
    logL = np.linspace(logLmin,logLmax,logLsteps)
    L = 10.**logL*Lsun
    M = np.linspace(Mmin,Mmax,Msteps)*Msun
    
    #Calculate the star formation rate in an individual galaxy with luminosity L
    R = RSF(L,usesalpeter=salpeter, gtype=gtype)
    
    #Calculate the fraction of stars that undergo core collapse
    IMFfactor = simps(IMF(M,L,salpeter),x=M)
    
    #Integrate over the galaxies in a region of space, weighting by the appropriate luminosity function Phi
    rhoCC = simps(R*Phi(L,z)*IMFfactor,x=logL)
    
    return rhoCC/(1./yr/Mpc**3)

def DSNB_galaxy(E, gtype='spiral', salpeter=False, SFR2alpha=2.6, 
    Mthresh = np.array([15, 22, 25, 27]), logLmin=8.0, logLmax=14.0, logLsteps=200, Mmin=8.0, Mmax=125.0, Msteps=52, zmin=0.0, zmax=5.0, zsteps=51):
    #DSNB_galaxy calculates the diffuse supernova neutrino background as a function of the neutrino energy E.  gtype describes the galaxy morphology, which can be one of the following: 'spiral', 'starburst', 'AGN starburst', or 'AGN spiral'. Alternatively, gtype can take on the value 'total' to describe an average local sampling of galaxy morphologies. Setting usesalpeter to True fixes the IMF in all galaxies to be the Salpeter IMF (IMF proportional to M^-2.35). Otherwise, the exponent of the IMF is allowed to depend on luminosity, according to the relationship in [1104.2379]. logLmin, logLmax, and logLsteps describe the range of luminosities to be considered. Likewise, Mmin, Mmax, and Msteps describe the range of stellar masses to consider, and zmin,zmax, and zsteps describe the range of redshifts to consider. The default values are set to those stars expected to undergo core collapse supenovae at redshifts up to 5. Mthresh is an array that describes how to evaluate the neutrino energy spectrum as a function of mass. If the array has length 0, the spectrum will be the energy spectrum of neutrinos that arises from a 27 Msun star core collapsing to a neutron star, over the entire mass range. If Mthresh has length 1, then below Mthresh[0], the energy spectrum is that of a 9.6 Msun star evolving to neutron star, and above Mthresh[0], the energy spectrum is that of a 40 Msun star evolving to a black hole. If Mthresh has length 4, the five regions are: below Mthresh[0]: 9.6 Msun star to neutron star, between Mthresh[0] and Mthresh[1]: 27 Msun star to neutron star, between Mthresh[1] and Mthresh[2]: 40 Msun star to black hole, between Mthresh[2] and Mthresh[3]: 27 Msun star to neutron star, and above Mthresh[3]: 40 Msun star to black hole.
    
    #Selecting the appropriate luminosity functions to use based on galaxy morphology
    if gtype == 'spiral':
        Phi = Phi_Spiral
    elif gtype == 'starburst':
        Phi = Phi_Starburst
    elif gtype == 'AGN starburst':
        Phi = lambda Ltemp, ztemp: fSBAGN(ztemp).reshape(-1,1) * Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'AGN spiral':
        Phi = lambda Ltemp, ztemp: (1-fSBAGN(ztemp)).reshape(-1,1) * Phi_SFAGN(Ltemp, ztemp)
    elif gtype == 'total':
        Phi = Phi_tot
    else:
        print('Please enter correct galaxy type - spiral, starburst, AGN spiral, or AGN starburst')
        quit()

    #Set arrays of luminosity, redshift, and mass
    z = np.linspace(zmin, zmax, zsteps)
    logL = np.linspace(logLmin, logLmax, logLsteps)
    L = 10**logL * Lsun
    M = np.linspace(Mmin, Mmax, Msteps)* Msun
    
    #Account for cosmological redshift of neutrinos between source and observation. Eprime has dimensions [len(E), len(z)]. 
    Eprime = E.reshape(-1,1) * (1 + z)
    
    #Calculate neutrino spectrum, as a function of both mass and energy (which is itself a function of redshift)
    #dN/dE(E,M) from individual supernovae types:
    nu_NS_10 = FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_z9_6_ls220.dat') #9.7 Msun
    nu_NS_27 = FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_s27_ls220.dat')  #27 Msun
    nu_BH_40 = FermiDiracfitting('../Data/OSCILLATED_FLUXES/flux_BH_s40c.dat')    #40 Msun

    #Calculate overall neutrino spectrum, which may include different supernovae at different stellar masses.
    # returns in MeV^-1
    # In the FermiDiracfitting functions, E must be a one dimensional array.
    Eshape = np.shape(Eprime)
    Eprime1d = np.ravel(Eprime)
    # M<Mthresh[0]: 9.6 Msun star to neutron star, Mthresh[0]<=M: 40 Msun star to black hole
    if Mthresh.size == 1:
        dNdE = np.where(M.reshape(-1,1,1)<Mthresh[0], nu_NS_10(Eprime1d).reshape(Eshape), nu_BH_40(Eprime1d).reshape(Eshape))
    # M<Mthresh[0]: 9.6 Msun star to neutron star, Mthresh[0]<=M<Mthresh[1]: 27 Msun star to neutron star, Mthresh[1]<=M<Mthresh[2]: 40 Msun star to black hole, Mthresh[2]<=M<Mthresh[3]: 27 Msun star to neutron star, Mthresh[3]<=M: 40 Msun star to black hole
    elif Mthresh.size == 4:
        dNdE = np.where(M.reshape(-1,1,1)<Mthresh[0], nu_NS_10(Eprime1d).reshape(Eshape), np.where(M.reshape(-1,1,1)<Mthresh[1], nu_NS_27(Eprime1d).reshape(Eshape), np.where(M.reshape(-1,1,1) < Mthresh[2], nu_BH_40(Eprime1d).reshape(Eshape), np.where(M.reshape(-1,1,1) < Mthresh[3], nu_NS_27(Eprime1d).reshape(Eshape), nu_BH_40(Eprime1d).reshape(Eshape)))))
    # 27 Msun star to neutron star for all masses
    elif Mthresh.size == 0:
        dNdE = np.where(M.reshape(-1,1,1), nu_NS_27(Eprime1d).reshape(Eshape),0)
    else:
        print('Mthresh must have length 0, 1, or 4')
        return None
    
    #star formation rate as a function of luminosity
    R = RSF(L, usesalpeter=salpeter, SFR2alpha=SFR2alpha, gtype=gtype)
    #Calculate the spectrum of neutrinos weighted by the IMF, and integrate over all possible masses. Essentially this gives an average number of neutrinos per energy range produced from an arbitrary solar mass per year of stars produced.
    IMFint = [IMF(M,L,usesalpeter=salpeter,SFR2alpha=SFR2alpha)[:,i].reshape(-1,1,1) * dNdE[i,:,:] for i in range(len(M))]
    IMFtot = simps(IMFint, x=M, axis=0)
    #Multiply by the actual star formation rate, and the luminosity function of a particular galaxy. This gives the number per energy of neutrinos emitted by a specific galaxy type.
    dL = [(R * Phi(L,z)).T*IMFtot[:,i,:] for i in range(len(E))]

    #Integrate over galaxy luminosities, this calculates the total number per energy of neutrinos emitted by all galaxies of a given type in a given region of space. 
    intdL = simps(dL,x=logL,axis=1)
    #Calculate the number per energy of neutrinos emitted by all galaxies in a spherical shell with redshift z
    dz = intdL*c*Hinv(z)

    #Integrate over z to give the total number per energy of neutrinos emitted by all galaxies within some redshift range
    Nuflux_ebar = simps(dz, x=z, axis=1)
    return Nuflux_ebar/(cm**-2*sec**-1*MeV**-1)



#Luminosity functions: See [1302.5209] for details


def mod_Schechter_func(L, z, alpha, sigma, Phi_star, L_star):
    #mod_Schechter_func defines the general shape of the luminosity function, as a function of the luminosity and redshift. alpha and sigma are parameters which are independent of L or z, while Phi_star and L_star are both z-dependent functions. Each of alpha, sigma, L_star, and Phi_star may differ for different galaxy types.
    
    z = z.reshape(-1,1)
    Phi = Phi_star(z)*np.power(L/L_star(z),1-alpha)*np.exp(-0.5/sigma**2*np.log10(1.0+L/L_star(z))**2)
    return Phi

def Phi_Spiral(L, z):
    #Phi_Spiral is a convenient renaming of mod_Schechter_func that includes all of the parameters for a spiral galaxy.
    
    #create callable functions for Phi_star and L_star
    Phi_star_Spiral = lambda z: Phi_star(z, parameters='spiral')
    L_star_Spiral = lambda z: L_star(z, parameters='spiral')
    return mod_Schechter_func(L,z,alpha=1.0,sigma=0.5,Phi_star=Phi_star_Spiral,L_star=L_star_Spiral)


def Phi_Starburst(L, z):
    #Phi_Starburst is a convenient renaming of mod_Schechter_func that includes all of the parameters for a starburst galaxy.
    
    #create callable functions for Phi_star and L_star
    Phi_star_Starburst = lambda z: Phi_star(z, parameters='starburst')
    L_star_Starburst = lambda z: L_star(z, parameters='starburst')
    return mod_Schechter_func(L,z,alpha=1.0,sigma=0.35,Phi_star=Phi_star_Starburst,L_star=L_star_Starburst)

def Phi_SFAGN(L, z):
    #Phi_SFAGN is a convenient renaming of mod_Schechter_func that includes all of the parameters for a star forming galaxy with an active galactic nucleus.
    
    #create callable functions for Phi_star and L_star
    Phi_star_SFAGN = lambda z: Phi_star(z, parameters='SFAGN')
    L_star_SFAGN = lambda z: L_star(z, parameters='SFAGN')
    return mod_Schechter_func(L,z,alpha=1.2,sigma=0.4,Phi_star=Phi_star_SFAGN,L_star=L_star_SFAGN)

def Phi_tot(L,z):
    #Phi_Spiral is a convenient renaming of mod_Schechter_func that includes all of the parameters for a mix of galaxies that mimics the local universe. Parameters here are determined separately from the individual galaxy types.
    
    #create callable functions for Phi_star and L_star
    Phi_star_tot = lambda z: Phi_star(z, parameters='total')
    L_star_tot = lambda z: L_star(z, parameters='total')
    return mod_Schechter_func(L,z,alpha=1.15,sigma=0.52,Phi_star=Phi_star_tot,L_star=L_star_tot)

def L_star(z, parameters):
    #L_star defines the redshift-dependent parameter L* from [1302.5209]. Parameters is either a 2- or 4- element list, or one of a set of predefined lists. The predefined lists can be called by setting parameters to 'spiral', 'starburst', 'SFAGN', or 'total' for the respective galaxy morphologies. If parameters is a 2-element list, Lstar = 10^('logL0') * (1+z)^'exp', and parameters takes the form (logL0, exp). If parameters is a 4-element list, Lstar is piecewise defined: Lstar = 10^(logL0) * (1+z)^exp1 for z<zk, and Lstar propto (1+z)^exp2 for z>zk, with the proportionality set so that Lstar is a continuous function of z. In this case, parameters takes the form (logL0, exp1, zk, exp2).
    
    #checking parameters input for validity and matching predefined parameters
    if parameters == 'spiral':
        paramlist = (9.58, 4.49, 1.1, 0.0)
    elif parameters == 'starburst':
        paramlist = (11.08, 1.96)
    elif parameters == 'SFAGN':
        paramlist = (10.66, 3.17)
    elif parameters == 'total':
        paramlist = (9.96, 3.55, 1.85, 1.62)
    elif isinstance(parameters, str):
        print("Unrecognized parameters name: accepted values are 'spiral', 'starburst', 'SFAGN', and 'total'.")
        return None
    elif len(parameters)==2 or len(parameters)==4:
        paramlist = parameters
    else:
        print("parameters must be either one of 'spiral', 'starburst', 'SFAGN', or 'total or an iterable with length 2 or 4.")
        return None
    
    #define Lstar using provided parameters, separately for 2- and 4- element lists.
    if len(paramlist)==2:
        return Lsun * (10.0**paramlist[0]) * np.power(1.0+z, paramlist[1])
    else:
        return Lsun * np.where(z<paramlist[2], 10**paramlist[0] * np.power(1.0+z, paramlist[1]), 10**paramlist[0] * (1.0+paramlist[2])**(paramlist[1]-paramlist[3]) * np.power(1.0+z, paramlist[3]))
    
def Phi_star(z, parameters):
    #Phi_star defines the redshift-dependent parameter Phi* from [1302.5209]. parameters is either a 2- or 4- element list, or one of a set of predefined lists. The predefined lists can be called by setting parameters to 'spiral', 'starburst', 'SFAGN', or 'total' for the respective galaxy morphologies. If parameters is a 2-element list, Phistar = 10^('Phi0') * (1+z)^'exp', and parameters takes the form (Phi0, exp). If parameters is a 4-element list, Lstar is piecewise defined: Phistar = 10^(Phi0) * (1+z)^exp1 for z<zk, and Lstar propto (1+z)^exp2 for z>zk, with the proportionality set so that Lstar is a continuous function of z. In this case, parameters takes the form (Phi0, exp1, zk, exp2).
    
    if parameters == 'spiral':
        paramlist = (-2.1, -0.54, 0.53, -7.13)
    elif parameters == 'starburst':
        paramlist = (-4.63, 3.79, 1.1, -1.06)
    elif parameters == 'SFAGN':
        paramlist = (-3.23, 0.67, 1.1, -3.17)
    elif parameters == 'total':
        paramlist = (-2.26, -0.57, 1.1, -3.92)
    elif isinstance(parameters, str):
        print("Unrecognized parameters name: accepted values are 'spiral', 'starburst', 'SFAGN', and 'total'.")
        return None
    elif len(parameters)==2 or len(parameters)==4:
        paramlist = parameters
    else:
        print("parameters must be either one of 'spiral', 'starburst', 'SFAGN', or 'total or an iterable with length 2 or 4.")
        return None
    
    #define Phistar using provided parameters, separately for 2- and 4- element lists.
    if len(paramlist)==2:
        return Mpc**-3 * (10.0**paramlist[0]) * np.power(1.0+z, paramlist[1])
    else:
        return Mpc**-3 * np.where(z<paramlist[2], 10**paramlist[0] * np.power(1.0+z, paramlist[1]), 10**paramlist[0] * (1.0+paramlist[2])**(paramlist[1]-paramlist[3]) * np.power(1.0+z, paramlist[3]))

def fSBAGN(z):
    #fSBAGN calculates the fraction of star forming galaxies with active galactic nuclei that are of starburst type. 
    
    zlow = np.array([0.0,0.3,0.45,0.6,0.8,1.0,1.2,1.7,2.0,2.5,3.0])-1.0e-6
    zhigh = np.append(zlow[1:],5.0+1.0e-6)
    fSBAGN_table = np.array([15,9,10,13,27,68,25,25,81,76,72])*0.01
    z = z.reshape(np.alen(z),1)
    fSBAGN_table = fSBAGN_table*np.ones((np.alen(z),1))
    return fSBAGN_table[(zlow<z)*(z<zhigh)]