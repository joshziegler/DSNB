import SN_rates as sn
import numpy as np
import scipy.integrate as integ
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use("plot_style.mplstyle")

massrange = np.logspace(-1, 2)
kroupa = np.where(massrange < 0.5, massrange ** -1.3, 0.5 * massrange ** -2.3)
kroupa /= integ.simps(kroupa, x=massrange)
chabrier = np.where(
    massrange < 1,
    0.158
    * (1 / (np.log(10) * massrange))
    * np.exp(-((np.log10(massrange) - np.log10(0.08)) ** 2) / (2 * 0.69 ** 2)),
    0.158
    * (1 / (np.log(10) * 1.0))
    * np.exp(-((np.log10(1.0) - np.log10(0.08)) ** 2) / (2 * 0.69 ** 2))
    * massrange ** -2.3,
)
chabrier /= integ.simps(chabrier, x=massrange)
varying18 = np.where(massrange < 1, massrange ** -1.3, massrange ** -1.8)
varying18 /= integ.simps(varying18, x=massrange)
varying235 = np.where(massrange < 1, massrange ** -1.3, massrange ** -2.35)
varying235 /= integ.simps(varying235, x=massrange)

plt.figure()
# fig, ax = plt.subplots()
plt.loglog(massrange, kroupa, label="Kroupa (2001)", ls=":", color="C2")
plt.loglog(massrange, chabrier, label="Chabrier (2003)", ls="-.", color="C3")
plt.fill_between(
    massrange, varying18, varying235, alpha=0.3, label="Varying IMF", color="C0"
)
plt.loglog(massrange, varying235, label="Salpeter (this work)", ls="-", color="C0")

plt.xlim(1e-1, 1e2)
plt.legend(loc="lower left")
plt.ylabel(r"IMF ($dN/dM$)")
plt.xlabel(r"Mass [$M_\odot$]")
plt.savefig("../plots/IMF.pdf", bbox_inches="tight")
