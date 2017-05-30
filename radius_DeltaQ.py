import numpy as np
from matplotlib import pyplot as plt

from utils import get_DeltaQ


r = np.arange(0, 1000)

DeltaE = 80
energy = 9600
WDshift = -4.75e2
DeltaQ = get_DeltaQ(DeltaE, r, energy, WDshift)


fig, ax = plt.subplots()
ax.plot(r, DeltaQ)

ax.set_xlabel('rho')
ax.set_ylabel('\DeltaQ')

plt.show()
