import matplotlib.pyplot as plt

import numpy as np





fig = plt.figure()

ax = plt.axes()



ax.set_aspect(1)

theta = np.linspace(-np.pi, np.pi, 200)

plt.plot(np.sin(theta), np.cos(theta))



plt.show()