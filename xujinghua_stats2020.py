import matplotlib.pyplot as plt

import numpy as np

time = np.arange(0, 10, 0.1)



plt.plot(np.sin(time))

plt.plot(np.sin(2*time))
plt.plot(np.sin(2*time))
plt.plot(np.sin(2*time))

plt.plot(2*np.sin(2*time))