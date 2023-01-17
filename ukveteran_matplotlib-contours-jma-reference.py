import numpy as np

import matplotlib.pyplot as plt

from matplotlib.image import NonUniformImage

from matplotlib import cm



interp = 'nearest'



x = np.linspace(-4, 4, 9)



x2 = x**3



y = np.linspace(-4, 4, 9)



z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)



fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

fig.suptitle('NonUniformImage class', fontsize='large')

ax = axs[0, 0]

im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),

                     cmap=cm.Purples)

im.set_data(x, y, z)

ax.images.append(im)

ax.set_xlim(-4, 4)

ax.set_ylim(-4, 4)

ax.set_title(interp)



ax = axs[0, 1]

im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),

                     cmap=cm.Purples)

im.set_data(x2, y, z)

ax.images.append(im)

ax.set_xlim(-64, 64)

ax.set_ylim(-4, 4)

ax.set_title(interp)



interp = 'bilinear'



ax = axs[1, 0]

im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),

                     cmap=cm.Purples)

im.set_data(x, y, z)

ax.images.append(im)

ax.set_xlim(-4, 4)

ax.set_ylim(-4, 4)

ax.set_title(interp)



ax = axs[1, 1]

im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),

                     cmap=cm.Purples)

im.set_data(x2, y, z)

ax.images.append(im)

ax.set_xlim(-64, 64)

ax.set_ylim(-4, 4)

ax.set_title(interp)



plt.show()
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize





def normal_pdf(x, mean, var):

    return np.exp(-(x - mean)**2 / (2*var))







xmin, xmax, ymin, ymax = (0, 100, 0, 100)

n_bins = 100

xx = np.linspace(xmin, xmax, n_bins)

yy = np.linspace(ymin, ymax, n_bins)



means_high = [20, 50]

means_low = [50, 60]

var = [150, 200]



gauss_x_high = normal_pdf(xx, means_high[0], var[0])

gauss_y_high = normal_pdf(yy, means_high[1], var[0])



gauss_x_low = normal_pdf(xx, means_low[0], var[1])

gauss_y_low = normal_pdf(yy, means_low[1], var[1])



weights_high = np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)

weights_low = -1 * np.array(np.meshgrid(gauss_x_low, gauss_y_low)).prod(0)

weights = weights_high + weights_low





greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)



vmax = np.abs(weights).max()

vmin = -vmax

cmap = plt.cm.RdYlBu



fig, ax = plt.subplots()

ax.imshow(greys)

ax.imshow(weights, extent=(xmin, xmax, ymin, ymax), cmap=cmap)

ax.set_axis_off()
import matplotlib.pyplot as plt

import numpy as np

from numpy import ma

from matplotlib import ticker, cm



N = 100

x = np.linspace(-3.0, 3.0, N)

y = np.linspace(-2.0, 2.0, N)



X, Y = np.meshgrid(x, y)



Z1 = np.exp(-(X)**2 - (Y)**2)

Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)

z = Z1 + 50 * Z2



z[:5, :5] = -1



z = ma.masked_where(z <= 0, z)





fig, ax = plt.subplots()

cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)



cbar = fig.colorbar(cs)



plt.show()