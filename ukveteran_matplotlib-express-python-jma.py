import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])

plt.ylabel('some numbers')

plt.show()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')

plt.axis([0, 6, 0, 20])

plt.show()
import numpy as np



# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)



# red dashes, blue squares and green triangles

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

plt.show()
data = {'a': np.arange(50),

        'c': np.random.randint(0, 50, 50),

        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100



plt.scatter('a', 'b', c='c', s='d', data=data)

plt.xlabel('entry a')

plt.ylabel('entry b')

plt.show()
from matplotlib.ticker import NullFormatter  # useful for `logit` scale



# Fixing random state for reproducibility

np.random.seed(19680801)



# make up some data in the interval ]0, 1[

y = np.random.normal(loc=0.5, scale=0.4, size=1000)

y = y[(y > 0) & (y < 1)]

y.sort()

x = np.arange(len(y))



# plot with various axes scales

plt.figure()



# linear

plt.subplot(221)

plt.plot(x, y)

plt.yscale('linear')

plt.title('linear')

plt.grid(True)





# log

plt.subplot(222)

plt.plot(x, y)

plt.yscale('log')

plt.title('log')

plt.grid(True)





# symmetric log

plt.subplot(223)

plt.plot(x, y - y.mean())

plt.yscale('symlog', linthreshy=0.01)

plt.title('symlog')

plt.grid(True)



# logit

plt.subplot(224)

plt.plot(x, y)

plt.yscale('logit')

plt.title('logit')

plt.grid(True)

# Format the minor tick labels of the y-axis into empty strings with

# `NullFormatter`, to avoid cumbering the axis with too many labels.

plt.gca().yaxis.set_minor_formatter(NullFormatter())

# Adjust the subplot layout, because the logit one may take more space

# than usual, due to y-tick labels like "1 - 10^{-3}"

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)



plt.show()
import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.style.use('ggplot')

data = np.random.randn(50)
mpl.rcParams['lines.linewidth'] = 2

mpl.rcParams['lines.color'] = 'r'

plt.plot(data)
mpl.rc('lines', linewidth=4, color='g')

plt.plot(data)
import matplotlib.pyplot as plt

import numpy as np



np.random.seed(19680801)

data = np.random.randn(2, 100)



fig, axs = plt.subplots(2, 2, figsize=(5, 5))

axs[0, 0].hist(data[0])

axs[1, 0].scatter(data[0], data[1])

axs[0, 1].plot(data[0], data[1])

axs[1, 1].hist2d(data[0], data[1])



plt.show()
import matplotlib.patches as mpatches

fig, ax = plt.subplots()

x, y = 10*np.random.rand(2, 1000)

ax.plot(x, y, 'go', alpha=0.2)  # plot some data in data coordinates



circ = mpatches.Circle((0.5, 0.5), 0.25, transform=ax.transAxes,

                       facecolor='blue', alpha=0.75)

ax.add_patch(circ)

plt.show()
import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm



# Default delta is large because that makes it fast, and it illustrates

# the correct registration between image and contours.

delta = 0.5



extent = (-3, 4, -4, 3)



x = np.arange(-3.0, 4.001, delta)

y = np.arange(-4.0, 3.001, delta)

X, Y = np.meshgrid(x, y)

Z1 = np.exp(-X**2 - Y**2)

Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)

Z = (Z1 - Z2) * 2



# Boost the upper limit to avoid truncation errors.

levels = np.arange(-2.0, 1.601, 0.4)



norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())

cmap = cm.PRGn



fig, _axs = plt.subplots(nrows=2, ncols=2)

fig.subplots_adjust(hspace=0.3)

axs = _axs.flatten()



cset1 = axs[0].contourf(X, Y, Z, levels, norm=norm,

                     cmap=cm.get_cmap(cmap, len(levels) - 1))

# It is not necessary, but for the colormap, we need only the

# number of levels minus 1.  To avoid discretization error, use

# either this number or a large number such as the default (256).



# If we want lines as well as filled regions, we need to call

# contour separately; don't try to change the edgecolor or edgewidth

# of the polygons in the collections returned by contourf.

# Use levels output from previous call to guarantee they are the same.



cset2 = axs[0].contour(X, Y, Z, cset1.levels, colors='k')



# We don't really need dashed contour lines to indicate negative

# regions, so let's turn them off.



for c in cset2.collections:

    c.set_linestyle('solid')



# It is easier here to make a separate call to contour than

# to set up an array of colors and linewidths.

# We are making a thick green line as a zero contour.

# Specify the zero level as a tuple with only 0 in it.



cset3 = axs[0].contour(X, Y, Z, (0,), colors='g', linewidths=2)

axs[0].set_title('Filled contours')

fig.colorbar(cset1, ax=axs[0])





axs[1].imshow(Z, extent=extent, cmap=cmap, norm=norm)

axs[1].contour(Z, levels, colors='k', origin='upper', extent=extent)

axs[1].set_title("Image, origin 'upper'")



axs[2].imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)

axs[2].contour(Z, levels, colors='k', origin='lower', extent=extent)

axs[2].set_title("Image, origin 'lower'")



# We will use the interpolation "nearest" here to show the actual

# image pixels.

# Note that the contour lines don't extend to the edge of the box.

# This is intentional. The Z values are defined at the center of each

# image pixel (each color block on the following subplot), so the

# domain that is contoured does not extend beyond these pixel centers.

im = axs[3].imshow(Z, interpolation='nearest', extent=extent,

                cmap=cmap, norm=norm)

axs[3].contour(Z, levels, colors='k', origin='image', extent=extent)

ylim = axs[3].get_ylim()

axs[3].set_ylim(ylim[::-1])

axs[3].set_title("Origin from rc, reversed y-axis")

fig.colorbar(im, ax=axs[3])



fig.tight_layout()

plt.show()
from mpl_toolkits.mplot3d import axes3d

import matplotlib.pyplot as plt





fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



# Grab some test data.

X, Y, Z = axes3d.get_test_data(0.05)



# Plot a basic wireframe.

ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)



plt.show()
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  





def lorenz(x, y, z, s=10, r=28, b=2.667):

    '''

    Given:

       x, y, z: a point of interest in three dimensional space

       s, r, b: parameters defining the lorenz attractor

    Returns:

       x_dot, y_dot, z_dot: values of the lorenz attractor's partial

           derivatives at the point x, y, z

    '''

    x_dot = s*(y - x)

    y_dot = r*x - y - x*z

    z_dot = x*y - b*z

    return x_dot, y_dot, z_dot





dt = 0.01

num_steps = 10000



# Need one more for the initial values

xs = np.empty(num_steps + 1)

ys = np.empty(num_steps + 1)

zs = np.empty(num_steps + 1)



# Set initial values

xs[0], ys[0], zs[0] = (0., 1., 1.05)



# Step through "time", calculating the partial derivatives at the current point

# and using them to estimate the next point

for i in range(num_steps):

    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])

    xs[i + 1] = xs[i] + (x_dot * dt)

    ys[i + 1] = ys[i] + (y_dot * dt)

    zs[i + 1] = zs[i] + (z_dot * dt)





# Plot

fig = plt.figure()

ax = fig.gca(projection='3d')



ax.plot(xs, ys, zs, lw=0.5)

ax.set_xlabel("X Axis")

ax.set_ylabel("Y Axis")

ax.set_zlabel("Z Axis")

ax.set_title("Lorenz Attractor")



plt.show()
import numpy as np





def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):

    X = np.linspace(xmin, xmax, xn).astype(np.float32)

    Y = np.linspace(ymin, ymax, yn).astype(np.float32)

    C = X + Y[:, None] * 1j

    N = np.zeros_like(C, dtype=int)

    Z = np.zeros_like(C)

    for n in range(maxiter):

        I = abs(Z) < horizon

        N[I] = n

        Z[I] = Z[I]**2 + C[I]

    N[N == maxiter-1] = 0

    return Z, N





if __name__ == '__main__':

    import time

    import matplotlib

    from matplotlib import colors

    import matplotlib.pyplot as plt



    xmin, xmax, xn = -2.25, +0.75, 3000 // 2

    ymin, ymax, yn = -1.25, +1.25, 2500 // 2

    maxiter = 200

    horizon = 2.0 ** 40

    log_horizon = np.log2(np.log(horizon))

    Z, N = mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon)



    with np.errstate(invalid='ignore'):

        M = np.nan_to_num(N + 1 - np.log2(np.log(abs(Z))) + log_horizon)



    dpi = 72

    width = 10

    height = 10*yn/xn

    fig = plt.figure(figsize=(width, height), dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)



    # Shaded rendering

    light = colors.LightSource(azdeg=315, altdeg=10)

    M = light.shade(M, cmap=plt.cm.hot, vert_exag=1.5,

                    norm=colors.PowerNorm(0.3), blend_mode='hsv')

    ax.imshow(M, extent=[xmin, xmax, ymin, ymax], interpolation="bicubic")

    ax.set_xticks([])

    ax.set_yticks([])



    plt.show()
import numpy as np

import matplotlib.pyplot as plt



# Fixing random state for reproducibility

np.random.seed(19680801)



n = 100000

x = np.random.standard_normal(n)

y = 2.0 + 3.0 * x + 4.0 * np.random.standard_normal(n)

xmin = x.min()

xmax = x.max()

ymin = y.min()

ymax = y.max()



fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))

fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

ax = axs[0]

hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')

ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

ax.set_title("Hexagon binning")

cb = fig.colorbar(hb, ax=ax)

cb.set_label('counts')



ax = axs[1]

hb = ax.hexbin(x, y, gridsize=50, bins='log', cmap='inferno')

ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

ax.set_title("With a log color scale")

cb = fig.colorbar(hb, ax=ax)

cb.set_label('log10(N)')



plt.show()