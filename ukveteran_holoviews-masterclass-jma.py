import numpy as np

import holoviews as hv

hv.extension('matplotlib')

hv.output(fig='svg')
n = 20

coords = np.linspace(-1.5, 1.5, n)

X,Y = np.meshgrid(coords, coords);

Qx = np.cos(Y) - np.cos(X)

Qz = np.sin(Y) + np.sin(X)

Z = np.sqrt(X**2 + Y**2)



qmesh = hv.QuadMesh((Qx, Qz, Z))
qmesh
import numpy as np

import holoviews as hv

from holoviews import opts



hv.extension('matplotlib')

hv.output(fig='svg')
# create some example data

python=np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])

pypy=np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])

jython=np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])



dims = dict(kdims='time', vdims='memory')

python = hv.Area(python, label='python', **dims)

pypy   = hv.Area(pypy,   label='pypy',   **dims)

jython = hv.Area(jython, label='jython', **dims)
overlay = (python * pypy * jython).opts(opts.Area(alpha=0.5))

overlay.relabel("Area Chart") + hv.Area.stack(overlay).relabel("Stacked Area Chart")
import holoviews as hv

hv.extension('matplotlib')

hv.output(fig='svg')
from bokeh.sampledata.autompg import autompg as df



title = "MPG by Cylinders and Data Source, Colored by Cylinders"

boxwhisker = hv.BoxWhisker(df, ['cyl', 'origin'], 'mpg', label=title)
boxwhisker.opts(bgcolor='white', aspect=2, fig_size=200)
import numpy as np

import holoviews as hv



hv.extension('matplotlib')
x = np.linspace(0, 4*np.pi, 100)

y = np.sin(x)



scatter1 = hv.Scatter((x, y), label='sin(x)')

scatter2 = hv.Scatter((x, y*2), label='2*sin(x)')

scatter3 = hv.Scatter((x, y*3), label='3*sin(x)')



curve1 = hv.Curve(scatter1)

curve2 = hv.Curve(scatter2)

curve3 = hv.Curve(scatter3)
example1 = scatter1 * scatter2.opts(color='orange') * scatter3.opts(color='green')

example2 = (

    scatter1 * curve1 * curve2.opts(color='orange', linestyle='--') *

    scatter3.opts(color='green', marker='s') * curve3)



example1.relabel("Legend Example") + example2.relabel("Another Legend Example")
import holoviews as hv

from holoviews import opts



hv.extension('matplotlib')
from bokeh.sampledata.iris import flowers

from holoviews.operation import gridmatrix



iris_ds = hv.Dataset(flowers)
density_grid = gridmatrix(iris_ds, diagonal_type=hv.Distribution, chart_type=hv.Bivariate)

point_grid = gridmatrix(iris_ds, chart_type=hv.Points)



(density_grid * point_grid).opts(

    opts.Bivariate(bandwidth=0.5, cmap='Blues'),

    opts.Points(s=4))
import numpy as np

import holoviews as hv

hv.extension('matplotlib')
from scipy.integrate import odeint



sigma = 10

rho = 28

beta = 8.0/3

theta = 3 * np.pi / 4



def lorenz(xyz, t):

    x, y, z = xyz

    x_dot = sigma * (y - x)

    y_dot = x * rho - x * z - y

    z_dot = x * y - beta* z

    return [x_dot, y_dot, z_dot]



initial = (-10, -7, 35)

t = np.arange(0, 100, 0.006)



solution = odeint(lorenz, initial, t)



x = solution[:, 0]

y = solution[:, 1]

z = solution[:, 2]

xprime = np.cos(theta) * x - np.sin(theta) * y



paths = zip(np.array_split(xprime, 7), np.array_split(z, 7))

lorenzian = hv.Path([{('x', 'y'): np.array(d).T, 'index': i}

                     for i, d in enumerate(paths)], vdims='index')
lorenzian.opts(color='index', cmap='Blues', linewidth=1)
import numpy as np

import holoviews as hv

hv.extension('matplotlib')
# Make data.

X = np.arange(-5, 5, 0.25)

Y = np.arange(-5, 5, 0.25)

X, Y = np.meshgrid(X, Y)

R = np.sqrt(X**2 + Y**2)

Z = np.sin(R)



surface = hv.Surface(Z, bounds=(-5, -5, 5, 5))
surface.opts(colorbar=True)
import numpy as np

import holoviews as hv

hv.extension('matplotlib')
n_radii = 8

n_angles = 36



# Make radii and angles spaces (radius r=0 omitted to eliminate duplication).

radii = np.linspace(0.125, 1.0, n_radii)

angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)



# Repeat all angles for each radius.

angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)



# Convert polar (radii, angles) coords to cartesian (x, y) coords.

# (0, 0) is manually added at this stage,  so there will be no duplicate

# points in the (x, y) plane.

x = np.append(0, (radii*np.cos(angles)).flatten())

y = np.append(0, (radii*np.sin(angles)).flatten())



# Compute z to make the pringle surface.

z = np.sin(-x*y)



trisurface = hv.TriSurface((x, y, z))
trisurface.opts(fig_size=200)