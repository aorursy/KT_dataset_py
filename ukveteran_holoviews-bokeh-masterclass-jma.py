import numpy as np

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')
# create some example data

python=np.array([2, 3, 7, 5, 26, 221, 44, 233, 254, 265, 266, 267, 120, 111])

pypy=np.array([12, 33, 47, 15, 126, 121, 144, 233, 254, 225, 226, 267, 110, 130])

jython=np.array([22, 43, 10, 25, 26, 101, 114, 203, 194, 215, 201, 227, 139, 160])



dims = dict(kdims='time', vdims='memory')

python = hv.Area(python, label='python', **dims)

pypy   = hv.Area(pypy,   label='pypy',   **dims)

jython = hv.Area(jython, label='jython', **dims)
opts.defaults(opts.Area(fill_alpha=0.5))

overlay = (python * pypy * jython)

overlay.relabel("Area Chart") + hv.Area.stack(overlay).relabel("Stacked Area Chart")
import holoviews as hv

from holoviews import dim



from  bokeh.sampledata.autompg import autompg

hv.extension('bokeh')
violin = hv.Violin(autompg, ('yr', 'Year'), ('mpg', 'Miles per Gallon')).redim.range(mpg=(8, 45))
violin.opts(height=500, width=900, violin_fill_color=dim('Year').str(), cmap='Set1')
import holoviews as hv

from holoviews import opts

import numpy as np

from matplotlib.path import Path

from matplotlib.transforms import Affine2D

hv.extension('bokeh')
spline=[(0.0,1.0),(0.08,0.98),(0.22,0.82),(0.29,0.72),(0.29,0.72),(0.3,0.64),(0.29,0.57),(0.3,0.5),

(0.3,0.5),(0.34,0.4),(0.43,0.32),(0.5,0.26),(0.5,0.26),(0.58,0.21),(0.66,0.22),(0.76,0.2),(0.76,0.2),

(0.82,0.12),(0.94,0.05),(1.0,0.0),(1.0,0.0),(0.9,0.03),(0.81,0.04),(0.76,0.05),(0.76,0.05),(0.69,0.04),

(0.62,0.04),(0.55,0.04),(0.55,0.04),(0.49,0.1),(0.4,0.17),(0.35,0.2),(0.35,0.2),(0.29,0.24),(0.19,0.28),

(0.14,0.31),(0.14,0.31),(0.09,0.35),(-0.03,0.43),(-0.05,0.72),(-0.05,0.72),(-0.04,0.82),(-0.02,0.95),(0.0,1.0),

(0.1,0.85),(0.14,0.82),(0.18,0.78),(0.18,0.75),(0.18,0.75),(0.16,0.74),(0.14,0.73),(0.12,0.73),(0.12,0.73),

(0.11,0.77),(0.11,0.81),(0.1,0.85),(0.05,0.82),(0.1,0.8),(0.08,0.74),(0.09,0.7),(0.09,0.7),(0.07,0.68),

(0.06,0.66),(0.04,0.67),(0.04,0.67),(0.04,0.73),(0.04,0.81),(0.05,0.82),(0.11,0.7),(0.16,0.56),(0.24,0.39),

(0.3,0.34),(0.3,0.34),(0.41,0.22),(0.62,0.16),(0.8,0.08),(0.23,0.8),(0.35,0.8),(0.44,0.78),(0.5,0.75),

(0.5,0.75),(0.5,0.67),(0.5,0.59),(0.5,0.51),(0.5,0.51),(0.46,0.47),(0.42,0.43),(0.38,0.39),(0.29,0.71),

(0.36,0.74),(0.43,0.73),(0.48,0.69),(0.34,0.61),(0.38,0.66),(0.44,0.64),(0.48,0.63),(0.34,0.51),(0.38,0.56),

(0.41,0.58),(0.48,0.57),(0.45,0.42),(0.46,0.4),(0.47,0.39),(0.48,0.39),(0.42,0.39),(0.43,0.36),(0.46,0.32),

(0.48,0.33),(0.25,0.26),(0.17,0.17),(0.08,0.09),(0.0,0.01),(0.0,0.01),(-0.08,0.09),(-0.17,0.18),(-0.25,0.26),

(-0.25,0.26),(-0.2,0.37),(-0.11,0.47),(-0.03,0.57),(-0.17,0.26),(-0.13,0.34),(-0.08,0.4),(-0.01,0.44),

(-0.12,0.21),(-0.07,0.29),(-0.02,0.34),(0.05,0.4),(-0.06,0.14),(-0.03,0.23),(0.03,0.28),(0.1,0.34),(-0.02,0.08),

(0.02,0.16),(0.09,0.23),(0.16,0.3)]



rotT = Affine2D().rotate_deg(90).translate(1, 0)

rot45T = Affine2D().rotate_deg(45).scale(1. / np.sqrt(2.), 1. / np.sqrt(2.)).translate(1 / 2., 1 / 2.)

flipT = Affine2D().scale(-1, 1).translate(1, 0)



def combine(obj):

    "Collapses overlays of Splines to allow transforms of compositions"

    if not isinstance(obj, hv.Overlay): return obj

    return hv.Spline((np.vstack([el.data[0] for el in obj.values()]),

                      np.hstack([el.data[1] for el in obj.values()])))

    

def T(spline, transform):

    "Apply a transform to a spline or overlay of splines"

    spline = combine(spline)        

    result = Path(spline.data[0], codes=spline.data[1]).transformed(transform)

    return hv.Spline((result.vertices, result.codes))



def beside(spline1, spline2, n=1, m=1):

    den = float(n + m)

    t1 = Affine2D().scale(n / den, 1)

    t2 = Affine2D().scale(m / den, 1).translate(n / den, 0)

    return combine(T(spline1, t1) * T(spline2, t2))



def above(spline1, spline2, n=1, m=1):

    den = float(n + m)

    t1 = Affine2D().scale(1, n / den).translate(0, m / den)

    t2 = Affine2D().scale(1, m / den)

    return combine(T(spline1, t1) * T(spline2, t2))



def nonet(p, q, r, s, t, u, v, w, x):

    return above(beside(p, beside(q, r), 1, 2),

                 above(beside(s, beside(t, u), 1, 2),

                       beside(v, beside(w, x), 1, 2)), 1, 2)



def quartet(p, q, r, s):

    return above(beside(p, q), beside(r, s))



def side(n,t):

    if n == 0: 

        return hv.Spline(([(np.nan, np.nan)],[1]))

    else: 

        return quartet(side(n-1,t), side(n-1,t), rot(t), t)



def corner(n,u,t):

    if n == 0:

        return hv.Spline(([(np.nan, np.nan)],[1]))

    else:

        return quartet(corner(n-1,u,t), side(n-1,t), rot(side(n-1,t)), u)

    

def squarelimit(n,u,t):

    return nonet(corner(n,u,t), side(n,t), rot(rot(rot(corner(n,u,t)))),

                 rot(side(n,t)), u, rot(rot(rot(side(n,t)))), 

                 rot(corner(n,u,t)), rot(rot(side(n,t))), rot(rot(corner(n,u,t))))



def rot(el):        return T(el,rotT)

def rot45(el):      return T(el, rot45T)

def flip(el):       return T(el, flipT)
fish = hv.Spline((spline, [1,4,4,4]*34)) # Cubic splines

smallfish = flip(rot45(fish))

t =  fish *  smallfish * rot(rot(rot(smallfish)))

u = smallfish * rot(smallfish) * rot(rot(smallfish)) * rot(rot(rot(smallfish)))

squarelimit(3,u,t).opts(

    opts.Spline(width=600, height=600, xaxis=None, yaxis=None))
import holoviews as hv

from holoviews import opts

from holoviews.streams import PointDraw

hv.extension('bokeh')
points = hv.Points([(0, 0), (0.5, 0.5), (1, 0)])

point_stream = PointDraw(data=points.columns(), source=points)

trimesh = hv.DynamicMap(hv.TriMesh.from_vertices, streams=[point_stream])
(trimesh * points).opts(

    opts.Points(width=500, height=500, size=10, line_color='black', nonselection_alpha=0.3))