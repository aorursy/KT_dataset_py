%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
import os

if not os.path.exists('figures'):

    os.makedirs('figures')
import numpy as np

from matplotlib import pyplot as plt



# Draw a figure and axis with no boundary

fig = plt.figure(figsize=(6, 4.5), facecolor='w')

ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)





def draw_cube(ax, xy, size, depth=0.4,

              edges=None, label=None, label_kwargs=None, **kwargs):

    """draw and label a cube.  edges is a list of numbers between

    1 and 12, specifying which of the 12 cube edges to draw"""

    if edges is None:

        edges = range(1, 13)



    x, y = xy



    if 1 in edges:

        ax.plot([x, x + size],

                [y + size, y + size], **kwargs)

    if 2 in edges:

        ax.plot([x + size, x + size],

                [y, y + size], **kwargs)

    if 3 in edges:

        ax.plot([x, x + size],

                [y, y], **kwargs)

    if 4 in edges:

        ax.plot([x, x],

                [y, y + size], **kwargs)



    if 5 in edges:

        ax.plot([x, x + depth],

                [y + size, y + depth + size], **kwargs)

    if 6 in edges:

        ax.plot([x + size, x + size + depth],

                [y + size, y + depth + size], **kwargs)

    if 7 in edges:

        ax.plot([x + size, x + size + depth],

                [y, y + depth], **kwargs)

    if 8 in edges:

        ax.plot([x, x + depth],

                [y, y + depth], **kwargs)



    if 9 in edges:

        ax.plot([x + depth, x + depth + size],

                [y + depth + size, y + depth + size], **kwargs)

    if 10 in edges:

        ax.plot([x + depth + size, x + depth + size],

                [y + depth, y + depth + size], **kwargs)

    if 11 in edges:

        ax.plot([x + depth, x + depth + size],

                [y + depth, y + depth], **kwargs)

    if 12 in edges:

        ax.plot([x + depth, x + depth],

                [y + depth, y + depth + size], **kwargs)



    if label:

        if label_kwargs is None:

            label_kwargs = {}

        ax.text(x + 0.5 * size, y + 0.5 * size, label,

                ha='center', va='center', **label_kwargs)



solid = dict(c='black', ls='-', lw=1,

             label_kwargs=dict(color='k'))

dotted = dict(c='black', ls='-', lw=0.5, alpha=0.5,

              label_kwargs=dict(color='gray'))

depth = 0.3



# Draw top operation: vector plus scalar

draw_cube(ax, (1, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)

draw_cube(ax, (2, 10), 1, depth, [1, 2, 3, 6, 9], '1', **solid)

draw_cube(ax, (3, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)



draw_cube(ax, (6, 10), 1, depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '5', **solid)

draw_cube(ax, (7, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)

draw_cube(ax, (8, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)



draw_cube(ax, (12, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '5', **solid)

draw_cube(ax, (13, 10), 1, depth, [1, 2, 3, 6, 9], '6', **solid)

draw_cube(ax, (14, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '7', **solid)



ax.text(5, 10.5, '+', size=12, ha='center', va='center')

ax.text(10.5, 10.5, '=', size=12, ha='center', va='center')

ax.text(1, 11.5, r'${\tt np.arange(3) + 5}$',

        size=12, ha='left', va='bottom')



# Draw middle operation: matrix plus vector



# first block

draw_cube(ax, (1, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)

draw_cube(ax, (2, 7.5), 1, depth, [1, 2, 3, 6, 9], '1', **solid)

draw_cube(ax, (3, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '1', **solid)



draw_cube(ax, (1, 6.5), 1, depth, [2, 3, 4], '1', **solid)

draw_cube(ax, (2, 6.5), 1, depth, [2, 3], '1', **solid)

draw_cube(ax, (3, 6.5), 1, depth, [2, 3, 7, 10], '1', **solid)



draw_cube(ax, (1, 5.5), 1, depth, [2, 3, 4], '1', **solid)

draw_cube(ax, (2, 5.5), 1, depth, [2, 3], '1', **solid)

draw_cube(ax, (3, 5.5), 1, depth, [2, 3, 7, 10], '1', **solid)



# second block

draw_cube(ax, (6, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)

draw_cube(ax, (7, 7.5), 1, depth, [1, 2, 3, 6, 9], '1', **solid)

draw_cube(ax, (8, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)



draw_cube(ax, (6, 6.5), 1, depth, range(2, 13), '0', **dotted)

draw_cube(ax, (7, 6.5), 1, depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)

draw_cube(ax, (8, 6.5), 1, depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)



draw_cube(ax, (6, 5.5), 1, depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)

draw_cube(ax, (7, 5.5), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)

draw_cube(ax, (8, 5.5), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)



# third block

draw_cube(ax, (12, 7.5), 1, depth, [1, 2, 3, 4, 5, 6, 9], '1', **solid)

draw_cube(ax, (13, 7.5), 1, depth, [1, 2, 3, 6, 9], '2', **solid)

draw_cube(ax, (14, 7.5), 1, depth, [1, 2, 3, 6, 7, 9, 10], '3', **solid)



draw_cube(ax, (12, 6.5), 1, depth, [2, 3, 4], '1', **solid)

draw_cube(ax, (13, 6.5), 1, depth, [2, 3], '2', **solid)

draw_cube(ax, (14, 6.5), 1, depth, [2, 3, 7, 10], '3', **solid)



draw_cube(ax, (12, 5.5), 1, depth, [2, 3, 4], '1', **solid)

draw_cube(ax, (13, 5.5), 1, depth, [2, 3], '2', **solid)

draw_cube(ax, (14, 5.5), 1, depth, [2, 3, 7, 10], '3', **solid)



ax.text(5, 7.0, '+', size=12, ha='center', va='center')

ax.text(10.5, 7.0, '=', size=12, ha='center', va='center')

ax.text(1, 9.0, r'${\tt np.ones((3,\, 3)) + np.arange(3)}$',

        size=12, ha='left', va='bottom')



# Draw bottom operation: vector plus vector, double broadcast



# first block

draw_cube(ax, (1, 3), 1, depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '0', **solid)

draw_cube(ax, (1, 2), 1, depth, [2, 3, 4, 7, 10], '1', **solid)

draw_cube(ax, (1, 1), 1, depth, [2, 3, 4, 7, 10], '2', **solid)



draw_cube(ax, (2, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)

draw_cube(ax, (2, 2), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)

draw_cube(ax, (2, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)



draw_cube(ax, (3, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '0', **dotted)

draw_cube(ax, (3, 2), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)

draw_cube(ax, (3, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)



# second block

draw_cube(ax, (6, 3), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)

draw_cube(ax, (7, 3), 1, depth, [1, 2, 3, 6, 9], '1', **solid)

draw_cube(ax, (8, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)



draw_cube(ax, (6, 2), 1, depth, range(2, 13), '0', **dotted)

draw_cube(ax, (7, 2), 1, depth, [2, 3, 6, 7, 9, 10, 11], '1', **dotted)

draw_cube(ax, (8, 2), 1, depth, [2, 3, 6, 7, 9, 10, 11], '2', **dotted)



draw_cube(ax, (6, 1), 1, depth, [2, 3, 4, 7, 8, 10, 11, 12], '0', **dotted)

draw_cube(ax, (7, 1), 1, depth, [2, 3, 7, 10, 11], '1', **dotted)

draw_cube(ax, (8, 1), 1, depth, [2, 3, 7, 10, 11], '2', **dotted)



# third block

draw_cube(ax, (12, 3), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)

draw_cube(ax, (13, 3), 1, depth, [1, 2, 3, 6, 9], '1', **solid)

draw_cube(ax, (14, 3), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)



draw_cube(ax, (12, 2), 1, depth, [2, 3, 4], '1', **solid)

draw_cube(ax, (13, 2), 1, depth, [2, 3], '2', **solid)

draw_cube(ax, (14, 2), 1, depth, [2, 3, 7, 10], '3', **solid)



draw_cube(ax, (12, 1), 1, depth, [2, 3, 4], '2', **solid)

draw_cube(ax, (13, 1), 1, depth, [2, 3], '3', **solid)

draw_cube(ax, (14, 1), 1, depth, [2, 3, 7, 10], '4', **solid)



ax.text(5, 2.5, '+', size=12, ha='center', va='center')

ax.text(10.5, 2.5, '=', size=12, ha='center', va='center')

ax.text(1, 4.5, r'${\tt np.arange(3).reshape((3,\, 1)) + np.arange(3)}$',

        ha='left', size=12, va='bottom')



ax.set_xlim(0, 16)

ax.set_ylim(0.5, 12.5)
def draw_dataframe(df, loc=None, width=None, ax=None, linestyle=None,

                   textstyle=None):

    loc = loc or [0, 0]

    width = width or 1



    x, y = loc



    if ax is None:

        ax = plt.gca()



    ncols = len(df.columns) + 1

    nrows = len(df.index) + 1



    dx = dy = width / ncols



    if linestyle is None:

        linestyle = {'color':'black'}



    if textstyle is None:

        textstyle = {'size': 12}



    textstyle.update({'ha':'center', 'va':'center'})



    # draw vertical lines

    for i in range(ncols + 1):

        plt.plot(2 * [x + i * dx], [y, y + dy * nrows], **linestyle)



    # draw horizontal lines

    for i in range(nrows + 1):

        plt.plot([x, x + dx * ncols], 2 * [y + i * dy], **linestyle)



    # Create index labels

    for i in range(nrows - 1):

        plt.text(x + 0.5 * dx, y + (i + 0.5) * dy,

                 str(df.index[::-1][i]), **textstyle)



    # Create column labels

    for i in range(ncols - 1):

        plt.text(x + (i + 1.5) * dx, y + (nrows - 0.5) * dy,

                 str(df.columns[i]), style='italic', **textstyle)

        

    # Add index label

    if df.index.name:

        plt.text(x + 0.5 * dx, y + (nrows - 0.5) * dy,

                 str(df.index.name), style='italic', **textstyle)



    # Insert data

    for i in range(nrows - 1):

        for j in range(ncols - 1):

            plt.text(x + (j + 1.5) * dx,

                     y + (i + 0.5) * dy,

                     str(df.values[::-1][i, j]), **textstyle)





# Draw figure



import pandas as pd

df = pd.DataFrame({'data': [1, 2, 3, 4, 5, 6]},

                   index=['A', 'B', 'C', 'A', 'B', 'C'])

df.index.name = 'key'





fig = plt.figure(figsize=(8, 6), facecolor='white')

ax = plt.axes([0, 0, 1, 1])



ax.axis('off')



draw_dataframe(df, [0, 0])



for y, ind in zip([3, 1, -1], 'ABC'):

    split = df[df.index == ind]

    draw_dataframe(split, [2, y])



    sum = pd.DataFrame(split.sum()).T

    sum.index = [ind]

    sum.index.name = 'key'

    sum.columns = ['data']

    draw_dataframe(sum, [4, y + 0.25])

    

result = df.groupby(df.index).sum()

draw_dataframe(result, [6, 0.75])



style = dict(fontsize=14, ha='center', weight='bold')

plt.text(0.5, 3.6, "Input", **style)

plt.text(2.5, 4.6, "Split", **style)

plt.text(4.5, 4.35, "Apply (sum)", **style)

plt.text(6.5, 2.85, "Combine", **style)



arrowprops = dict(facecolor='black', width=1, headwidth=6)

plt.annotate('', (1.8, 3.6), (1.2, 2.8), arrowprops=arrowprops)

plt.annotate('', (1.8, 1.75), (1.2, 1.75), arrowprops=arrowprops)

plt.annotate('', (1.8, -0.1), (1.2, 0.7), arrowprops=arrowprops)



plt.annotate('', (3.8, 3.8), (3.2, 3.8), arrowprops=arrowprops)

plt.annotate('', (3.8, 1.75), (3.2, 1.75), arrowprops=arrowprops)

plt.annotate('', (3.8, -0.3), (3.2, -0.3), arrowprops=arrowprops)



plt.annotate('', (5.8, 2.8), (5.2, 3.6), arrowprops=arrowprops)

plt.annotate('', (5.8, 1.75), (5.2, 1.75), arrowprops=arrowprops)

plt.annotate('', (5.8, 0.7), (5.2, -0.1), arrowprops=arrowprops)

    

plt.axis('equal')

plt.ylim(-1.5, 5);
fig = plt.figure(figsize=(6, 4))

ax = fig.add_axes([0, 0, 1, 1])

ax.axis('off')

ax.axis('equal')



# Draw features matrix

ax.vlines(range(6), ymin=0, ymax=9, lw=1)

ax.hlines(range(10), xmin=0, xmax=5, lw=1)

font_prop = dict(size=12, family='monospace')

ax.text(-1, -1, "Feature Matrix ($X$)", size=14)

ax.text(0.1, -0.3, r'n_features $\longrightarrow$', **font_prop)

ax.text(-0.1, 0.1, r'$\longleftarrow$ n_samples', rotation=90,

        va='top', ha='right', **font_prop)



# Draw labels vector

ax.vlines(range(8, 10), ymin=0, ymax=9, lw=1)

ax.hlines(range(10), xmin=8, xmax=9, lw=1)

ax.text(7, -1, "Target Vector ($y$)", size=14)

ax.text(7.9, 0.1, r'$\longleftarrow$ n_samples', rotation=90,

        va='top', ha='right', **font_prop)



ax.set_ylim(10, -2)
import numpy as np



def make_data(N=30, err=0.8, rseed=1):

    # randomly sample the data

    rng = np.random.RandomState(rseed)

    X = rng.rand(N, 1) ** 2

    y = 10 - 1. / (X.ravel() + 0.1)

    if err > 0:

        y += err * rng.randn(N)

    return X, y
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline



def PolynomialRegression(degree=2, **kwargs):

    return make_pipeline(PolynomialFeatures(degree),

                         LinearRegression(**kwargs))
X, y = make_data()

xfit = np.linspace(-0.1, 1.0, 1000)[:, None]

model1 = PolynomialRegression(1).fit(X, y)

model20 = PolynomialRegression(20).fit(X, y)



fig, ax = plt.subplots(1, 2, figsize=(16, 6))

fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)



ax[0].scatter(X.ravel(), y, s=40)

ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')

ax[0].axis([-0.1, 1.0, -2, 14])

ax[0].set_title('High-bias model: Underfits the data', size=14)



ax[1].scatter(X.ravel(), y, s=40)

ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')

ax[1].axis([-0.1, 1.0, -2, 14])

ax[1].set_title('High-variance model: Overfits the data', size=14)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)



X2, y2 = make_data(10, rseed=42)



ax[0].scatter(X.ravel(), y, s=40, c='blue')

ax[0].plot(xfit.ravel(), model1.predict(xfit), color='gray')

ax[0].axis([-0.1, 1.0, -2, 14])

ax[0].set_title('High-bias model: Underfits the data', size=14)

ax[0].scatter(X2.ravel(), y2, s=40, c='red')

ax[0].text(0.02, 0.98, "training score: $R^2$ = {0:.2f}".format(model1.score(X, y)),

           ha='left', va='top', transform=ax[0].transAxes, size=14, color='blue')

ax[0].text(0.02, 0.91, "validation score: $R^2$ = {0:.2f}".format(model1.score(X2, y2)),

           ha='left', va='top', transform=ax[0].transAxes, size=14, color='red')



ax[1].scatter(X.ravel(), y, s=40, c='blue')

ax[1].plot(xfit.ravel(), model20.predict(xfit), color='gray')

ax[1].axis([-0.1, 1.0, -2, 14])

ax[1].set_title('High-variance model: Overfits the data', size=14)

ax[1].scatter(X2.ravel(), y2, s=40, c='red')

ax[1].text(0.02, 0.98, "training score: $R^2$ = {0:.2g}".format(model20.score(X, y)),

           ha='left', va='top', transform=ax[1].transAxes, size=14, color='blue')

ax[1].text(0.02, 0.91, "validation score: $R^2$ = {0:.2g}".format(model20.score(X2, y2)),

           ha='left', va='top', transform=ax[1].transAxes, size=14, color='red')
x = np.linspace(0, 1, 1000)

y1 = -(x - 0.5) ** 2

y2 = y1 - 0.33 + np.exp(x - 1)



fig, ax = plt.subplots()

ax.plot(x, y2, lw=10, alpha=0.5, color='blue')

ax.plot(x, y1, lw=10, alpha=0.5, color='red')



ax.text(0.15, 0.2, "training score", rotation=45, size=16, color='blue')

ax.text(0.2, -0.05, "validation score", rotation=20, size=16, color='red')



ax.text(0.02, 0.1, r'$\longleftarrow$ High Bias', size=18, rotation=90, va='center')

ax.text(0.98, 0.1, r'$\longleftarrow$ High Variance $\longrightarrow$', size=18, rotation=90, ha='right', va='center')

ax.text(0.48, -0.12, 'Best$\\longrightarrow$\nModel', size=18, rotation=90, va='center')



ax.set_xlim(0, 1)

ax.set_ylim(-0.3, 0.5)



ax.set_xlabel(r'model complexity $\longrightarrow$', size=14)

ax.set_ylabel(r'model score $\longrightarrow$', size=14)



ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.set_major_formatter(plt.NullFormatter())



ax.set_title("Validation Curve Schematic", size=16)
N = np.linspace(0, 1, 1000)

y1 = 0.75 + 0.2 * np.exp(-4 * N)

y2 = 0.7 - 0.6 * np.exp(-4 * N)



fig, ax = plt.subplots()

ax.plot(x, y1, lw=10, alpha=0.5, color='blue')

ax.plot(x, y2, lw=10, alpha=0.5, color='red')



ax.text(0.2, 0.88, "training score", rotation=-10, size=16, color='blue')

ax.text(0.2, 0.5, "validation score", rotation=30, size=16, color='red')



ax.text(0.98, 0.45, r'Good Fit $\longrightarrow$', size=18, rotation=90, ha='right', va='center')

ax.text(0.02, 0.57, r'$\longleftarrow$ High Variance $\longrightarrow$', size=18, rotation=90, va='center')



ax.set_xlim(0, 1)

ax.set_ylim(0, 1)



ax.set_xlabel(r'training set size $\longrightarrow$', size=14)

ax.set_ylabel(r'model score $\longrightarrow$', size=14)



ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.set_major_formatter(plt.NullFormatter())



ax.set_title("Learning Curve Schematic", size=16)
from sklearn.datasets import make_blobs

X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)



fig, ax = plt.subplots()



ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

ax.set_title('Naive Bayes Model', size=14)



xlim = (-8, 8)

ylim = (-15, 5)



xg = np.linspace(xlim[0], xlim[1], 60)

yg = np.linspace(ylim[0], ylim[1], 40)

xx, yy = np.meshgrid(xg, yg)

Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T



for label, color in enumerate(['red', 'blue']):

    mask = (y == label)

    mu, std = X[mask].mean(0), X[mask].std(0)

    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)

    Pm = np.ma.masked_array(P, P < 0.03)

    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,

                  cmap=color.title() + 's')

    ax.contour(xx, yy, P.reshape(xx.shape),

               levels=[0.01, 0.1, 0.5, 0.9],

               colors=color, alpha=0.2)

    

ax.set(xlim=xlim, ylim=ylim)
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression



from sklearn.base import BaseEstimator, TransformerMixin



class GaussianFeatures(BaseEstimator, TransformerMixin):

    """Uniformly-spaced Gaussian Features for 1D input"""

    

    def __init__(self, N, width_factor=2.0):

        self.N = N

        self.width_factor = width_factor

    

    @staticmethod

    def _gauss_basis(x, y, width, axis=None):

        arg = (x - y) / width

        return np.exp(-0.5 * np.sum(arg ** 2, axis))

        

    def fit(self, X, y=None):

        # create N centers spread along the data range

        self.centers_ = np.linspace(X.min(), X.max(), self.N)

        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])

        return self

        

    def transform(self, X):

        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,

                                 self.width_, axis=1)



rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = np.sin(x) + 0.1 * rng.randn(50)

xfit = np.linspace(0, 10, 1000)



gauss_model = make_pipeline(GaussianFeatures(10, 1.0),

                            LinearRegression())

gauss_model.fit(x[:, np.newaxis], y)

yfit = gauss_model.predict(xfit[:, np.newaxis])



gf = gauss_model.named_steps['gaussianfeatures']

lm = gauss_model.named_steps['linearregression']



fig, ax = plt.subplots()



for i in range(10):

    selector = np.zeros(10)

    selector[i] = 1

    Xfit = gf.transform(xfit[:, None]) * selector

    yfit = lm.predict(Xfit)

    ax.fill_between(xfit, yfit.min(), yfit, color='gray', alpha=0.2)



ax.scatter(x, y)

ax.plot(xfit, gauss_model.predict(xfit[:, np.newaxis]))

ax.set_xlim(0, 10)

ax.set_ylim(yfit.min(), 1.5)
from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import pairwise_distances_argmin



X, y_true = make_blobs(n_samples=300, centers=4,

                       cluster_std=0.60, random_state=0)



rng = np.random.RandomState(42)

centers = [0, 4] + rng.randn(4, 2)



def draw_points(ax, c, factor=1):

    ax.scatter(X[:, 0], X[:, 1], c=c, cmap='viridis',

               s=50 * factor, alpha=0.3)

    

def draw_centers(ax, centers, factor=1, alpha=1.0):

    ax.scatter(centers[:, 0], centers[:, 1],

               c=np.arange(4), cmap='viridis', s=200 * factor,

               alpha=alpha)

    ax.scatter(centers[:, 0], centers[:, 1],

               c='black', s=50 * factor, alpha=alpha)



def make_ax(fig, gs):

    ax = fig.add_subplot(gs)

    ax.xaxis.set_major_formatter(plt.NullFormatter())

    ax.yaxis.set_major_formatter(plt.NullFormatter())

    return ax



fig = plt.figure(figsize=(15, 4))

gs = plt.GridSpec(4, 15, left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)

ax0 = make_ax(fig, gs[:4, :4])

ax0.text(0.98, 0.98, "Random Initialization", transform=ax0.transAxes,

         ha='right', va='top', size=16)

draw_points(ax0, 'gray', factor=2)

draw_centers(ax0, centers, factor=2)



for i in range(3):

    ax1 = make_ax(fig, gs[:2, 4 + 2 * i:6 + 2 * i])

    ax2 = make_ax(fig, gs[2:, 5 + 2 * i:7 + 2 * i])

    

    # E-step

    y_pred = pairwise_distances_argmin(X, centers)

    draw_points(ax1, y_pred)

    draw_centers(ax1, centers)

    

    # M-step

    new_centers = np.array([X[y_pred == i].mean(0) for i in range(4)])

    draw_points(ax2, y_pred)

    draw_centers(ax2, centers, alpha=0.3)

    draw_centers(ax2, new_centers)

    for i in range(4):

        ax2.annotate('', new_centers[i], centers[i],

                     arrowprops=dict(arrowstyle='->', linewidth=1))

        

    

    # Finish iteration

    centers = new_centers

    ax1.text(0.95, 0.95, "E-Step", transform=ax1.transAxes, ha='right', va='top', size=14)

    ax2.text(0.95, 0.95, "M-Step", transform=ax2.transAxes, ha='right', va='top', size=14)





# Final E-step    

y_pred = pairwise_distances_argmin(X, centers)

axf = make_ax(fig, gs[:4, -4:])

draw_points(axf, y_pred, factor=2)

draw_centers(axf, centers, factor=2)

axf.text(0.98, 0.98, "Final Clustering", transform=axf.transAxes,

         ha='right', va='top', size=16)