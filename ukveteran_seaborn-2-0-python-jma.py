import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.set(color_codes=True)
x = np.random.normal(size=100)

sns.distplot(x);
sns.distplot(x, kde=False, rug=True);
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, hist=False, rug=True);
x = np.random.normal(0, 1, size=30)

bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)

support = np.linspace(-4, 4, 200)



kernels = []

for x_i in x:



    kernel = stats.norm(x_i, bandwidth).pdf(support)

    kernels.append(kernel)

    plt.plot(support, kernel, color="r")



sns.rugplot(x, color=".2", linewidth=3);
from scipy.integrate import trapz

density = np.sum(kernels, axis=0)

density /= trapz(density, support)

plt.plot(support, density);
sns.kdeplot(x, shade=True);
sns.kdeplot(x)

sns.kdeplot(x, bw=.2, label="bw: 0.2")

sns.kdeplot(x, bw=2, label="bw: 2")

plt.legend();
sns.kdeplot(x, shade=True, cut=0)

sns.rugplot(x);
x = np.random.gamma(6, size=200)

sns.distplot(x, kde=False, fit=stats.gamma);
mean, cov = [0, 1], [(1, .5), (.5, 1)]

data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df);
x, y = np.random.multivariate_normal(mean, cov, 1000).T

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k");
sns.jointplot(x="x", y="y", data=df, kind="kde");
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.x, df.y, ax=ax)

sns.rugplot(df.x, color="g", ax=ax)

sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$");