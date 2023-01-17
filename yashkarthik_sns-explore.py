# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from scipy import stats

from scipy.integrate import trapz





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sns.set(style='darkgrid')

tips = sns.load_dataset('tips')
tips.head()
sns.relplot(x='total_bill', y='tip',hue='smoker',style='time', data=tips);
sns.relplot(x='total_bill', y='tip', hue='size', data=tips);
sns.relplot(x='total_bill', y='tip', hue='smoker', style='time', size='size',sizes=(15, 200), data=tips);
df = pd.DataFrame(dict(time=np.arange(500),

                       value=np.random.randn(500).cumsum()))

sns.relplot(x='time', y='value', data=df, kind='line').fig.autofmt_xdate();
df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])

sns.relplot(x="x", y="y", sort=False, kind="line", data=df);
sns.relplot(x="x", y="y", sort=True, kind="line", data=df);
fmri = sns.load_dataset('fmri')

fmri.head()
sns.relplot(x='timepoint', y='signal', data=fmri, kind='line');
sns.relplot(x='timepoint', y='signal', data=fmri, kind='line', ci=None); # ci=None can also be ci=False
sns.relplot(x='timepoint', y='signal', data=fmri, kind='line', ci='sd'); # sd stand for - standard devieation
sns.relplot(x='timepoint', y='signal', data=fmri, kind='line', estimator=None); # CANNOT  use estimator=False; only None.
sns.relplot(x='timepoint', y='signal', data=fmri, hue='event', kind='line');
sns.relplot(x='timepoint', y='signal', data=fmri, hue='region', style='event', kind='line');
sns.relplot(x='timepoint', y='signal', data=fmri, kind='line', hue='region', style='event', dashes=False, markers=True);
sns.relplot(x='timepoint', y='signal', data=fmri.query("event=='cue'"), kind='line', estimator=None, units='subject', hue='region');
dots = sns.load_dataset('dots').query("align=='dots'")

dots
sns.relplot(x='time', y='firing_rate', data=dots, hue='coherence', style='choice', kind='line');
sns.relplot(x='time', y='firing_rate', data=dots, style='choice', kind='line', hue_norm=LogNorm(), size='coherence');
df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),

                       value=np.random.randn(500).cumsum()))

g = sns.relplot(x="time", y="value", kind="line", data=df)

g.fig.autofmt_xdate()
sns.relplot(x='total_bill', y='tip', data=tips, col='time', hue='smoker');
sns.relplot(x='timepoint', y='signal', data=fmri, hue='subject', col='region', row='event', height=5, kind='line');
sns.relplot(x='timepoint', y='signal', data=fmri.query("region=='frontal'"),

            hue='event', style='event', col='subject',

            col_wrap=4, kind='line', estimator=None,

            height=3, aspect=.75, linewidth=2.5,);
sns.set(style='ticks', color_codes=True)
sns.catplot(x='day', y='total_bill', data=tips);
sns.catplot(x='day', y='total_bill', data=tips, jitter=False);
sns.swarmplot(x='day', y='total_bill', data=tips);
sns.catplot(x='day', y='total_bill', data=tips, hue='sex');
sns.catplot(x='size', y='total_bill', data=tips, kind='swarm');
sns.catplot(x='smoker', y='tip', data=tips, order=['No', 'Yes']);
sns.catplot(x='total_bill', y='day', data=tips, kind='swarm', hue='time');
sns.catplot(x='day', y='total_bill', data=tips, kind='box', palette='magma');
sns.catplot(x='day', y='total_bill', data=tips, kind='box', hue='smoker');
tips['weekend'] = tips['day'].isin(['Sat', 'Sun'])

sns.catplot(x='day', y='total_bill', data=tips, kind='box', hue='weekend', dodge=False);
sns.catplot(x='day', y='total_bill', data=tips, kind='box', hue='weekend');
diamonds = sns.load_dataset('diamonds')

diamonds.head()
sns.catplot(x='color', y='price',data=diamonds.sort_values('color'), kind='boxen');
sns.catplot(x='total_bill', y='day', data=tips, kind='violin', hue='sex');
sns.catplot(x="total_bill", y="day", hue="sex",

            kind="violin", bw=.15, cut=0,

            data=tips);
sns.catplot(x='day', y='total_bill', data=tips, kind='violin', split=True, hue='sex');
sns.catplot(x="day", y="total_bill", hue="sex",

            kind="violin", inner="stick", split=True,

            palette="pastel", data=tips);
g = sns.catplot(x="day", y="total_bill", kind="violin", inner=None, data=tips, palette='pastel')

sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax); # we used swarmplot instead of catlpot as

# catplot is a figure level function.
titanic = sns.load_dataset('titanic')

titanic.head()
sns.catplot(x='sex', y='survived', hue='class', kind='bar', data=titanic, palette='pastel');
sns.countplot(x='deck', palette='pastel', data=titanic, hue='class');
sns.catplot(x='sex', y='survived', data=titanic, kind='point', hue='class');
sns.catplot(x='class', y='survived', data=titanic, kind='point', hue='sex', linestyles=['-', '-.'], markers=['^', 'o']);
iris = sns.load_dataset('iris')

iris.head()
sns.boxplot(data=iris, orient='h');
iris['petal_length'].describe()
sns.violinplot(x='species', y='sepal_length', data=iris);
tips
sns.catplot(x="day", y="total_bill", hue="smoker",

            col="time", aspect=.6,

            kind="swarm", data=tips);
g = sns.catplot(x="fare", y="survived", row="class",

                kind="box", orient="h", height=1.5, aspect=4,

                data=titanic.query("fare > 0"))

g.set(xscale="log");
sns.set(color_codes=True)
x = np.random.normal(size=100);
sns.distplot(x);
sns.distplot(x, rug=True, kde=False);
sns.distplot(x, rug=True, kde=False, bins=20); # bins=20 => 20 categories
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
density = np.sum(kernels, axis=0)

density /= trapz(density, support)

plt.plot(support, density);
sns.kdeplot(x, shade=True);
sns.kdeplot(x, shade=True);

sns.kdeplot(x, bw=.2, label='bw:0.2');

sns.kdeplot(x, bw=2, label='bw:2');
sns.kdeplot(x, cut=0, shade=True);  # cut=0 implies the curve stops with the curve.

sns.kdeplot(x, cut=1);              # positive value of cut implies the curve extends past the actual observations.

sns.kdeplot(x, cut=2);      

sns.kdeplot(x, cut=-1);             # negative value of cut implies the curve stops before the actual farthest observations.

sns.rugplot(x);
x = np.random.gamma(6, size=200)

sns.distplot(x, fit=stats.gamma);
mean, cov = [0, 1], [(1, .5), (.5, 1)]

data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])

df.head()
sns.jointplot(x='x', y='y', data=df);
x, y = np.random.multivariate_normal(mean, cov, 1000).T

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k");
sns.jointplot(x='x', y='y', data=df, kind='kde');
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.x, df.y, ax=ax)

sns.rugplot(df.x, color="g", ax=ax)

sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

sns.kdeplot(df.x, df.y, ax=ax, shade=True)

sns.rugplot(df.x, ax=ax)

sns.rugplot(df.y, vertical=True, ax=ax);
f, ax = plt.subplots(figsize=(6, 6))

cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True);
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$X$", "$Y$");
sns.pairplot(iris);
sns.pairplot(iris, hue='species');
g = sns.PairGrid(iris)

g.map_diag(sns.kdeplot)

g.map_offdiag(sns.kdeplot, n_levels=6, shade=True);
sns.set(color_codes=True)
sns.regplot(x='total_bill', y='tip', data=tips);
sns.lmplot(x='total_bill', y='tip', data=tips);
sns.lmplot(x='size', y='tip', data=tips);
sns.lmplot(x='size', y='tip', data=tips, x_jitter=0.05);
sns.lmplot(x='size', y='tip', data=tips, x_estimator=np.mean);
anscombe = sns.load_dataset('anscombe')

anscombe
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='I'"));
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='I'"),

           ci=None);
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='II'"),

          ci=None, scatter_kws={'s':80});
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='II'"),

           order=2, ci=None);
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='III'"),

           ci=None);
sns.lmplot(x='x', y='y', data=anscombe.query("dataset=='III'"),

          robust=True, ci=None);
tips["big_tip"] = (tips.tip / tips.total_bill) > .15

sns.lmplot(x='total_bill', y='big_tip', data=tips,

          y_jitter=0.03);
sns.lmplot(x='total_bill', y='big_tip', data=tips,

          logistic=True, y_jitter=0.03);
sns.lmplot(x='total_bill', y='big_tip', data=tips, lowess=True, y_jitter=0.03);
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),

              scatter_kws={"s": 80});
sns.residplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),

              scatter_kws={"s": 80});
sns.lmplot(x='total_bill', y='tip', data=tips, hue='smoker');
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,

           markers=["o", "x"], palette="magma");
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker",

           col="time", row="sex", data=tips);
f, ax = plt.subplots(figsize=(5, 6))

sns.regplot(x="total_bill", y="tip", data=tips, ax=ax);
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,

           col_wrap=2, height=3);
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,

           aspect=.5);
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],

             height=5, aspect=.8, kind="reg");
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],

             hue="smoker", height=5, aspect=.8, kind="reg");
sns.set(style='ticks')
g = sns.FacetGrid(tips, col='time');
g = sns.FacetGrid(tips, col='time')

g.map(plt.hist, 'tip');
sns.FacetGrid(tips, col='time').map(plt.hist, 'tip');
sns.FacetGrid(tips, col='sex',

              hue='smoker').map(plt.scatter, 'total_bill',

              'tip', alpha=0.7).add_legend();
sns.FacetGrid(tips, row='smoker',

              col='time', margin_titles=True).map(sns.regplot, 'size', 'total_bill', 

                                                 fit_reg=False, x_jitter=.1, color='.3');
sns.FacetGrid(tips, col='day', height=4, aspect=.5).map(sns.barplot, 'sex', 'total_bill', palette='magma');
ordered_days = tips['day'].value_counts().index

ordered_days
sns.FacetGrid(tips, row='day', row_order=ordered_days,

             height=1.7, aspect=4).map(sns.distplot, 'total_bill', hist=False, rug=True);
sns.FacetGrid(tips, hue='time', palette='magma', height=7).map(plt.scatter, 'total_bill', 'tip').add_legend();
sns.FacetGrid(tips, hue='sex', palette='Set1', height=5.5, 

              hue_kws={'marker':['^', 'v']}).map(plt.scatter, 'total_bill', 'tip').add_legend();
attend = sns.load_dataset('attention', index_col=0).query("subject <= 12")

attend
sns.FacetGrid(attend, col='subject', col_wrap=4, 

              height=2, ylim=(0, 10)).map(sns.pointplot, 'solutions', 'score', 

                                         order=[1,2,3], color='.3', ci=None);