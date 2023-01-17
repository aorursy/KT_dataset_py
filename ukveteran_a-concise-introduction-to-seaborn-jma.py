import matplotlib.pyplot as plt
plt.style.use('classic')
%matplotlib inline
import numpy as np
import pandas as pd
# Create some data
rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)
# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');
import seaborn as sns
sns.set()
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
for col in 'xy':
    sns.kdeplot(data[col], shade=True)
sns.distplot(data['x'])
sns.distplot(data['y']);
sns.kdeplot(data)
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')
iris = sns.load_dataset("iris")
iris.head()
sns.pairplot(iris, hue='species', size=2.5)
tips = sns.load_dataset('tips')
tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))
with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")
with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')
sns.jointplot("total_bill", "tip", data=tips, kind='reg')
planets = sns.load_dataset('planets')
planets.head()
with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)
g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
g.set_ylabels('Number of Planets Discovered')