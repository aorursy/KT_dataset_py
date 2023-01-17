import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

hab = pd.read_csv('../input/haberman.csv', header=None, names=['age','year','lym_nodes', 'surv_5plus'])

print(hab.columns)

hab
hab.columns
(hab["surv_5plus"]).value_counts()
hab.describe()
hab["surv_5plus"]=hab["surv_5plus"].map({2:"failure",1:"success"})
hab
hab["surv_5plus"]=success.value_counts()
import numpy as np
sns.FacetGrid(hab, hue="surv_5plus", height=6).map(sns.distplot, "age").add_legend();

plt.show()
sns.FacetGrid(hab, hue="surv_5plus", height=6).map(sns.distplot, "year").add_legend();

plt.show()
sns.FacetGrid(hab, hue="surv_5plus", height=8).map(sns.distplot, "lym_nodes").add_legend();

plt.show()
#hab=pd.read_csv("haberman.csv")

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

hab = pd.read_csv('haberman.csv', header=None, names=['age','year','lym_nodes', 'surv_5plus'])

hab["surv_5plus"]=hab["surv_5plus"].map({2:"failure",1:"success"})

s = hab.loc[hab["surv_5plus"] == "success"]

f = hab.loc[hab["surv_5plus"] == "failure"]

plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(s["age"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.xlabel("age")

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

print(pdf);

print(bin_edges)



plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(s["year"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.xlabel("year")

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

print(pdf);

print(bin_edges)
plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(s["lym_nodes"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.xlabel("lym_nodes")

plt.show()

print(pdf);

print(bin_edges)
plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(f["age"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.xlabel("age")

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

print(pdf);

print(bin_edges)

plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(f["year"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.xlabel("age")

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

print(pdf);

print(bin_edges)
plt.figure(figsize=(15,5))

counts, bin_edges = np.histogram(f["lym_nodes"], bins=10, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)

plt.plot(bin_edges[1:], cdf)

plt.xlabel("age")

plt.show()

print(pdf);

print(bin_edges)

import seaborn as sns

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

hab = pd.read_csv('haberman.csv', header=None, names=['age','year','lym_nodes', 'surv_5plus'])

hab["surv_5plus"]=hab["surv_5plus"].map({2:"failure",1:"success"})

s = hab.loc[hab["surv_5plus"] == "success"]

f = hab.loc[hab["surv_5plus"] == "failure"]

sns.boxplot(x="surv_5plus",y="age",hue="surv_5plus",data = hab)

plt.show()
s = hab.loc[hab["surv_5plus"] == "success"]

f = hab.loc[hab["surv_5plus"] == "failure"]

sns.boxplot(x="surv_5plus",y="year",hue="surv_5plus",data = hab)

plt.show()
s = hab.loc[hab["surv_5plus"] == "success"]

f = hab.loc[hab["surv_5plus"] == "failure"]

sns.boxplot(x="surv_5plus",y="lym_nodes",hue="surv_5plus",data = hab)

plt.show()
sns.violinplot(x="surv_5plus", y="age", data=hab, size=8)

plt.show()

sns.violinplot(x="surv_5plus", y="year", data=hab, size=8)

plt.show()

sns.violinplot(x="surv_5plus", y="lym_nodes", data=hab, size=8)

plt.show()
#scatter plots are nothing but 2D plots(involves 2 features)

sns.pairplot(hab, hue='surv_5plus', height=4)

plt.show()