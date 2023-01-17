import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
'''Reading the Haberman's Data'''
#Load haberman.csv into a pandas dataFrame.
#To read the data from the specified convert the data file path from normal string to raw string
import os
print(os.listdir("../input"))
Haber = pd.read_csv("../input/haberman.csv",header = None)
print (Haber.shape)
print (Haber.columns)
#Adding headder to the Haber dataframe
Haber.columns=["Age","Operation_year","axil_nodes","Surv_Status"]
print (Haber.columns)

Haber.head(5)
Haber.describe()
Haber.plot(kind='Scatter',x='Age',y='Operation_year') 
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(Haber,hue='Age',size=4)\
  .map(plt.scatter,"Age","axil_nodes")
plt.show()
Haber["Surv_Status"] = Haber["Surv_Status"].apply(lambda y: "Survived" if y == 1 else "Died")
Survive_long=Haber.loc[Haber["Surv_Status"] == "Survived"]
Survive_short=Haber.loc[Haber["Surv_Status"] == "Died"]
plt.plot(Survive_long["Age"],np.zeros_like(Survive_long['axil_nodes']),'o')
plt.plot(Survive_short["Age"],np.zeros_like(Survive_short['axil_nodes']),'x')
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(Haber,hue='Surv_Status',size=3)

plt.show()
import numpy as np
Survive_long=Haber.loc[Haber["Surv_Status"] == "Survived"]
Survive_short=Haber.loc[Haber["Surv_Status"] == "Died"]
plt.plot(Survive_long["Age"],np.zeros_like(Survive_long['axil_nodes']),'o')
plt.plot(Survive_short["Age"],np.zeros_like(Survive_short['axil_nodes']),'x')
plt.show()
Survive_long.describe()
Survive_short.describe()
sns.FacetGrid(Haber,hue="Surv_Status",size=5)\
    .map(sns.distplot,"Operation_year")\
    .add_legend()
plt.show()
sns.FacetGrid(Haber,hue="Surv_Status",size=5)\
    .map(sns.distplot,"Age")\
    .add_legend()
plt.show()
sns.FacetGrid(Haber,hue="Surv_Status",size=5)\
    .map(sns.distplot,"axil_nodes")\
    .add_legend()
plt.show()
#Plot CDF of Survive_Status

# Survive_short
counts, bin_edges = np.histogram(Survive_short['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#Survive_long
counts, bin_edges = np.histogram(Survive_long['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();
#Plot CDF of Survive_Status

# Survive_short
counts, bin_edges = np.histogram(Survive_short['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#Survive_long
counts, bin_edges = np.histogram(Survive_long['axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();
# virginica
counts, bin_edges = np.histogram(Survive_short['Operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
 

#versicolor
counts, bin_edges = np.histogram(Survive_long['Operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();
sns.boxplot(x='Surv_Status',y='Age', data=Haber)
plt.show()
sns.boxplot(x='Surv_Status',y='Operation_year', data=Haber)
plt.show()
sns.boxplot(x='Surv_Status',y='axil_nodes', data=Haber)
plt.show()
sns.violinplot(x="Surv_Status", y="Age", data=Haber, size=8)
plt.show()
sns.violinplot(x="Surv_Status", y="Operation_year", data=Haber, size=8)
plt.show()
sns.violinplot(x="Surv_Status", y="axil_nodes", data=Haber, size=8)
plt.show()