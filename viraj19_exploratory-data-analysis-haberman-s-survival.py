import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')

sns.set_style("whitegrid")

haberman = pd.read_csv("../input/haberman.csv", header=None, names=["age_of_patient","year_of_operation",\
                                                           "positive_auxilary_nodes","survival_status"])

print(haberman.head())
#First check the number of datapoints and features(Attributes)
print(haberman.shape)
print(haberman.columns)
print(haberman.info())
print(haberman['survival_status'].unique())
haberman['survival_status'] = haberman['survival_status'].map({1:"alive", 2:"dead"})
haberman['survival_status'] = haberman['survival_status'].astype('category')

print(haberman.info())
print(haberman.head())

print(haberman['survival_status'].value_counts())
print(haberman['survival_status'].value_counts(normalize=True))
print(haberman.describe())
sns.FacetGrid(haberman,hue="survival_status",height=6)\
   .map(plt.scatter,'age_of_patient','year_of_operation')\
   .add_legend()
plt.show()
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="survival_status", height=4,diag_kind="hist");
plt.suptitle('Pair Plots for haberman cancer survival dataset', 
             size = 24);
plt.show()


sns.FacetGrid(haberman,hue='survival_status',height=5)\
   .map(sns.distplot,'age_of_patient')\
   .add_legend()
plt.show()

sns.FacetGrid(haberman,hue='survival_status',height=5)\
   .map(sns.distplot,'positive_auxilary_nodes')\
   .add_legend()
plt.show()
sns.FacetGrid(haberman,hue='survival_status',height=5)\
   .map(sns.distplot,'year_of_operation')\
   .add_legend()
plt.show()
haberman_alive = haberman.loc[haberman['survival_status']=='alive']
haberman_dead = haberman.loc[haberman['survival_status']=="dead"]
plt.close()
counts,bin_edges = np.histogram(haberman_alive['age_of_patient'],bins=10,density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

#plt.figure(1)
#plt.subplot(211)

plt.plot(bin_edges[1:],pdf,label="pdf plot alive")
plt.plot(bin_edges[1:],cdf,label="cdf plot alive")

plt.ylabel("probablity density normalised to 1")
plt.xlabel("age of patient")
plt.title("PDF & CDF of age_of_patient for haberman(dead)")

#plt.subplot(212)
counts,bin_edges = np.histogram(haberman_dead['age_of_patient'],bins=10,density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf plot dead")
plt.plot(bin_edges[1:],cdf,label="cdf plot dead")

plt.ylabel("probablity density normalised to 1")
plt.xlabel("age of patient")
plt.title("PDF & CDF of age_of_patient for haberman(dead)")


plt.legend()
plt.show()


counts,bin_edges = np.histogram(haberman_alive['year_of_operation'],bins=10,density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf plot alive")
plt.plot(bin_edges[1:],cdf,label="cdf plot alive")

plt.ylabel("probablity density normalised to 1")
plt.xlabel("year_of_operation")
plt.title('CDF & PDF of year_of_operation for haberman(alive)')

counts,bin_edges = np.histogram(haberman_dead['year_of_operation'],bins = 10, density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf plot dead")
plt.plot(bin_edges[1:],cdf,label="cdf plot dead")
plt.title("CDF & PDF of year of operation for haberman(dead)")

plt.legend()
plt.show()

counts,bin_edges = np.histogram(haberman_alive['positive_auxilary_nodes'],bins=10,density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf plot alive")
plt.plot(bin_edges[1:],cdf,label="cdf plot alive")

plt.ylabel("probablity density normalised to 1")
plt.xlabel("positive_auxilary_nodes")
plt.title('CDF & PDF of positive_auxilary_nodes for haberman(alive)')


counts,bin_edges = np.histogram(haberman_dead["positive_auxilary_nodes"],bins=10,density=True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf plot dead")
plt.plot(bin_edges[1:],cdf,label="cdf plot dead")

plt.ylabel("probablity density normalised to 1")
plt.xlabel("positive_auxilary_nodes")
plt.title('CDF & PDF of positive_auxilary_nodes for haberman(dead)')

plt.legend()
plt.show()


from statsmodels import robust

print("***"*12,"mean","***"*12)
print(np.mean(haberman_alive['positive_auxilary_nodes']))
print(np.mean(haberman_dead['positive_auxilary_nodes']))

print("***"*12,"standard deviation","***"*12)
print(np.std(haberman_alive['positive_auxilary_nodes']))
print(np.std(haberman_dead['positive_auxilary_nodes']))

print("***"*12,"median","***"*12)
print(np.median(haberman_alive['positive_auxilary_nodes']))
print(np.median(haberman_dead['positive_auxilary_nodes']))

print("***"*12,"25,50,75 percentiles","***"*12)
print(np.percentile(haberman_alive['positive_auxilary_nodes'],np.arange(0,100,25)))
print(np.percentile(haberman_dead['positive_auxilary_nodes'],np.arange(0,100,25)))

print("***"*12,"median absolute deviation","***"*12)
print(robust.mad(haberman_alive["positive_auxilary_nodes"]))
print(robust.mad(haberman_dead["positive_auxilary_nodes"]))

sns.boxplot(x='survival_status',y='positive_auxilary_nodes',data=haberman)
plt.show()
sns.violinplot(x='survival_status',y='positive_auxilary_nodes',data=haberman)
plt.show()
sns.jointplot(x="age_of_patient", y="positive_auxilary_nodes", data=haberman_alive, kind="kde");
plt.show();

sns.jointplot(x="age_of_patient", y="positive_auxilary_nodes", data=haberman_dead, kind="kde");
plt.show();