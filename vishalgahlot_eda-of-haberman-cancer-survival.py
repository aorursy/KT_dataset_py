import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


haber = pd.read_csv("../input/haberman.csv")
print(haber.shape)
print("no. of Instances:{} ".format(haber.shape[0]))
for i in haber.columns:print(i)
haber["Survival_status"].value_counts()
haber.plot(kind='scatter', x='Patient_age', y="Patient's_year_of_operation ") ;
plt.grid()
plt.show()

s=haber["Patient's_year_of_operation "].value_counts()
s=dict(s)

plt.bar(range(len(s)), list(s.values()), align='center')
plt.xticks(range(len(s)), list(s.keys()))

plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(haber, hue="Survival_status", size=4) \
   .map(plt.scatter, "Patient_age", "Patient's_year_of_operation ") \
   .add_legend();

plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haber, hue="Survival_status", size=3);
plt.show()
import numpy as np
haber_1 = haber.loc[haber["Survival_status"] == 1];
haber_2 = haber.loc[haber["Survival_status"] == 2];

haber["Survival_status"].value_counts()
print(haber.columns)
sns.FacetGrid(haber, hue="Survival_status", size=5) \
   .map(sns.distplot, "No_ofpositive_axillary_node_detected") \
   .add_legend();
plt.show();

sns.FacetGrid(haber, hue="Survival_status", size=5) \
   .map(sns.distplot, "Patient's_year_of_operation ") \
   .add_legend();
plt.show();

sns.FacetGrid(haber, hue="Survival_status", size=5) \
   .map(sns.distplot, "Patient_age") \
   .add_legend();
plt.show();
print("Analysis using No_ofpositive_axillary_node_detected\n")

counts, bin_edges = np.histogram(haber_1['No_ofpositive_axillary_node_detected'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show()
print("Analysis using Patient_age")
counts, bin_edges = np.histogram(haber_1['Patient_age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf)
plt.show()
print("Analysis using No_ofpositive_axillary_node_detected\n")

counts, bin_edges = np.histogram(haber_2['No_ofpositive_axillary_node_detected'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.show()
print("Analysis using Patient_age")
counts, bin_edges = np.histogram(haber_2['Patient_age'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf)
plt.show()
print("\nMean:")
print(np.mean(haber_1["No_ofpositive_axillary_node_detected"]))
#Median with an outlier
print(np.mean(haber_2["No_ofpositive_axillary_node_detected"]))

print(haber_1["No_ofpositive_axillary_node_detected"].value_counts())

print("\nMedian:")
print(np.median(haber_1["No_ofpositive_axillary_node_detected"]))
print(np.median(haber_2["No_ofpositive_axillary_node_detected"]))

print("\nStd-dev:");
print(np.std(haber_1["No_ofpositive_axillary_node_detected"]))
print(np.std(haber_2["No_ofpositive_axillary_node_detected"]))


print("\n90th Percentiles:")
print(np.percentile(haber_1["No_ofpositive_axillary_node_detected"],90))
print(np.percentile(haber_2["No_ofpositive_axillary_node_detected"],90))

sns.boxplot(x=haber["Survival_status"],y=haber["No_ofpositive_axillary_node_detected"],linewidth=2.5)
plt.show()
sns.boxplot(x=haber["Survival_status"],y=haber["Patient_age"],linewidth=2.5)
plt.show()
sns.jointplot(x="Patient_age", y="No_ofpositive_axillary_node_detected", data=haber_1, kind="kde");
plt.show();
sns.jointplot(x="Patient_age", y="No_ofpositive_axillary_node_detected", data=haber_2, kind="kde");
plt.show();
