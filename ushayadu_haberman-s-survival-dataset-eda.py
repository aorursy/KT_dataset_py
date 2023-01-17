# The dataset contains cases from a study that was conducted between 1958 and 1970 at the University of Chicago's 
# Billings Hospital on the survival of patients who had undergone surgery for breast cancer.
# Attribute Information:
    # Age of patient at time of operation (numerical)
    # Patient's year of operation (year - 1900, numerical)
    # Number of positive axillary nodes detected (numerical)
    # Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
habermans = pd.read_csv("../input/haberman.csv",header=None,names=['age','year of operation','positive axillary nodes','Survival status'])
# As the dataset is large will be displaying top records from top and last using head and tail 
habermans.head()
warnings.filterwarnings("ignore")
# Displaying dataset from the bottom
habermans.tail()
#2
print(habermans.describe())
print("50th percentile of Survival Status " ,np.percentile(habermans['Survival status'],50))
print("75th percentile of Survival Status ",np.percentile(habermans['Survival status'],75))
print("70th percentile of Survival Status ",np.percentile(habermans['Survival status'],70))
print("73rd percentile of Survival Status ",np.percentile(habermans['Survival status'],73))
print("74th percentile of Survival Status ",np.percentile(habermans['Survival status'],74))
#3
print ("Number of Data Points & Features {0}". format(habermans.shape))
print("Name of coulmns or data features {0}". format(habermans.columns))
print("Number of classes, data-points per class : \n{0}" .format(habermans['Survival status'].value_counts()))

# 5.1 Lets plot Histogram with PDF of 'age', 'year of operations', 'positive axillary nodes'
sns.FacetGrid(habermans, hue="Survival status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.title("Histogram with PDF of age");
sns.FacetGrid(habermans, hue="Survival status", size=5) \
   .map(sns.distplot, "year of operation") \
   .add_legend();
plt.title("Histogram with PDF of year of operation");
sns.FacetGrid(habermans, hue="Survival status", size=5) \
   .map(sns.distplot, "positive axillary nodes") \
   .add_legend();
plt.title("Histogram with PDF of positive axillary nodes");
plt.show();
# 5.2 Lets plot CDF of 'age', 'year of operations', 'positive axillary nodes' using KDE
habermans_above5=habermans.loc[habermans["Survival status"]==1]
habermans_below5=habermans.loc[habermans['Survival status']==2]
# PDF & CDF on positive axillary nodes
counts, bin_edges = np.histogram(habermans_above5['positive axillary nodes'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['positive axillary nodes'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for Survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for Survival status \'2')
pylab.legend(loc='center')
plt.xlabel('positive axillary nodes')
plt.title('PDF & CDF on positive axillary nodes')
plt.show()
# PDF & CDF on age
counts, bin_edges = np.histogram(habermans_above5['age'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['age'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for survival status \'2')
pylab.legend(loc='upper left')
plt.xlabel('age')
plt.title('PDF & CDF on age')
plt.show()
# PDF & CDF on year of operations
counts, bin_edges = np.histogram(habermans_above5['year of operation'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='green',label='pdf for survival status \'1')
plt.plot(bin_edges[1:], cdf,color='blue',label='cdf for survival status \'1')
counts, bin_edges = np.histogram(habermans_below5['year of operation'], bins=10, density = True,)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf,color='red',label='pdf for survival status \'2')
plt.plot(bin_edges[1:], cdf,color='black',label='cdf for survival status \'2')
pylab.legend(loc='upper left')
plt.xlabel('year of operation')
plt.title('PDF & CDF on year of operation')
plt.show()
# 5.3 Lets plot BoxPlot of 'age', 'year of operations', 'positive axillary nodes' 
sns.boxplot(x='Survival status',y='positive axillary nodes', data=habermans)
plt.title('BoxPlot of positive axillary nodes')
plt.show()
sns.boxplot(x='Survival status',y='year of operation', data=habermans)
plt.title('BoxPlot of year of operation')
plt.show()
sns.boxplot(x='Survival status',y='age', data=habermans)
plt.title('BoxPlot of age')
plt.show()
# 5.4 Lets plot ViolinPlot of 'age', 'year of operations', 'positive axillary nodes' 
sns.violinplot(x="Survival status", y="positive axillary nodes", data=habermans, size=8)
plt.title('ViolinPlot of positive axillary nodes')
plt.show()
sns.violinplot(x='Survival status',y='year of operation', data=habermans)
plt.title('ViolinPlot of year of operation')
plt.show()
sns.violinplot(x='Survival status',y='age', data=habermans)
plt.title('ViolinPlot of age')
plt.show()
plt.close();
sns.set_style("whitegrid");
sns.pairplot(habermans, hue="Survival status", vars=['age', 'year of operation', 'positive axillary nodes'],size=4);
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(habermans, hue="Survival status", size=4) \
   .map(plt.scatter, "positive axillary nodes", "year of operation") \
   .add_legend();
plt.show();
