import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kde

column_name = ['Age' , 'Operation_Year' ,'axil_nodes' , 'Surv_status' ]
surv_data = pd.read_csv('../input/haberman.csv' , header = None , names = column_name)
# surv_data.head() will print first 5 rows
surv_data.head()
# printing number of data points and features
surv_data.shape
surv_data['Surv_status'].value_counts() # imbalanced dataset
print('Total patient survived >= 5 years: {}%'.format((225/306)*100))

# patients who survived more than 5 years
surv_more = surv_data[surv_data['Surv_status']==1]
surv_more.describe()
# patients who survived less than 5 years
surv_less = surv_data[surv_data['Surv_status']==2]
surv_less.describe()

# AGE
sns.FacetGrid(surv_data , hue = 'Surv_status' , size = 5).map(sns.distplot , 'Age').add_legend();
plt.show();
# OPERATION YEAR
sns.FacetGrid(surv_data , hue = 'Surv_status' , size =4).map(sns.distplot , 'Operation_Year').add_legend();
plt.show()
# AXIL NODES
sns.FacetGrid(surv_data , hue = 'Surv_status' , size =4).map(sns.distplot , 'axil_nodes').add_legend();
plt.show()
#cdf on axil_node 

count1 , bin_edges1 = np.histogram(surv_more['axil_nodes'] , bins = 10 )
pdf1 = count1/sum(count1)
print(pdf1)
print(bin_edges1)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:] , pdf1)
plt.plot(bin_edges1[1:], cdf1 , label = 'surv >= 5 years')


count2 ,bin_edges2 = np.histogram(surv_less['axil_nodes'] , bins=10)
pdf2 = count2/sum(count2)
print(pdf2)
print(bin_edges2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges2[1:] , pdf2)
plt.plot(bin_edges2[1:] , cdf2 , label = 'surv < 5 years')
plt.xlabel('axil_nodes')
plt.legend()
plt.show()


# cdf on Age

count1 , bin_edges1 = np.histogram(surv_more['Age'] , bins = 10 )
pdf1 = count1/sum(count1)
print(pdf1)
print(bin_edges1)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:] , pdf1)
plt.plot(bin_edges1[1:], cdf1 , label = 'surv >= 5 years')


count2 ,bin_edges2 = np.histogram(surv_less['Age'] , bins=10)
pdf2 = count2/sum(count2)
print(pdf2)
print(bin_edges2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges2[1:] , pdf2)
plt.plot(bin_edges2[1:] , cdf2 , label = 'surv < 5 years')
plt.xlabel('Age')
plt.legend()
plt.show()
#cdf on Operation_Year

count1 , bin_edges1 = np.histogram(surv_more['Operation_Year'] , bins = 10 )
pdf1 = count1/sum(count1)
print(pdf1)
print(bin_edges1)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges1[1:] , pdf1)
plt.plot(bin_edges1[1:], cdf1 , label = 'surv >= 5 years')


count2 ,bin_edges2 = np.histogram(surv_less['Operation_Year'] , bins=10)
pdf2 = count2/sum(count2)
print(pdf2)
print(bin_edges2)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges2[1:] , pdf2)
plt.plot(bin_edges2[1:] , cdf2 , label = 'surv < 5 years')
plt.xlabel('Operation_Year')
plt.legend()
plt.show()
# box and violin plot ---> axil_nodes
plt.subplot(121)
sns.boxplot(x='Surv_status' , y='axil_nodes' , data = surv_data )
plt.subplot(122)
sns.violinplot(x='Surv_status' , y='axil_nodes' , data = surv_data )
plt.show()
# box and violin plot ---> Age
plt.subplot(121)
sns.boxplot(x='Surv_status' , y='Age' , data = surv_data)
plt.subplot(122)
sns.violinplot(x='Surv_status' , y='Age' , data = surv_data)
plt.show()
plt.subplot(121)
sns.boxplot(x='Surv_status' , y='Operation_Year' , data = surv_data)
plt.subplot(122)
sns.violinplot(x='Surv_status' , y='Operation_Year' , data = surv_data)
plt.show()
# Pair Plot

plt.close()
sns.set_style('whitegrid')
sns.pairplot(surv_data, hue ='Surv_status',vars = ['Age' , 'Operation_Year' ,'axil_nodes' ], size =3)
plt.show();
sns.jointplot(x='axil_nodes' , y = 'Age' , data = surv_more , kind = 'kde')
plt.show()
#2-D density plot
plt.close()
sns.jointplot(x='Operation_Year' , y='axil_nodes' , data = surv_more , kind='kde')
plt.show()