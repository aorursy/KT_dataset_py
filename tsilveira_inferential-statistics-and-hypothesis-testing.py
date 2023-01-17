# Data analysis packages:
import pandas as pd
import numpy as np
#from datetime import datetime as dt

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## Loading the dataset
data = [[12.079,19.278],[16.791,18.741],[9.564,21.214],[8.630,15.687],[14.669,22.803],[12.238,20.878],[14.692,24.572],[8.987,17.394],[9.401,20.762],[14.480,26.282],[22.328,24.524],[15.298,18.644],[15.073,17.510],[16.929,20.330],[18.200,35.255],[12.130,22.158],[18.495,25.139],[10.639,20.429],[11.344,17.425],[12.369,34.288],[12.944,23.894],[14.233,17.960],[19.710,22.058],[16.004,21.157]]
dataset = pd.DataFrame(data, columns=['Congruent','Incongruent'])
## Printing the number of samples in the dataset:
print('The dataset has {0} samples.'.format(len(dataset)))
## Printing out some few lines:
dataset.head(3)
dataset['Difference'] = dataset['Congruent'] - dataset['Incongruent']
dataset.head(3)
## Obtaining the dataset descriptive statistics:
dataset.describe()
fig1, ax = plt.subplots(figsize=[12,6])  #Defines the graph window size
fig1.subplots_adjust(top=0.92)
plt.suptitle('Sample distributions for Stroop Effect experiment', fontsize=14, fontweight='bold')

sns.distplot(dataset['Congruent'], label='Congruent', ax=ax)
sns.distplot(dataset['Incongruent'], label='Incongruent', ax=ax)
sns.distplot(dataset['Difference'], label='Difference', ax=ax)
ax.legend(ncol=2, loc="upper right", frameon=True)

ax.set_xlabel('Time to read and say out the printed color of the word set', fontsize=12)
plt.show()
fig2, ax2 = plt.subplots(figsize=[12,6])  #Defines the graph window size
fig2.subplots_adjust(top=0.92)
plt.suptitle('Boxplot of the sample distributions for Stroop Effect experiment', fontsize=14, fontweight='bold')
sns.boxplot(data=dataset, orient='v', ax=ax2)
plt.show()
n = len(dataset)
df = n-1
print('Degrees of freedom: {}'.format(df))
## Difference mean
Xd = dataset['Difference'].mean()
print('Sample mean = {}'.format(Xd))
## Difference standard deviation
Sd = dataset['Difference'].std()
print('Sample std = {}'.format(Sd))
SEM = Sd / np.sqrt(n)
print('SEM = {}'.format(SEM))
t = Xd / SEM
print('t-test = {}'.format(t))
from scipy import stats
p = stats.t.cdf(t, df=df)
print('p-value = {0:.9f}'.format(p))
stats.ttest_rel(dataset['Congruent'],dataset['Incongruent'])
t_critical = 1.714
ME = t_critical*SEM
CI = (Xd-ME, Xd+ME)
print('Confidence interval: ({0:.2f}, {1:.2f})'.format(CI[0], CI[1]))