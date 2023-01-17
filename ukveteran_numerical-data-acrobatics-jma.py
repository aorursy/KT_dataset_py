import numpy as np
import pandas as pd
from collections import Counter

# pandas display data frames as tables
from IPython.display import display, HTML

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sns.set_style('whitegrid')
sns.set_context('talk')
# load dataset 
df = pd.read_excel('../input/credit-default/credit_default.xls',
                             skiprows=1,index_col=0)
df.shape
df.head
df[['LIMIT_BAL','BILL_AMT1',
                   'BILL_AMT2','BILL_AMT3',
                   'BILL_AMT4','BILL_AMT5',
                   'BILL_AMT6']].head()
def default_month_count(row):
    count = 0 
    for i in [0,2,3,4,5,6]:
        if row['PAY_'+str(i)] > 0:
            count +=1
    return count
df['number_of_default_months'] = df.apply(default_month_count,axis=1)
df[['number_of_default_months']].head()
df['has_ever_defaulted'] = df.number_of_default_months.apply(lambda x: 1 if x>0 else 0)
df[['number_of_default_months','has_ever_defaulted']].head()
df.AGE.plot(kind='hist',bins=60)
plt.title('Age Histogram', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.xlabel('Frequency', fontsize=12)
df['age_bin_fixed'] = df.AGE.apply(lambda age: np.floor(age/10.))
df[['AGE','age_bin_fixed']].head()
quantile_list = [0, .25, .5, .75, 1.]
quantiles = df.AGE.quantile(quantile_list)
quantiles
fig, ax = plt.subplots()
df.AGE.plot(kind='hist',bins=60)

for quantile in quantiles:
    qvl = plt.axvline(quantile, color='r')
ax.legend([qvl], ['Quantiles'], fontsize=10)

ax.set_title('Age Histogram with Quantiles', fontsize=12)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
df['age_quantile_range'] = pd.qcut(df['AGE'],
                                          q=quantile_list)
df['age_quantile_label'] = pd.qcut(df['AGE'],
                                          q=quantile_list,
                                          labels=quantile_labels)
df[['AGE','age_quantile_range','age_quantile_label']].head()