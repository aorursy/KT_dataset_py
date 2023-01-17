# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



color = sns.color_palette()
data_path = "../input/"

train = pd.read_csv(data_path+"train_ver2.csv", usecols=['ncodpers'])

test = pd.read_csv(data_path+"test_ver2.csv", usecols=['ncodpers'])

print("Number of rows in train : ", train.shape[0])

print("Number of rows in test : ", test.shape[0])
train_unique_customers = set(train.ncodpers.unique())

test_unique_customers = set(test.ncodpers.unique())

print("Number of customers in train : ", len(train_unique_customers))

print("Number of customers in test : ", len(test_unique_customers))

print("Number of common customers : ", len(train_unique_customers.intersection(test_unique_customers)))
sub = pd.read_csv(data_path+"sample_submission.csv")

print("Number of rows in sample submission file : ",sub.shape[0])

sub_unique_customers = set(sub.ncodpers.unique())

print("Number of unique customers in sample submission file : ",len(sub_unique_customers))

print("Number of customers from test missing in sample sub :", len(test_unique_customers - sub_unique_customers))
num_occur = train.groupby('ncodpers').agg('size').value_counts()



plt.figure(figsize=(8,4))

sns.barplot(num_occur.index, num_occur.values, alpha=0.8, color=color[0])

plt.xlabel('Number of Occurrences of the customer', fontsize=12)

plt.ylabel('Number of customers', fontsize=12)

plt.show()
del train_unique_customers

del test_unique_customers

del sub

del sub_unique_customers
train = pd.read_csv(data_path+"train_ver2.csv", dtype='float16', 

                    usecols=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 

                             'ind_cco_fin_ult1', 'ind_cder_fin_ult1',

                             'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

                             'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',

                             'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',

                             'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

                             'ind_ecue_fin_ult1', 'ind_fond_fin_ult1',

                             'ind_hip_fin_ult1', 'ind_plan_fin_ult1',

                             'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

                             'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',

                             'ind_viv_fin_ult1', 'ind_nomina_ult1',

                             'ind_nom_pens_ult1', 'ind_recibo_ult1'])
target_counts = train.astype('float64').sum(axis=0)

#print(target_counts)

plt.figure(figsize=(8,4))

sns.barplot(target_counts.index, target_counts.values, alpha=0.8, color=color[0])

plt.xlabel('Product Name', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()