import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv('../input/HR_comma_sep.csv');

dict_salary = {'low':1,'medium':2,'high':3};

data['salary'] = [dict_salary[x] for x in data['salary']];

#dict_sales = {'sales':1,'accounting':2, 'hr':3, 'technical':4, 'support':5, 'management':6, 'IT':7,'product_mng':8, 'marketing':9, 'RandD':10};

#data['sales'] = [dict_sales[x] for x in data['sales']];

corr = data.corr()

print(corr);

sns.set()

plt.figure(figsize=(10,10));

sns.heatmap(corr,annot=True,vmax=1,vmin=-1,cmap='cubehelix');
data_left = data[data['left']==1];

data_unleft = data[data['left']==0];



print (data_left.groupby('salary').size())#look each group's size
print ('%f the salary is not high in left group'%(1-82/float(len(data_left))))
data_left_high = data_left[data_left['salary']==3]

print(data_left_high.describe())

#we can see the pepole have none promotion_last_5years with high salary from data_left_high.describe()

# what's more 75% pepole with high salary have low satisfaction_level 
data_left_lowAndmedium = data_left[data_left['salary']!=3];

print(data_left_lowAndmedium.describe());

#we can see the 75% pepole have none promotion_last_5years with low or medium salary from data_left_lowAndmedium.describe()

#50% pepole with high salary have low satisfaction_level 
data.groupby('sales').size()
sales_mean = data.groupby('sales').mean()

data.groupby('sales').mean()
workType = data.sales.unique();

value = [];

value.append(data.satisfaction_level[data['sales']=='IT'].mean());

value.append(data.satisfaction_level[data['sales']=='RandD'].mean());

value.append(sales_mean['satisfaction_level'].accounting);

value.append(sales_mean['satisfaction_level'].hr);

value.append(sales_mean['satisfaction_level'].management);

value.append(sales_mean['satisfaction_level'].marketing);

value.append(sales_mean['satisfaction_level'].product_mng);

value.append(data.satisfaction_level[data['sales']=='sales'].mean());

value.append(data.satisfaction_level[data['sales']=='support'].mean());

value.append(sales_mean['satisfaction_level'].technical);
plt.barh(np.arange(len(workType)),value);

plt.yticks(np.arange(len(workType))+0.4,workType)

plt.show()