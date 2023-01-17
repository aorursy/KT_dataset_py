#Import libraries



import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualization

import seaborn as sns # visualization

from scipy import stats

from scipy.stats import norm 

import warnings 

warnings.filterwarnings('ignore') #ignore warnings



%matplotlib inline

import gc
train=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv", parse_dates=["Date"])

test=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv", parse_dates=["Date"])

stores=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
print("the structure of train data is ", train.shape)

print("the structure of test  data is ", test.shape)

print("the ratio of train data : test data is ", (round(train.shape[0]*100/(train.shape[0]+test.shape[0])),100-round(train.shape[0]*100/(train.shape[0]+test.shape[0]))))
train=train.merge(stores, on='Store', how='left')

train.head()
train['Year']=train['Date'].dt.year

train['Month']=train['Date'].dt.month

train['Week']=train['Date'].dt.week

train['Day']=train['Date'].dt.day

train['n_days']=(train['Date'].dt.date-train['Date'].dt.date.min()).apply(lambda x:x.days)

Year=pd.Series(train['Year'].unique())

Week=pd.Series(train['Week'].unique())

Month=pd.Series(train['Month'].unique())

Day=pd.Series(train['Day'].unique())

n_days=pd.Series(train['n_days'].unique())
print("the shape of stores data set is", stores.shape)

print("the unique value of store is", stores['Store'].unique())

print("the unique value of Type is", stores['Type'].unique())
print(stores.head())

grouped=stores.groupby('Type')

print(grouped.describe()['Size'].round(2))
plt.style.use('ggplot')

labels=['A store','B store','C store']

sizes=grouped.describe()['Size'].round(1)

sizes=[(22/(17+6+22))*100,(17/(17+6+22))*100,(6/(17+6+22))*100] # convert to the proportion





fig, axes = plt.subplots(1,1, figsize=(10,10))



wprops={'edgecolor':'black',

      'linewidth':2}



tprops = {'fontsize':30}





axes.pie(sizes,

        labels=labels,

        explode=(0.02,0,0),

        autopct='%1.1f%%',

        pctdistance=0.6,

        labeldistance=1.2,

        wedgeprops=wprops,

        textprops=tprops,

        radius=0.8,

        center=(0.5,0.5))

plt.show()

data = pd.concat([stores['Type'], stores['Size']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Type', y='Size', data=data)
data = pd.concat([train['Type'], train['Weekly_Sales']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Type', y='Weekly_Sales', data=data, showfliers=False)
plt.style.use('ggplot')



fig=plt.figure()

ax=fig.add_subplot(111)



ax.scatter(train['Size'],train['Weekly_Sales'], alpha=0.5)



plt.show()



types=stores['Type'].unique()



plt.style.use('ggplot')



fig=plt.figure(figsize=(10,5))

ax=fig.add_subplot(111)



for t in types:

    x=train.loc[train['Type']==t, 'Size']

    y=train.loc[train['Type']==t, 'Weekly_Sales']

    

    ax.scatter(x,y,alpha=0.5, label=t)



ax.set_title('Scatter plot size and sales by store type')

ax.set_xlabel('Size')

ax.set_ylabel('Weekly_Sales')



ax.legend(loc='higher right',fontsize=12)



plt.show()
train.head()
data = pd.concat([train['Store'], train['Weekly_Sales'], train['Type']], axis=1)

f, ax = plt.subplots(figsize=(25, 8))

fig = sns.boxplot(x='Store', y='Weekly_Sales', data=data, showfliers=False, hue="Type")
data = pd.concat([train['Store'], train['Weekly_Sales'], train['IsHoliday']], axis=1)

f, ax = plt.subplots(figsize=(25, 8))

fig = sns.boxplot(x='Store', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")
data = pd.concat([train['Dept'], train['Weekly_Sales'], train['Type']], axis=1)

f, ax = plt.subplots(figsize=(25, 10))

fig = sns.boxplot(x='Dept', y='Weekly_Sales', data=data, showfliers=False)
data = pd.concat([train['Dept'], train['Weekly_Sales'], train['Type']], axis=1)

f, ax = plt.subplots(figsize=(10, 50))

fig = sns.boxplot(y='Dept', x='Weekly_Sales', data=data, showfliers=False, hue="Type",orient="h") 
data = pd.concat([train['Dept'], train['Weekly_Sales'], train['IsHoliday']], axis=1)

f, ax = plt.subplots(figsize=(25, 10))

fig = sns.boxplot(x='Dept', y='Weekly_Sales', data=data, showfliers=False, hue="IsHoliday")
train.head()
plt.style.use('ggplot')

fig, axes = plt.subplots(1,2, figsize = (20,5))

fig.subplots_adjust(wspace=1, hspace=1)

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)



sales_holiday=train[['IsHoliday','Weekly_Sales']]

target=[sales_holiday['Weekly_Sales'].loc[sales_holiday['IsHoliday']==True],sales_holiday['Weekly_Sales'].loc[sales_holiday['IsHoliday']==False]]

labels=['Holiday','Not Holiday']



#median

medianprop={'color':'#2196F3',

            'linewidth': 2,

            'linestyle':'-'}



# outliers



flierprop={'color' : '#EC407A',

          'marker' : 'o',

          'markerfacecolor': '#2196F3',

          'markeredgecolor':'white',

          'markersize' : 3,

          'linestyle' : 'None',

          'linewidth' : 0.1}







axes[0].boxplot(target,labels=labels, patch_artist = 'Patch',

                  showmeans=True,

                  flierprops=flierprop,

                  medianprops=medianprop)









axes[1].boxplot(target,labels=labels, patch_artist = 'Patch',

                  showmeans=True,

                  flierprops=flierprop,

                  medianprops=medianprop)



axes[1].set_ylim(-6000,80000)



plt.show()





print(train[train['IsHoliday']==True]['Weekly_Sales'].describe().round(1))

print(train[train['IsHoliday']==False]['Weekly_Sales'].describe().round(1))
train.head()
data = pd.concat([train['Month'], train['Weekly_Sales']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Month', y="Weekly_Sales", data=data, showfliers=False)
data = pd.concat([train['Month'], train['Weekly_Sales'],train['IsHoliday']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Month', y="Weekly_Sales", data=data, showfliers=False, hue='IsHoliday')
data = pd.concat([train['Month'], train['Weekly_Sales'],train['Type']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Month', y="Weekly_Sales", data=data, showfliers=False, hue='Type')
data = pd.concat([train['Year'], train['Weekly_Sales']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='Year', y="Weekly_Sales", data=data, showfliers=False)
data = pd.concat([train['Week'], train['Weekly_Sales']], axis=1)

f, ax = plt.subplots(figsize=(20, 6))

fig = sns.boxplot(x='Week', y="Weekly_Sales", data=data, showfliers=False)
f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(train['Weekly_Sales'])
print("Skewness: ", train['Weekly_Sales'].skew()) #skewness

print("Kurtosis: ", train['Weekly_Sales'].kurt()) #kurtosis
train['Weekly_Sales'].min()
fig = plt.figure(figsize = (10,5))



fig.add_subplot(1,2,1)

res = stats.probplot(train.loc[train['Weekly_Sales']>0,'Weekly_Sales'], plot=plt)



fig.add_subplot(1,2,2)

res = stats.probplot(np.log1p(train.loc[train['Weekly_Sales']>0,'Weekly_Sales']), plot=plt)
train.describe()['Weekly_Sales']
train_over_zero=train[train['Weekly_Sales']>0]

train_below_zero=train[train['Weekly_Sales']<=0]

sales_over_zero = np.log1p(train_over_zero['Weekly_Sales'])

#histogram

f, ax = plt.subplots(figsize=(8, 6))

sns.distplot(sales_over_zero)
print("Skewness: ", sales_over_zero.skew()) #skewness

print("Kurtosis: ", sales_over_zero.kurt()) #kurtosis
grouped=train.groupby(['Dept','Date']).mean().round(0).reset_index()

print(grouped.shape)

print(grouped.head())

data=grouped[['Dept','Date','Weekly_Sales']]





dept=train['Dept'].unique()

dept.sort()

dept_1=dept[0:20]

dept_2=dept[20:40]

dept_3=dept[40:60]

dept_4=dept[60:]



fig, ax = plt.subplots(2,2,figsize=(20,10))

fig.subplots_adjust(wspace=0.5, hspace=0.5)

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)



for i in dept_1 :

    data_1=data[data['Dept']==i]

    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')



for i in dept_2 :

    data_1=data[data['Dept']==i]

    ax[0,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')

    

for i in dept_3 :

    data_1=data[data['Dept']==i]

    ax[1,0].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')    



for i in dept_4 :

    data_1=data[data['Dept']==i]

    ax[1,1].plot(data_1['Date'], data_1['Weekly_Sales'],label='Dept_1_mean_sales')        

    

ax[0,0].set_title('Mean sales record by department(0~19)')

ax[0,1].set_title('Mean sales record by department(20~39)')

ax[1,0].set_title('Mean sales record by department(40~59)')

ax[1,1].set_title('Mean sales record by department(60~)')





ax[0,0].set_ylabel('Mean sales')

ax[0,0].set_xlabel('Date')

ax[0,1].set_ylabel('Mean sales')

ax[0,1].set_xlabel('Date')

ax[1,0].set_ylabel('Mean sales')

ax[1,0].set_xlabel('Date')

ax[1,1].set_ylabel('Mean sales')

ax[1,1].set_xlabel('Date')





plt.show()
grouped=train.groupby(['Store','Date']).mean().round(0).reset_index()

grouped.shape

grouped.head()



data=grouped[['Store','Date','Weekly_Sales']]

type(data)





store=train['Store'].unique()

store.sort()

store_1=store[0:5]

store_2=store[5:10]

store_3=store[10:15]

store_4=store[15:20]

store_5=store[20:25]

store_6=store[25:30]

store_7=store[30:35]

store_8=store[35:40]

store_9=store[40:]



fig, ax = plt.subplots(5,2,figsize=(20,15))



fig.subplots_adjust(wspace=0.5, hspace=0.5)

fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)



for i in store_1 :

    data_1=data[data['Store']==i]

    ax[0,0].plot(data_1['Date'], data_1['Weekly_Sales'])

    

for i in store_2 :

    data_2=data[data['Store']==i]

    ax[0,1].plot(data_2['Date'], data_2['Weekly_Sales'])

    

for i in store_3 :

    data_3=data[data['Store']==i]

    ax[1,0].plot(data_3['Date'], data_3['Weekly_Sales'])



for i in store_4 :

    data_4=data[data['Store']==i]

    ax[1,1].plot(data_4['Date'], data_4['Weekly_Sales'])

    

for i in store_5 :

    data_5=data[data['Store']==i]

    ax[2,0].plot(data_5['Date'], data_5['Weekly_Sales'])  



for i in store_6 :

    data_6=data[data['Store']==i]

    ax[2,1].plot(data_6['Date'], data_6['Weekly_Sales'])  



for i in store_7 :

    data_7=data[data['Store']==i]

    ax[3,0].plot(data_7['Date'], data_7['Weekly_Sales'])      



for i in store_8 :

    data_8=data[data['Store']==i]

    ax[3,1].plot(data_8['Date'], data_8['Weekly_Sales'])     

    

for i in store_9 :

    data_9=data[data['Store']==i]

    ax[4,0].plot(data_9['Date'], data_9['Weekly_Sales'])     



    

ax[0,0].set_title('Mean sales record by store(0~4)')

ax[0,1].set_title('Mean sales record by store(5~9)')

ax[1,0].set_title('Mean sales record by store(10~14)')

ax[1,1].set_title('Mean sales record by store(15~19)')

ax[2,0].set_title('Mean sales record by store(20~24)')

ax[2,1].set_title('Mean sales record by store(25~29)')

ax[3,0].set_title('Mean sales record by store(30~34)')

ax[3,1].set_title('Mean sales record by store(35~39)')

ax[4,0].set_title('Mean sales record by store(40~)')







ax[0,0].set_ylabel('Mean sales')

ax[0,0].set_xlabel('Date')

ax[0,1].set_ylabel('Mean sales')

ax[0,1].set_xlabel('Date')

ax[1,0].set_ylabel('Mean sales')

ax[1,0].set_xlabel('Date')

ax[1,1].set_ylabel('Mean sales')

ax[1,1].set_xlabel('Date')

ax[2,0].set_ylabel('Mean sales')

ax[2,0].set_xlabel('Date')

ax[2,1].set_ylabel('Mean sales')

ax[2,1].set_xlabel('Date')

ax[3,0].set_ylabel('Mean sales')

ax[3,0].set_xlabel('Date')

ax[3,1].set_ylabel('Mean sales')

ax[3,1].set_xlabel('Date')

ax[4,0].set_ylabel('Mean sales')

ax[4,0].set_xlabel('Date')







plt.show()
grouped=train.groupby(['Store','Dept'])['Weekly_Sales'].max().reset_index()

grouped['Store']=grouped['Store'].astype(str)

grouped['Dept']=grouped['Dept'].astype(str)

grouped['Weekly_Sales']=grouped['Weekly_Sales'].astype(str)

grouped['key']=grouped['Weekly_Sales'] +'_'+ grouped['Store'] +'_'+ grouped['Dept']

grouped.head()



train['Store']=train['Store'].astype(str)

train['Dept']=train['Dept'].astype(str)

train['Weekly_Sales_2']=train['Weekly_Sales'].astype(str)

train['key']=train['Weekly_Sales'].astype(str) +'_'+ train['Store'].astype(str) +'_'+ train['Dept'].astype(str)

train



train_2=pd.merge(train, grouped['key'], how='inner', on='key' )

train_2['Date_2']=train_2['Month'].astype(str) + '-' + train_2['Day'].astype(str)

train_2



grouped_2=train_2.groupby(['Date_2','Store','Dept']).count().reset_index()

grouped_2.sort_values('Weekly_Sales',ascending=False,inplace=True)



grouped_2['key_2']=grouped_2['Date_2'].astype(str) + grouped_2['Store'].astype(str) + grouped_2['Dept'].astype(str)

grouped_2['Count']=grouped_2['Weekly_Sales']

data=grouped_2[['key_2','Count']]



train['Date_2']=train['Month'].astype(str) + '-' + train['Day'].astype(str)

train['key_2']=train['Date_2'].astype(str) + train['Store'].astype(str) + train['Dept'].astype(str)

train=pd.merge(train, data, how='left', on='key_2' )

train.loc[train['Count'].isnull(),'Count']=0



#grouped_2['proportion']=grouped_2['Weekly_Sales']/sum(grouped_2['Store'])

#grouped_2['Count']=grouped_2['Weekly_Sales']

#data=grouped_2[['Date_2','Count']]

#print(data.head(100))



#train['Date_2']=train['Month'].astype(str) + '-' + train['Day'].astype(str)



#train=pd.merge(train, data, how='left', on='Date_2' )

#train.head(150)
data = pd.concat([train['Count'], train['Weekly_Sales'], train['Store']], axis=1)

f, ax = plt.subplots(figsize=(5, 5))

fig=sns.boxplot(x='Count', y="Weekly_Sales", data=data, showfliers=False)