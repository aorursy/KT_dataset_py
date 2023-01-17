# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline
data=pd.read_csv('../input/nyc-rolling-sales.csv')
data.head()
data.info()
data.describe()
data.columns
x = data['SALE PRICE']==' -  '

data=data[x==False]

data.head()





data['SALE PRICE'].mean()

data['SALE PRICE'].describe()


#skewness and kurtosis`

print("Skewness:%f"% data['SALE PRICE'].skew())

print("Kurtosis %f" % data['SALE PRICE'].kurt())
data['SALE PRICE'] = data['SALE PRICE'].convert_objects(convert_numeric=True)

plt.subplot(121)

sns.kdeplot(data['SALE PRICE'],shade=True)

plt.xlabel('Sale price')





x = np.log(data['SALE PRICE'])

sns.kdeplot(x,shade= True)

plt.xlabel('Log (SALE PRICE)')

# correlation weith numerical valuable

corr = data.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()

sns.heatmap(corr,vmax=1,square= True)
#scatter plot with land square feet and sale price

ydata = data['SALE PRICE'] 

data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].convert_objects(convert_numeric= True)

data1 = pd.concat((data['LAND SQUARE FEET'],ydata),axis=1)

sns.regplot(data1['LAND SQUARE FEET'],ydata,data=data1)
data['GROSS SQUARE FEET']=data['GROSS SQUARE FEET'].convert_objects(convert_numeric= True)

ydata=(data['SALE PRICE'] ==0).sum()

data1 = data[data['SALE PRICE'] > 21493.722932]

yd=data1['SALE PRICE']



xdata=data1['GROSS SQUARE FEET']

data1

yd.plot()

corr_list = corr['SALE PRICE'].sort_values(axis=0,ascending = False).iloc[1:]

corr_list
plt.figure(figsize = (30,12))

sns.boxplot( x = 'BUILDING CLASS AT PRESENT', y = 'SALE PRICE', data=data)

xt = plt.xticks(rotation = 90)
sns.stripplot(x='YEAR BUILT',y ='SALE PRICE',data=data)