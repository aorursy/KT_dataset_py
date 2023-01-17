# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
telco = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
telco.tail()



# lets take a look
filt = telco['Churn'] == 'Yes'

gone = telco[filt]

stayed = telco[~filt]
gone.tail()
stayed.tail()
telco.dtypes
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize']=20,10
# I will look at tenure. The no. of months the customer has been with the company.





x = telco['tenure']

y = gone['tenure']

z = stayed['tenure']



fig,((ax1, ax2, ax3)) = plt.subplots(1,3)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
# I will look at monthly charges.



x = telco['MonthlyCharges']

y = gone['MonthlyCharges']

z = stayed['MonthlyCharges']



fig,((ax1, ax2, ax3)) = plt.subplots(1,3)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')

gone['TotalCharges'] = pd.to_numeric(gone['TotalCharges'], errors='coerce')

stayed['TotalCharges'] = pd.to_numeric(stayed['TotalCharges'], errors='coerce')



# this is here because it was taking too long
# I will look at monthly charges.



rcParams['figure.figsize']= 10,20



x = telco['TotalCharges']

y = gone['TotalCharges']

z = stayed['TotalCharges']

fig,((ax1, ax2, ax3)) = plt.subplots(3,1)



ax1.hist(x)

ax1.set_title('Telco')

ax1.set_xticks(np.arange(0,7000,1000))



ax2.hist(y)

ax2.set_title('Gone')

ax2.set_xticks(np.arange(0,3000,1000))



ax3.hist(z)

ax3.set_title('Stayed')

ax3.set_xticks(np.arange(0,7000,1000))
# I'll make a new column with five cateogies for Monthly charges.

mc = telco['MonthlyCharges']
print(max(mc), min(mc))
telco['Monthly'] = 'under40'
telco['MonthlyCharges']
filt=telco['MonthlyCharges']>40

telco.loc[filt, 'Monthly']='20-40'

filt=telco['MonthlyCharges']>60

telco.loc[filt, 'Monthly']='40-60'
filt=telco['MonthlyCharges']>80

telco.loc[filt, 'Monthly']='80-100'
filt=telco['MonthlyCharges']>100

telco.loc[filt, 'Monthly']='over 100'
telco['Monthly']
# I'll make a new column with five cateogies for Monthly charges.

t = telco['tenure']
max(t)
telco['New_tenure'] = 'Long-standing'
filt=telco['tenure']<24

telco.loc[filt, 'New_tenure']='1-2 Years'
filt=telco['tenure']<12

telco.loc[filt, 'New_tenure']='1st year'
tc= telco['TotalCharges']
telco['total'] = 'big_spenders'
filt=telco['TotalCharges']<5000

telco.loc[filt, 'total']='regulars'
filt=telco['TotalCharges']<1000

telco.loc[filt, 'total']='newbies'
telco['total'].value_counts()
telco['New_tenure'].value_counts()
telco['Monthly'].value_counts()
telco.columns
x = telco['gender']

y = gone['gender']

z = stayed['gender']



rcParams['figure.figsize']= 20,5



fig,((ax1, ax2, ax3)) = plt.subplots(1,3)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
x = telco['SeniorCitizen']

y = gone['SeniorCitizen']

z = stayed['SeniorCitizen']



rcParams['figure.figsize']= 20,5



fig,((ax1, ax2, ax3)) = plt.subplots(1,3)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
x = telco['Partner']

y = gone['Partner']

z = stayed['Partner']



rcParams['figure.figsize']= 20,5



fig,((ax1, ax2, ax3)) = plt.subplots(1,3)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
telco['PaymentMethod'].value_counts()
x = telco['PaymentMethod']

y = gone['PaymentMethod']

z = stayed['PaymentMethod']



rcParams['figure.figsize']= 10,20



fig,((ax1, ax2, ax3)) = plt.subplots(3,1)



ax1.hist(x)

ax1.set_title('Telco')



ax2.hist(y)

ax2.set_title('Gone')



ax3.hist(z)

ax3.set_title('Stayed')
telco.head()
pd.set_option('display.max_columns', 50)

telco.head()
telco.drop(['customerID','tenure','MonthlyCharges','TotalCharges'], axis=1, inplace=True)
telco = pd.get_dummies(telco,drop_first=False)



telco.drop('Churn_No', axis=1, inplace=True)

telco.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(telco.drop('Churn_Yes', axis=1), telco['Churn_Yes'])
from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()

LogReg.fit(x_train, y_train)
LogReg.score(x_train, y_train)
LogReg.score(x_test, y_test)
LogReg_summary = pd.DataFrame(x_train.columns.values, columns=['Features'])
coefs = LogReg.coef_

coefs.shape
coefs = coefs.reshape(53,1)
LogReg_summary['coefs'] = coefs
from sklearn.feature_selection import f_regression
f_regression(x_train, y_train)

p_values = f_regression(x_train, y_train)[1]

LogReg_summary['p_value'] = p_values.round(4)

LogReg_summary.sort_values('coefs')