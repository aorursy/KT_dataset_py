import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
cust_data = pd.read_csv('../input/new-york-ecommerce-customers/Ecommerce Customers')
cust_data.head()
cust_data.describe()
cust_data.info()
cust_data.isnull().sum()

#the dataset has no null values. 
plt.rcParams["patch.force_edgecolor"] = True

sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=cust_data,kind='hex')
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=cust_data,kind='hex')
sns.heatmap(cust_data.corr(),cmap = 'GnBu', annot=True)
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=cust_data)

plt.xlim(0,cust_data['Length of Membership'].max()+1)
cust_data.columns
X = cust_data[[ 'Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

y = cust_data['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train,y_train)
predictions = linear_model.predict(X_test)
plt.scatter(y_test,predictions,edgecolors='r')

plt.xlabel('Y test')

plt.ylabel('Predicted Y values')
cust_coef_df = pd.DataFrame(linear_model.coef_,index=X.columns,columns=['Coefficient'])
cust_coef_df