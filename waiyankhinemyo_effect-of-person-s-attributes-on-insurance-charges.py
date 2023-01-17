#import libraries 
#structures
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

#analysis
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load dataset
data = '../input/insurance/insurance.csv'
dataset = pd.read_csv(data)
dataset.shape
dataset.dtypes
dataset.describe()
dataset.head()
dataset.isnull().sum()
#check for unreasonable data
dataset.applymap(np.isreal)
dataset.sex.unique()
dataset.smoker.unique()
dataset.region.unique()
le = LabelEncoder()
X = dataset
a = dataset['sex']
b = dataset['smoker']
c = dataset['region']
X['sex'] = le.fit_transform(X['sex'])

a = le.transform(a)
X['smoker'] = le.fit_transform(X['smoker'])

b = le.transform(b)
X['region'] = le.fit_transform(X['region'])

c = le.transform(c)
dataset = X
dataset.head()
dataset.dtypes
sns_plot = sns.pairplot(dataset)
sns_plot = sns.distplot(dataset['charges'])
#set x and y
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = dataset.iloc[:,0:6]
y = dataset['charges']

#stadardize data
X_scaled = StandardScaler().fit_transform(X)

#get feature names
X_columns = dataset.columns[:6]

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)
dataset.head()
#get correlation map
corr_mat=dataset.corr()
#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# import model
from sklearn.linear_model import LinearRegression

#instantiate
linReg = LinearRegression()

start_time = time.time()
# fit out linear model to the train set data
linReg_model = linReg.fit(X_train, y_train)
today = date.today()
print("--- %s seconds ---" % (time.time() - start_time))
#get coefficient values
coeff_df = pd.DataFrame(linReg.coef_, X_columns, columns=['Coefficient'])  
coeff_df
#validate model
y_pred = linReg.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(10)
df1.plot(kind='bar',figsize=(8,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print the intercept and coefficients
print('Intercept: ',linReg.intercept_)
print('r2 score: ',linReg.score(X_train, y_train))
# define input
X2 = sm.add_constant(X)

# create a OLS model
model = sm.OLS(y, X2)

# fit the data
est = model.fit()
# make some confidence intervals, 95% by default
est.conf_int()
print(est.summary())
