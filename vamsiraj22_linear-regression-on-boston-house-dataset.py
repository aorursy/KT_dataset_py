import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import seaborn as sns

import matplotlib 

import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, linear_model

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
cols=['CRIM',

'ZN',

'INDUS',

'CHAS',

'NOX',

'RM',

'AGE',

'DIS',

'RAD',

'TAX',

'PTRATIO',

'B',

'LSTAT',

'MEDV']

df=pd.read_csv(r'/kaggle/input/boston-house-prices/housing.csv',header=None,delim_whitespace=True)

df.columns=cols

price=df['MEDV']

#df.drop('MEDV', axis=1, inplace=True)

df
pplot=sns.pairplot(df[:-1])

pplot.fig.set_size_inches(15,15)
scaler = preprocessing.StandardScaler()

df_stand = scaler.fit_transform(df)

df_stand=pd.DataFrame(df_stand,columns=cols)

df=df_stand
fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(df.corr(), center=0, cmap='BrBG',annot=True)

ax.set_title('Multi-Collinearity of Features')


print(df.columns)

plt.figure(num=None, figsize=(15,30), dpi=80, facecolor='w', edgecolor='k')

plt.figure(1)

var=1

for index,feature in enumerate(list(df.columns)):

    plt.subplot(4,4,index+1,xlabel=feature)

    sns.boxplot(y=df[feature])

    plt.grid()

    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    plt.subplots_adjust(top=0.92, bottom=0, left=0.10, right=0.95, hspace=0.5,wspace=0.5)

    var+=1


medv_correlation={}

for i in df.columns:

    if i!='MEDV':

        corr_value=df['MEDV'].corr(df[i]) 

        medv_correlation[i]=corr_value

        print("Correlation value between MEDV and {} is {}".format(i,corr_value))

print("MAX correlated features are:", max(medv_correlation, key=medv_correlation.get),min(medv_correlation, key=medv_correlation.get))
percentage=0

for k, v in df.items():

    if k=='RM':

        Q1 = np.array(v.quantile(0.25))

        Q3 = np.array(v.quantile(0.75))

        IQR = Q3 - Q1

        v_col = v[(v <= Q1 - 1.5 * IQR) | (v >= Q3 + 1.5 * IQR)]

        percentage = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]

        print("Column %s outliers = %.2f%%" % (k, percentage))

print("Total number of outliers for feature RM is %d  which is %.2f%% of total data points"%(v_col.count(),percentage)) 



df=df[~(df['RM'] >= Q3 + 1.5 * IQR)|(df['RM'] <=Q1 - 1.5 * IQR)]

df.shape
df
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

scaler = preprocessing.StandardScaler()

df_stand = scaler.fit_transform(df)

df_stand=pd.DataFrame(df_stand,columns=cols)

df_stand
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculating VIF

s=df._get_numeric_data()

vif = pd.DataFrame()

vif["variables"] = s.columns



vif["VIF"] = [variance_inflation_factor(np.array(s.values,dtype=float), i) for i in range(s.shape[1])]
vif['Result'] = vif['VIF'].apply(lambda x: 'Non Collinear' if x <= 3.7 else 'Collinear')

vif
features=list(vif[vif['Result']=='Non Collinear']['variables'])



print(features)
for i in df.columns:

    if i!='MEDV':

        if i not in features:

            df.drop(i,axis=1,inplace=True)

df
from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')

plt.figure(1)

var=1

for index,feature in enumerate(list(df.columns)):

    colors = (0,0,0)

    area = np.pi*3

    plt.subplot(4,4,index+1,xlabel=feature,ylabel='MEDV')

    sns.regplot(y='MEDV',x=feature,data=df,marker="+",ci=80)

    plt.grid()

    plt.subplots_adjust(top=0.92, bottom=0, left=0.10, right=0.95, hspace=0.5,wspace=0.5)

    var+=1
df_plot=pd.read_csv(r'/kaggle/input/boston-house-prices/housing.csv',header=None,delim_whitespace=True)

df_plot.columns=cols





for i in df.columns:

    if i!='MEDV':

        if i not in features:

            df_plot.drop(i,axis=1,inplace=True)



from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 8), dpi=80, facecolor='w', edgecolor='k')

plt.figure(1)

var=1

for index,feature in enumerate(list(df.columns)[:-1]):

    counts,bin_edges= np.histogram(df_plot[feature],bins=10,density=True)

    pdf=counts/sum(counts)

    #a=440+var

    cdf=np.cumsum(pdf)

    plt.subplot(4,4,index+1,xlabel=feature)

    plt.plot(bin_edges[1:],pdf)

    plt.plot(bin_edges[1:],cdf)

    plt.grid()

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,wspace=0.5)

    var+=1

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics





y=df_stand['MEDV'] 

df_stand.drop('MEDV',axis=1,inplace=True)

X=df_stand

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.regplot(y_test,y_pred)