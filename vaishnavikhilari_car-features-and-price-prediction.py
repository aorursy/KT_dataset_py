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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px



#importing libraries to use the library functions
df=pd.read_csv("../input/cardataset/data.csv")

#df contains the file information  which is in csv format
df
df.columns

#listing the column names of the dataset/dataframe
df.dtypes

#checking the datatypes of different columns of the dataframe
df.info()

#getting the information of dataframe such as no. of entries,data columns,non-null count,data types,etc
df.shape

#shape of the dataframe ie no. of rows and columns
df.describe()

#checking for statistical summary such as count,mean,etc. of numeric columns
df.drop(df[df['MSRP'] == 0].index,inplace=True)

#dropping rows which have zero as a value for MSRP column as it is our dependent/target variable.
df.shape
df.drop(['Market Category'], axis=1, inplace=True)

#dropping 'market category' column as MSRP is independent of it and hence not useful in predicting price of car. 
df.shape
df=df.rename(columns={'Engine HP':'HP','Engine Cylinders':'Cylinders','Transmission Type':'Transmission','Driven_Wheels':'Drive Mode','highway MPG':'MPG-H','city mpg':'MPG-C','MSRP':'Price'})

#renaming the column names as per mentioned in the steps of the problem statement
df
df.duplicated().sum()

#checking for any duplicates in the data
df.drop_duplicates(keep=False,inplace=True)

#removing the duplicates in the data
df
df.isnull().sum()

#checking for any null values in the data
df.dropna(inplace=True,axis=0)

#removing the null values in the data
df.isnull().sum()

#verfying for any null values
sns.boxplot(data=df,orient='h',palette='Set2')

#checking for any outliers in the data
df.drop(df[df['Price'] >= 500000].index,inplace=True)

#removing the unnecessary data points from the dataset
df
sns.boxplot(x=df['Price'])
sns.boxplot(x=df['Cylinders'])
sns.boxplot(x=df['HP'])
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df = df[~((df < (Q1 - 1.5*IQR))|(df > (Q3 + 1.5*IQR))).any(axis = 1)]

df.shape
counts=df['Make'].value_counts()*100/sum(df['Make'].value_counts())

#calculating percentage of each brand
popular_labels=counts.index[:10]



colors=['lightslategray',]*len(popular_labels)

colors[0]='crimson'



fig=go.Figure(data=[go.Bar(x=counts[:10],y=popular_labels,marker_color=colors,orientation='h')])

fig.update_layout(title_text='Most represented Car Brands in the Dataset',xaxis_title="Percentage",yaxis_title="Car Brand")

#plotting the top 10 brands represented in the dataset
prices = df[['Make','Price']].loc[(df['Make'].isin(popular_labels))].groupby('Make').mean()

print(prices)

#calculating the average price of top 10 brands represented in the dataset
display_p=df[['Make','Year','Price']].loc[(df['Make'].isin(popular_labels))]



fig=px.box(display_p,x="Make",y="Price")

fig.update_layout(title_text='Average Price over 10 most represented Car Brands',xaxis_title="Make",yaxis_title="Average Price")
df.corr()
df_corr=df.corr()

f,ax=plt.subplots(figsize=(12,7))

sns.heatmap(df_corr,cmap='viridis',annot=True)

plt.title("Correlation between features",weight='bold',fontsize=18)

plt.show()



#plotting the heatmap for different features
fig,ax = plt.subplots(figsize=(12,7))

ax.scatter(df['HP'],df['Price'])

ax.set_xlabel('HP')

ax.set_ylabel('Price')

plt.show()
fig,ax = plt.subplots(figsize=(12,7))

ax.scatter(df['HP'],df['Price'])

ax.set_xlabel('HP')

ax.set_ylabel('Price')

plt.show()
fig,ax = plt.subplots(figsize=(12,7))

ax.scatter(df['HP'],df['Cylinders'])

ax.set_xlabel('HP')

ax.set_ylabel('Cylinders')

plt.show()
#creating new column 'Price Range' for easy visualization

def getrange(Price):

    if (Price >= 0 and Price < 25000):

        return '0 - 25000'

    if (Price >= 25000 and Price < 50000):

        return '25000 - 50000'

    if (Price >= 50000 and Price < 75000):

        return '50000 - 75000'

    if (Price >= 75000 and Price < 100000):

        return '75000 - 100000'

       

df['Price Range'] = df.apply(lambda x:getrange(x['Price']),axis = 1)



df['Price Range'].value_counts()
#distribution of number of cars over the years

dic = {1990+i : sum(df['Year']==1990+i) for i in range(28)}

x_dic = [1990 + i for i in range(28)]

y_dic = [dic[1990 + i] for i in range(28)]



# Plot

fig = go.Figure([go.Bar(x=x_dic, y=y_dic)])



fig.update_layout(title="Car year distribution",

                  xaxis_title="Year",

                  yaxis_title="Count Cars sold")





fig.show()
plt.rcParams['figure.figsize'] = (15,9)



x = pd.crosstab(df['Price Range'],df['Engine Fuel Type'])

color = plt.cm.copper(np.linspace(0,1,9))

x.div(x.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = True ,color=color)

plt.title("Price vs Engine Fuel Type",fontweight = 30,fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15,9)

x = pd.crosstab(df['Price Range'],df['Drive Mode'])

x.div(x.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = False)

plt.title('Price vs Drive Mode',fontweight = 30,fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15,9)

x = pd.crosstab(df['Price Range'],df['Vehicle Size'])

x.div(x.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = False)

plt.title('Price vs Size',fontweight = 30,fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (15,9)



x = pd.crosstab(df['Price Range'],df['Vehicle Style'])

x.div(x.sum(1).astype(float),axis = 0).plot(kind = 'bar',stacked = True)

plt.title("Price vs Vehicle Style",fontweight = 30,fontsize = 20)

plt.show()
data_pie = df['Transmission'].value_counts()



fig = go.Figure(data=[go.Pie(labels=data_pie.index, values=data_pie.tolist(), textinfo='label+percent',insidetextorientation='radial')])



fig.update_traces(hole=.3, hoverinfo="label+percent+name")
df.head()
df.shape
# performing label encoding to the categorical columns

columns_to_convert=['Make','Model','Engine Fuel Type','Transmission','Drive Mode','Vehicle Size','Vehicle Style','Price Range']

df[columns_to_convert] = df[columns_to_convert].astype('category')
df.dtypes
from sklearn import preprocessing

  

# label_encoder object knows how to understand word labels.

label_encoder = preprocessing.LabelEncoder()

  

# Encode labels in column 'species'.

for col in ['Make','Model','Engine Fuel Type','Transmission','Drive Mode','Vehicle Size','Vehicle Style','Price Range']: df[col] = label_encoder.fit_transform(df[col])
df.head()
# splitting the dependent and independent variables



x = df[['Popularity','Year','HP','Cylinders','MPG-H','MPG-C']].values

y = df['Price'].values



print(x.shape)

print(y.shape)
#normalizing the data

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

sc_y=StandardScaler()



x=sc_x.fit_transform(x)

y=sc_y.fit_transform(y.reshape(-1,1))
# splitting the dataset into training and test sets



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression



lr_model = LinearRegression()

lr_model.fit(x_train,y_train)



# calculating the accuracies

print("Training Accuracy :",lr_model.score(x_train,y_train))

print("Testing Accuracy :",lr_model.score(x_test,y_test))
y_pred = lr_model.predict(x_test)

y_pred[0:5]
plt.scatter(y_test,y_pred)

plt.xlabel("True Values")

plt.ylabel("Predicted Values")
sns.distplot((y_test-y_pred),bins=50)
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

import math



print("R2_Score : ", r2_score(y_test,y_pred))

print("Mean Squared Error : ", mean_squared_error(y_test,y_pred))

print("MAE : ",mean_absolute_error(y_test,y_pred))

print("RSME : ",math.sqrt(mean_squared_error(y_test,y_pred)))
results_df = pd.DataFrame(data=[["Linear Regression", lr_model.score(x_train,y_train),lr_model.score(x_test,y_test),r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),math.sqrt(mean_squared_error(y_test,y_pred))]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %','r2 score','MSE','MAE','RSME'])



results_df
from sklearn.svm import SVR

svr_model=SVR(kernel = 'rbf')

svr_model.fit(x_train,y_train)



# calculating the accuracies

print("Training Accuracy :",svr_model.score(x_train,y_train))

print("Testing Accuracy :",svr_model.score(x_test,y_test))
y_pred = svr_model.predict(x_test)

y_pred[0:5]
plt.scatter(y_test,y_pred)

plt.xlabel("True Values")

plt.ylabel("Predicted Values")
sns.distplot((y_test-y_pred),bins=50)
print("R2_Score : ", r2_score(y_test,y_pred))

print("Mean Squared Error : ", mean_squared_error(y_test,y_pred))

print("MAE : ",mean_absolute_error(y_test,y_pred))

print("RSME : ",math.sqrt(mean_squared_error(y_test,y_pred)))
results_df_2 = pd.DataFrame(data=[["Support Vector Machine", svr_model.score(x_train,y_train),svr_model.score(x_test,y_test),r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),math.sqrt(mean_squared_error(y_test,y_pred))]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %','r2 score','MSE','MAE','RSME'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.ensemble import RandomForestRegressor



rfc_model=RandomForestRegressor(n_estimators=300,random_state=0)

rfc_model.fit(x_train,y_train)



# calculating the accuracies

print("Training Accuracy :",rfc_model.score(x_train,y_train))

print("Testing Accuracy :",rfc_model.score(x_test,y_test))
y_pred = rfc_model.predict(x_test)

y_pred[0:5]
plt.scatter(y_test,y_pred)

plt.xlabel("True Values")

plt.ylabel("Predicted Values")
sns.distplot((y_test-y_pred),bins=50)
print("R2_Score : ", r2_score(y_test,y_pred))

print("Mean Squared Error : ", mean_squared_error(y_test,y_pred))

print("MAE : ",mean_absolute_error(y_test,y_pred))

print("RSME : ",math.sqrt(mean_squared_error(y_test,y_pred)))
results_df_2 = pd.DataFrame(data=[["Random Forest", rfc_model.score(x_train,y_train),rfc_model.score(x_test,y_test),r2_score(y_test,y_pred),mean_squared_error(y_test,y_pred),mean_absolute_error(y_test,y_pred),math.sqrt(mean_squared_error(y_test,y_pred))]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %','r2 score','MSE','MAE','RSME'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df