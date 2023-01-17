#import the necessary libraries required 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn import metrics



#%matplotlib notebook

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
#read the data file

housing_data = pd.read_csv("../input/housing.csv")
print("The number of rows and colums are {} and also called shape of the matrix".format(housing_data.shape))

print("Columns names are \n {}".format(housing_data.columns))

print(housing_data.head())

print(housing_data.tail())
print(housing_data.dtypes)
#display scatter_matrix also

fig = plt.figure()

scatter_matrix(housing_data,figsize =(25,25),alpha=0.9,diagonal="kde",marker="o");

housing_data.hist(figsize=(25,25),bins=50);
hcorr = housing_data.corr()

hcorr.style.background_gradient()
#heatmap using seaborn

#set the context for plotting 

sns.set(context="paper",font="monospace")

housing_corr_matrix = housing_data.corr()

#set the matplotlib figure

fig, axe = plt.subplots(figsize=(12,8))

#Generate color palettes 

cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)

#draw the heatmap

sns.heatmap(housing_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True );
def getOutliers(dataframe,column):

    column = "total_rooms" 

    #housing[column].plot.box(figsize=(8,8))

    des = dataframe[column].describe()

    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}

    Q1 = des[desPairs['25']]

    Q3 = des[desPairs['75']]

    IQR = Q3-Q1

    lowerBound = Q1-1.5*IQR

    upperBound = Q3+1.5*IQR

    print("(IQR = {})Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))

    #b = df[(df['a'] > 1) & (df['a'] < 5)]

    data = dataframe[(dataframe [column] < lowerBound) | (dataframe [column] > upperBound)]



    print("Outliers out of total = {} are \n {}".format(housing_data[column].size,len(data[column])))

    #remove the outliers from the dataframe

    outlierRemoved = housing_data[~housing_data[column].isin(data[column])]

    return outlierRemoved
#get the outlier

df_outliersRemoved = getOutliers(housing_data,"total_rooms")
#check wheather there are any missing values or null

housing_data.isnull().sum()
#Total_bedrooms columns is having 207 missing values

#Now we need to impute the missing values
#statistics for missing values

print ("Total_bedrooms column Mode is  "+str(housing_data["total_bedrooms"].mode())+"\n")

print(housing_data["total_bedrooms"].describe())
total_bedroms = housing_data[housing["total_bedrooms"].notnull()]["total_bedrooms"]#["total_bedrooms"]

total_bedroms.hist(figsize=(12,8),bins=50)
print(housing_data.iloc[:,4:5].head())

imputer = Imputer(np.nan,strategy ="median")

imputer.fit(housing_data.iloc[:,4:5])

housing_data.iloc[:,4:5] = imputer.transform(housing_data.iloc[:,4:5])

housing_data.isnull().sum()
## Label encode for categorical feature (ocean_proximity)
labelEncoder = LabelEncoder()

print(housing_data["ocean_proximity"].value_counts())

housing_data["ocean_proximity"] = labelEncoder.fit_transform(housing_data["ocean_proximity"])

housing_data["ocean_proximity"].value_counts()

housing_data.describe()
housing_ind = housing_data.drop("median_house_value",axis=1)

print(housing_ind.head())

housing_dep = housing_data["median_house_value"]

print("Medain Housing Values")

print(housing_dep.head())
#check for rand_state

X_train,X_test,y_train,y_test = train_test_split(housing_ind,housing_dep,test_size=0.2,random_state=42)

#print(X_train.head())

#print(X_test.head())

#print(y_train.head())

#print(y_test.head())

print("X_train shape {} and size {}".format(X_train.shape,X_train.size))

print("X_test shape {} and size {}".format(X_test.shape,X_test.size))

print("y_train shape {} and size {}".format(y_train.shape,y_train.size))

print("y_test shape {} and size {}".format(y_test.shape,y_test.size))

X_train.head()
#Standardize training and test datasets.

#==============================================================================

# Feature scaling is to bring all the independent variables in a dataset into

# same scale, to avoid any variable dominating  the model. Here we will not 

# transform the dependent variables.

#==============================================================================

independent_scaler = StandardScaler()

X_train = independent_scaler.fit_transform(X_train)

X_test = independent_scaler.transform(X_test)

print(X_train[0:5,:])

print("test data")

print(X_test[0:5,:])
#initantiate the linear regression

linearRegModel = LinearRegression(n_jobs=-1)

#fit the model to the training data (learn the coefficients)

linearRegModel.fit(X_train,y_train)

#print the intercept and coefficients 

print("Intercept is "+str(linearRegModel.intercept_))

print("coefficients  is "+str(linearRegModel.coef_))
#predict on the test data

y_pred = linearRegModel.predict(X_test)
print(len(y_pred))

print(len(y_test))

print(y_pred[0:5])

print(y_test[0:5])

test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind='reg',);
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print(np.sqrt(metrics.mean_squared_error(y_train,linearRegModel.predict(X_train))))

dtReg = DecisionTreeRegressor(max_depth=9)

dtReg.fit(X_train,y_train)
dtReg_y_pred = dtReg.predict(X_test)

dtReg_y_pred
print(len(dtReg_y_pred))

print(len(y_test))

print(dtReg_y_pred[0:5])

print(y_test[0:5])
print(np.sqrt(metrics.mean_squared_error(y_test,dtReg_y_pred)))
test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
rfReg = RandomForestRegressor(30)

rfReg.fit(X_train,y_train)
rfReg_y_pred = rfReg.predict(X_test)

print(len(rfReg_y_pred))

print(len(y_test))

print(rfReg_y_pred[0:5])

print(y_test[0:5])
print(np.sqrt(metrics.mean_squared_error(y_test,rfReg_y_pred)))
test = pd.DataFrame({'Predicted':dtReg_y_pred,'Actual':y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg")
#Extract median_income 

dropcol = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","ocean_proximity"]

print(dropcol)

housing_med = housing_ind.drop(dropcol,axis=1)

print(type(housing_med))
#check for rand_state

X_train2,X_test2,y_train2,y_test2 = train_test_split(housing_med,housing_dep,test_size=0.2,random_state=42)

#print(X_train.head())

#print(X_test.head())

#print(y_train.head())

#print(y_test.head())

print("X_train2 shape {} and size {}".format(X_train2.shape,X_train2.size))

print("X_test2 shape {} and size {}".format(X_test2.shape,X_test2.size))

print("y_train2 shape {} and size {}".format(y_train2.shape,y_train2.size))

print("y_test2 shape {} and size {}".format(y_test2.shape,y_test2.size))
linReg2 = LinearRegression()

linReg2.fit(X_train2,y_train2)
y_pred2 = linReg2.predict(X_test2)

print(len(y_pred2))

print(len(y_test2))

print(y_pred2[0:5])

print(y_test2[0:5])
fig = plt.figure(figsize=(25,8))

plt.scatter(y_test2,y_pred2,marker="o",edgecolors ="r",s=60)

plt.scatter(y_train2,linReg2.predict(X_train2),marker="+",s=50,alpha=0.5)

plt.xlabel(" Actual median_house_value")

plt.ylabel(" Predicted median_house_value")