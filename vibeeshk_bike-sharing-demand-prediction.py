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
import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

#from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from sklearn.linear_model import Ridge

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import  GridSearchCV

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
filepath= '/kaggle/input/bike-sharing-demand/train.csv'

filepath1= '/kaggle/input/bike-sharing-demand/test.csv'

testdata= pd.read_csv(filepath1)

testdata2=testdata

traindata= pd.read_csv(filepath)

traindata.head()
#We remove the id column in both the training and testing datasets.

traindata=traindata.drop('datetime',axis=1)

testdata=testdata.drop('datetime',axis=1)



#We also remove the casual and registered columns because they are not present in the test dataset

traindata=traindata.drop('casual',axis=1)

traindata=traindata.drop('registered',axis=1)
#working with numeric features (They are all numerical features)

numeric_features = traindata.select_dtypes(include=[np.number])

numeric_features.dtypes
#Checking the correlation between each column with the Cover_Type

corr = numeric_features.corr()

print (corr['count'].sort_values(ascending=False), '\n')

print (corr['count'].sort_values(ascending=False))

#We see how various features compare with the Cover type



column_names=['temp','atemp','windspeed']                





for i in column_names:

    plt.scatter(x=traindata[i], y=traindata['count'])

    plt.ylabel('count')

    plt.xlabel(i)

    plt.show()
#Lettuce visualize the other columns and see how they relate with counts



col=['season','holiday','workingday','weather']

for i in col:

    

    sns.factorplot(x=i,y="count",data=traindata,kind='bar',size=5,aspect=1.5)

traindata.temp.unique()

fig,axes=plt.subplots(2,2)

axes[0,0].hist(x="temp",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,0].set_title("Variation of windspeed")

axes[1,1].hist(x="humidity",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,1].set_title("Variation of humidity")

fig.set_size_inches(10,10)
#Now we will visualise the remaining features and compare them with the number of rentals

column_names=['season','holiday','workingday','weather']                



for i in column_names:

    feature = traindata.pivot_table(index=i,

                  values='count')

    feature.plot(kind='bar', color='blue')

    plt.xlabel(i)

    plt.ylabel('counts')

    plt.xticks(rotation=0)

    plt.show()
traindata.temp.unique()

fig,axes=plt.subplots(2,2)

axes[0,0].hist(x="temp",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,0].set_title("Variation of windspeed")

axes[1,1].hist(x="humidity",data=traindata,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,1].set_title("Variation of humidity")

fig.set_size_inches(10,10)
#Split the data into train and test

y=traindata['count']

x=traindata.drop('count',axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)


#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x_train, y_train)

y_predicted = linearRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# Decision Tree - CART

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(x_train, y_train)

y_predicted_d = regressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_d)

r = r2_score(y_test, y_predicted_d)

mae = mean_absolute_error(y_test,y_predicted_d)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
#Polynomial Regression

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(x_train)

x_poly_test = polynomial_features.fit_transform(x_test)

model = LinearRegression()

model.fit(x_poly, y_train)

y_predicted_p = model.predict(x_poly_test)

mse = mean_squared_error(y_test, y_predicted_p)

r = r2_score(y_test, y_predicted_p)

mae = mean_absolute_error(y_test,y_predicted_p)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
#for random forest regresion.  (tuning)

no_of_test=[500]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

mse = mean_squared_error(y_test, pred)

r = r2_score(y_test, pred)

mae = mean_absolute_error(y_test,pred)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# for KNN  (tuning)



n_neighbors=[]

for i in range (0,50,5):

    if(i!=0):

        n_neighbors.append(i)

params_dict={'n_neighbors':n_neighbors,'n_jobs':[-1]}

clf_knn=GridSearchCV(estimator=KNeighborsRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_knn.fit(x_train,y_train)

pred=clf_knn.predict(x_test)

mse = mean_squared_error(y_test, pred)

r = r2_score(y_test, pred)

mae = mean_absolute_error(y_test,pred)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# Thus we can use RandomForest Regresson.



no_of_test=[500]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf.fit(x,y)

Prediction=clf_rf.predict(testdata)
predictionlist=Prediction.tolist()

counts=testdata2['datetime'].tolist() 

output=pd.DataFrame(list(zip(counts, predictionlist)),

              columns=['datetime','count'])

output.head()

output.to_csv('my_submission(ikeSharingDemand).csv', index=False)