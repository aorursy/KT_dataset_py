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

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from sklearn.linear_model import Ridge

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import  GridSearchCV
filepath= '/kaggle/input/forest-cover-type-prediction/train.csv'

filepath1= '/kaggle/input/forest-cover-type-prediction/test.csv'

testdata= pd.read_csv(filepath1)

testdata2=testdata

traindata= pd.read_csv(filepath)

traindata.head()
#We remove the id column in both the training and testing datasets.

traindata=traindata.drop('Id',axis=1)

testdata=testdata.drop('Id',axis=1)
#working with numeric features (They are all numerical features)

numeric_features = traindata.select_dtypes(include=[np.number])

numeric_features.dtypes
#Checking the correlation between each column with the Cover_Type

corr = numeric_features.corr()

print (corr['Cover_Type'].sort_values(ascending=False)[:5], '\n')

print (corr['Cover_Type'].sort_values(ascending=False)[-5:])
#We see how various features compare with the Cover type



column_names=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']                





for i in column_names:

    feature = traindata.pivot_table(index='Cover_Type',

                  values=i, aggfunc=np.median)

    feature.plot(kind='bar', color='blue')

    plt.xlabel(i)

    plt.ylabel('Median Cover_Type')

    plt.xticks(rotation=0)

    plt.show()

       
#We will define the training and testing data here:



y=traindata['Cover_Type']

x=traindata.drop('Cover_Type',axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
##Now we will run a few machine learning techiniques to see which one is the most applicable



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
# Random Forest

rf = RandomForestClassifier()

rf.fit(x_train,y_train);

y_predicted_r = rf.predict(x_test)

mse = mean_squared_error(y_test, y_predicted_r)

r = r2_score(y_test, y_predicted_r)

mae = mean_absolute_error(y_test,y_predicted_r)

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
#Ridge Regression

ridgereg = Ridge(normalize=True)

ridgereg.fit(x_train, y_train)

y_pred = ridgereg.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test,y_pred)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# LGBMClassifier

lgb_clf = LGBMClassifier(random_state=17)

lgb_clf.fit(x_train, y_train)

y_pred = lgb_clf.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test,y_pred)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
#GridSearchCV

param_grid = {'num_leaves': [7, 15, 31, 63], 

              'max_depth': [3, 4, 5, 6, -1]}

grid_searcher = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, 

                             cv=5, verbose=1, n_jobs=4)

grid_searcher.fit(x_train, y_train)

mse = mean_squared_error(y_test, y_pred)

r = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test,y_pred)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)
# Random Forest

rf = RandomForestClassifier()

rf.fit(x,y);

Prediction = rf.predict(testdata)
predictionlist=Prediction.tolist()

Passengerid=testdata2['Id'].tolist() 

output=pd.DataFrame(list(zip(Passengerid, predictionlist)),

              columns=['Id','Cover_type'])

output.head()

output.to_csv('my_submission(ForestCoverTypePrediction).csv', index=False)