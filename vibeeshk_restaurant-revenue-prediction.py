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







data = pd.read_csv( '/kaggle/input/restaurant-revenue-prediction/train.csv.zip')

test_data = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')



data.head()







data.describe()



data.dtypes
data['City Group'].unique()
data['City'].unique()
#Creating a flag for each type of restaurant

data['Type_IL'] = np.where(data['Type'] == 'IL', 1, 0)

data['Type_FC'] = np.where(data['Type'] == 'FC', 1, 0)

data['Type_DT'] = np.where(data['Type'] == 'DT', 1, 0)



#Creating a flag for 'Big Cities'

data['Big_Cities'] = np.where(data['City Group'] == 'Big Cities', 1, 0)



#Converting Open_Date into day count

#Considering the same date the dataset was made available

data['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(data['Open Date'])).dt.days



#Removing unused columns

data = data.drop('Type', axis=1)

data = data.drop('City Group', axis=1)

data = data.drop('City', axis=1)

data = data.drop('Open Date', axis=1)



#Adjusting test data as well

test_data['Type_IL'] = np.where(test_data['Type'] == 'IL', 1, 0)

test_data['Type_FC'] = np.where(test_data['Type'] == 'FC', 1, 0)

test_data['Type_DT'] = np.where(test_data['Type'] == 'DT', 1, 0)

test_data['Big_Cities'] = np.where(test_data['City Group'] == 'Big Cities', 1, 0)

test_data['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(test_data['Open Date'])).dt.days

test_data = test_data.drop('Type', axis=1)

test_data = test_data.drop('City Group', axis=1)

test_data = test_data.drop('City', axis=1)

test_data = test_data.drop('Open Date', axis=1)




from sklearn import model_selection

from sklearn import linear_model





X = data.drop(['Id', 'revenue'], axis=1)

Y = data.revenue







from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge



from sklearn import metrics



def check_rmse(X, Y, alpha):

    RMSE_lasso = []

    RMSE_ridge = []



    for i in alpha:

        lasso = Lasso(alpha=i)

        lasso.fit(X, Y)



        ridge = Ridge(alpha=i)

        ridge.fit(X, Y)



        RMSE_lasso.append(metrics.mean_squared_error(Y, lasso.predict(X)))

        RMSE_ridge.append(metrics.mean_squared_error(Y, ridge.predict(X)))

        

    

    return (RMSE_lasso, RMSE_ridge)







import matplotlib.pyplot as plt



alpha = [i/10 for i in range(25, 100, 10)]

RMSE_lasso, RMSE_ridge = check_rmse(X, Y, alpha)



plt.figure()

plt.plot(alpha, RMSE_lasso, 'o-', color="r", label="RMSE_lasso")

plt.plot(alpha, RMSE_ridge, 'o-', color="b", label="RMSE_ridge")

plt.legend(loc='best')

plt.show()



plt.figure()

plt.plot(alpha, RMSE_lasso, 'o-', color="r", label="RMSE_lasso")

plt.legend(loc='best')

plt.show()
lasso = Lasso(alpha=5.5)

lasso.fit(X, Y)



metrics.mean_squared_error(Y, lasso.predict(X))




model = Lasso(alpha=5.5)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission-lasso-5.5.csv', index=False)

test_predicted.describe()







from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=150)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission-random-forest.csv', index=False)

test_predicted.describe()







model = Ridge(alpha=330)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission-ridge-330.csv', index=False)

test_predicted.describe()







model = Lasso(alpha=200000)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission-lasso-high-alpha.csv', index=False)

test_predicted.describe()



data['Days_Open'].unique()







data['Time_Open'] = round(data['Days_Open'] / 700, 0)

data = data.drop('Days_Open', axis=1)



test_data['Time_Open'] = round(test_data['Days_Open'] / 700, 0)

test_data = test_data.drop('Days_Open', axis=1)







X = data.drop(['Id', 'revenue'], axis=1)

Y = data.revenue



model = Ridge(alpha=330)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission.csv', index=False)

test_predicted




model = Lasso(alpha=200000)

model.fit(X, Y)



test_predicted = pd.DataFrame()

test_predicted['Id'] = test_data.Id

test_predicted['Prediction'] = model.predict(test_data.drop('Id', axis=1))

test_predicted.to_csv('submission-lasso-.csv', index=False)

test_predicted.describe()


