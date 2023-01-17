# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date, datetime

from scipy import stats

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

import plotly.graph_objs as go

import plotly.express as px



from sklearn.metrics import mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/nba2k20-player-dataset/nba2k20-full.csv')

data.head(5)
data.info()
def prepare_data(data: pd.DataFrame):

    '''

        Preprocesses data

    '''

    def calculateAge(birthDate: str):

        '''

        calculates age of person, on given birth day

        '''

        datetime_object = datetime.strptime(birthDate, '%m/%d/%y')

        today = date.today() 

        age = today.year - datetime_object.year -  ((today.month, today.day) < (datetime_object.month, datetime_object.day)) 

        return age 

    

    data['jersey'] = data['jersey'].apply(lambda x: int(x[1:]))

    data['age'] = data['b_day'].apply(calculateAge)

    data['height'] = data['height'].apply(lambda x: float(x.split('/')[1]))

    data['weight'] = data['weight'].apply(lambda x: float(x.split('/')[1].split(' ')[1]))

    data['salary'] = data['salary'].apply(lambda x: float(x[1:]))

    data['draft_round'].replace('Undrafted', 0, inplace = True)

    data['draft_round'] = data['draft_round'].apply(int)

    data['team'] = data['team'].fillna('No team')

    data['college'] = data['college'].fillna('No education')

    data.drop(['b_day', 'draft_peak'], axis = 1, inplace = True)


data = pd.read_csv('../input/nba2k20-player-dataset/nba2k20-full.csv')

prepare_data(data)



#creating categories to teams by mean salary

salary = data[['salary', 'team']]

new_sal = salary.groupby('team').mean().reset_index()

boundaries = [np.NINF, 7E+6, 7.6E+6, 8.1E+6, 9E+6, 9.5E+6, np.Inf]

new_sal['team_salary'] = pd.cut(salary.groupby('team').mean().\

                                reset_index()['salary'], bins=boundaries)

new_sal.drop(['salary'], axis = 1, inplace = True)

#merging this categories to data

data = data.merge(new_sal, on = 'team', how = 'left')



#removing imbalanced data

data.loc[data['country'] != 'USA', 'country'] = 'not USA'

data.loc[data['position'] == 'C-F', 'position'] = 'F-C'

data.loc[data['position'] == 'F-G', 'position'] = 'F'

data.loc[data['position'] == 'G-F', 'position'] = 'F'



# we should drop full_name because it doesn't have anything meaning for this type of model

# we should drop jersey because it doesn't have high correlation

# we should drop team because we have already preprocessed it

# For now we should drop college because there is too much colleges with just 5 or less occurances

data = data.drop(['full_name', 'jersey',  'team', 'college'], axis = 1)



# converting categorical data to one-hot encoding

data = pd.get_dummies(data, 

                      columns = ['team_salary', 'position', 'country', 'draft_round'],

                      drop_first = True)



X, y = data.drop(['salary'], axis = 1), data['salary']

#normalizing input features

normalizer = preprocessing.Normalizer().fit(X)

X = normalizer.transform(X)

#Split data into random train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBRegressor



model = XGBRegressor( 

    n_estimators = 300,

    learning_rate=0.06,

    colsample_bytree=0.9, 

    min_child_weight=3,

    max_depth = 2,

    subsample = 0.63,

    eta = 0.1,

    seed=0)





model = model.fit(

    X_train, 

    y_train, 

    eval_metric="rmse", 

    early_stopping_rounds=20,

    eval_set=[(X_test,y_test)],

    verbose=False)



predictions = model.predict(X_test)







print("RMSE: ", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
x_ax = list(range(len(y_test)))

fig = go.Figure([go.Scatter(x=x_ax, y=y_test, name='original'), go.Scatter(x=x_ax, y=predictions, name='predicted')])

fig.show()
def plot_features(booster):    

    importance = pd.DataFrame({'importance': model.feature_importances_, \

                               'name' : data.drop('salary', axis=1).columns})



    fig = px.bar(importance.sort_values(by='importance', ascending=True), 

                 x = 'importance', y = 'name')

    fig.show()

  



plot_features(model)