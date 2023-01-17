import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pylab as plt

%matplotlib inline

import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

df_train=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')
# Convert Categorical variables: Province_State & Country_Region, into integers for training the model.
display(df_train.head())

display(df_train.describe())

display(df_train.info())
top_cases=df_train.groupby('Country_Region')['ConfirmedCases'].max().sort_values(ascending=False).to_frame()

top_cases=top_cases

top_cases.style.background_gradient(cmap='Reds')
data=df_train.groupby(["Date"])['ConfirmedCases'].sum().to_frame()

data=data.reset_index()
data
fix=px.bar(data,x="Date",y='ConfirmedCases',color="ConfirmedCases")

fix.show()
fig=py.iplot([go.Scatter(

    x=data['Date'],

    y=data['ConfirmedCases'])])           
train_data = pd.get_dummies(df_train, columns=['Country_Region', 'Province_State'], dummy_na=True)
X = train_data.drop(['Id', 'ConfirmedCases', 'Fatalities'], axis = 1)

Y = train_data[['Fatalities', 'ConfirmedCases']]
### Preprocessing 
def preprocessor(data):

    ids = data['ForecastId']

    frame = pd.get_dummies(data, columns = ['Country_Region', 'Province_State'], dummy_na = True).drop(['ForecastId'], axis = 1)

    frame['Date'] = pd.to_datetime(frame['Date']).astype(int)/ 10**9

    return (ids, frame)

#Model
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error

train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size = .2)
print(train_x.shape)

print(train_y.shape)
# model = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# model.fit(train_x, train_y)

import pickle

filename = '/kaggle/input/corona-random-forest-model/random_forest_model.sav'

model = pickle.load(open(filename, 'rb'))
predicted = model.predict(valid_x)

predicted = predicted.round()

rmse = np.sqrt(mean_squared_error(predicted, valid_y))

mae = mean_absolute_error(predicted, valid_y)

print(rmse, mae)
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

output = pd.DataFrame(columns = submission.columns)

output['ForecastId'] = ids

predicted = model.predict(df_test)

predicted = predicted.round()

output[['ConfirmedCases', 'Fatalities']] = predicted

print(output)
# import pickle

# filename = 'random_forest_model.sav'

# pickle.dump(model, open(filename, 'wb'))

 

# some time later...

 

# load the model from disk

# loaded_model = pickle.load(open(filename, 'rb'))

# result = loaded_model.score(X_test, Y_test)

# print(result)
# import sklearn

# help(sklearn.metrics)



output.to_csv('submission.csv', index = False)