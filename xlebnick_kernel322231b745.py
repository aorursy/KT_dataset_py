import pandas as pd

import numpy as np

import matplotlib.pyplot as pl

import datetime

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

# from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
FILENAME_IN = "../input/robot-shop-competition/train_sales.csv"

# this function reads the pandas data frame, parse_dates indicates that the column 'date'

#   is of type Datetime. dtype parameter specifies that the sales should ba a float

sales_frame = pd.read_csv(FILENAME_IN, parse_dates = ['date'], dtype = {"sales": np.float64})

sales_frame
# this function adds autoregressors to the frame, as was shown in the slides

# Argument: data_frame, where each row of the data_frame is a training point, which consists

#           of the label under column "sales", and the date, which can be used in some way later

# Returns: a new data frame, where each row has a training point. The labels are in the column "sales",

#          while regressors/features are now placed in columns 'ar1', 'ar2', etc. - depends on the 

#          argument n_autoregressors. These autoregressors are simply the preceding values of the time series

#          for each data (sliding window)

def add_autoregressors(sales_frame, n_autoregressors):

    sales_frame_new = sales_frame.copy()

    for i in range(1, n_autoregressors + 1):

        sales_frame_new = sales_frame_new.iloc[1:sales_frame_new.shape[0], :].reset_index(drop = True)

        new_column_name = "ar" + str(i)

        sales_frame_new[new_column_name] = sales_frame.loc[0:(sales_frame.shape[0] - i), "sales"]

    

    return sales_frame_new
data_frame = add_autoregressors(sales_frame, 3)



data_frame = data_frame.reset_index()

data_frame['weekday'] = data_frame.date.apply(lambda x: x.dayofweek)

data_frame['weekend'] = (data_frame['weekday'] > 4).astype(int) 

data_frame
# remove date column, convert the feature and label matrices into numpy for scikit-learn

def get_XY(data_frame, x_name_list, y_name_list):

    return (data_frame[x_name_list].values, data_frame[y_name_list].values)



X, y = get_XY(data_frame, data_frame.columns.difference(['date', 'sales']), ['sales'])
X
# Fit a scikit-learn model. Here it is a simple regression. 

# Be careful about the intercept

# You can find the documentation and examples here: 

#     https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

# scikit-learn library fits all models in a similar way: 

# 1) Instantiate the model: can be many different classes

#    You can select a lot of models from here: 

#        https://scikit-learn.org/stable/supervised_learning.html#supervised-learning

#    All of these models have different parameters you can try to improve the accuracy. 

#    The specific parameters vary from model to model

reg_model = LinearRegression()

# reg_model = MLPRegressor()

# 2) Perform model training/fitting: X is features, y is labels. 

#    This notation is held across the whole scikit-library.



fit = reg_model.fit(X, y)

# 3) Predictions can be usually made by calling the predict() method, which we previously got from fit()

#    It usually requires a new set of features that we supply (for which predictions are to be made)

# scores = fit.predict(X_valid)

# scores



# ids = range(0,scores.__len__())

# pl.plot(ids,scores,color='red')

# pl.plot(ids,y_valid,color='blue')

# pl.show()
### This block demonstrates how dates can be used. 

### In Pandas, datetime functionality can be accessed with .dt suffix on the column name. 

### You can use it for seasonal feature creation! 

def get_season(month_inx):

    if month_inx in [12, 1, 2]:

        return "winter"

    elif month_inx in [3, 4, 5]:

        return "spring"

    elif month_inx in [6, 7, 8]:

        return "summer"

    else:

        return "autumn"

    

sales_frame.date.apply(lambda x: get_season(x.month)).values[0:10]
# This method will plot things for you! 

# Argument 1 - dates: supply the dates - these will be drawn automatically on the x-axis

# Argument 2 - y:     supply the real values of the time series/time series you want to draw. 

#                     This should correspond to dates.

# Argument 3 - predicted: supply the second time series you want to plot. Usually, the predicted values.

#                         This should correspond to dates-parameter too.

def plot_time_series(dates, y, predicted = None):

    pl.rcParams["figure.figsize"] = (100,50)

    pl.rcParams['xtick.labelsize'] = 40 

    pl.rcParams['ytick.labelsize'] = 40 

    pl.xlabel('date', fontsize=40)

    pl.ylabel('sales', fontsize=40)

    

    pl.plot(dates, y)

    if predicted is not None: 

        pl.plot(dates, predicted, color = "red")

    



plot_time_series(sales_frame.date, sales_frame.sales, np.linspace(0, 200, 600))

features = X[-1:,:]





scores = np.zeros(100)

for i in range(0,100):

#     new_value = np.array([reg_model.predict(features)])

    new_value = reg_model.predict(features)

    print(new_value)

    features = np.concatenate([new_value,features[:, :2],

                               [features[:, -3] + 1],

                               [(features[:, -2]+1)%7],

                               [(((features[:, -2]+1)%7) > 4)*1]], axis = 1)

    scores[i] = new_value



    

scores = pd.DataFrame(scores)

scores = scores.reset_index()

scores.columns = ['Id', 'Predicted']

new_value

# scores


import base64

from IPython.display import HTML

def create_download_link(df, filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">Download CSV file</a>'

    html = html.format(payload=payload,title="Download CSV file",filename=filename)

    return HTML(html)



create_download_link(scores)



plot_time_series(np.array(range(0,X.shape[0]+100)), np.concatenate([reg_model.predict(X), scores.Predicted.values.reshape([-1, 1])]))