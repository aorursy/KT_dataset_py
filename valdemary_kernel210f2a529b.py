# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
from keras import losses, models, optimizers
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Activation, Flatten) 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
import os # reading the input files we have access to

# print(os.listdir('../input'))
df_train =  pd.read_csv('../input/train.csv', nrows = 100_000, parse_dates=["pickup_datetime"])
df_test =  pd.read_csv('../input/test.csv')

#Dropping data with fare_amount<0
print('Dropping data with fare_amount<0')
print('Old size: %d' % len(df_train))
df_train = df_train[df_train.fare_amount>=0]
print('New size: %d \n' % len(df_train))

#Dropping missing data
print('Dropping missing data')
print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d\n' % len(df_train))

#Adding new columns to dataframe to use them during regression
def add_columns_to_df(df):
    df['distance'] = np.sqrt((df.dropoff_longitude - df.pickup_longitude)**2 + (df.dropoff_latitude - df.pickup_latitude)**2)
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['time'] = df['pickup_datetime'].dt.time
    df['time'] = pd.to_timedelta(df['time'].astype(str))
    df['time'] = df['time'].dt.total_seconds()
add_columns_to_df(df_train)

#Dropping exceedingly big distance
print('Dropping exceedingly big distance')
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.distance < 1.5)]
print('New size: %d\n' % len(df_train))
df_train.head()
#df_train.describe()
#df_train.dtypes

#plotting correlation between data
fig1=plt.figure()
dis=fig1.add_subplot(1,1,1)
dis.scatter(df_train.distance, df_train.fare_amount, color='b')
dis.set_title('Distance - Fare')
dis.set_xlabel('Distance')
dis.set_ylabel('Fare')

fig2=plt.figure()
pas=fig2.add_subplot(1,1,1)
pas.scatter(df_train.passenger_count, df_train.fare_amount, color='r')
pas.set_title('Passenger_Count - Fare')
pas.set_xlabel('Passenger_Count')
pas.set_ylabel('Fare')

fig3=plt.figure()
day=fig3.add_subplot(1,1,1)
day.scatter(df_train.day_of_week, df_train.fare_amount, color='g')
day.set_title('Day_Of_Week - Fare')
day.set_xlabel('Day_Of_Week')
day.set_ylabel('Fare')

fig4=plt.figure()
time=fig4.add_subplot(1,1,1)
time.scatter(df_train.time, df_train.fare_amount, color='y')
time.set_title('Time_in_seconds - Fare')
time.set_xlabel('Time_in_seconds')
time.set_ylabel('Fare')

#Linear Regression:
x = df_train.distance
y = df_train.fare_amount
x = np.transpose(np.atleast_2d(x))
lr = LinearRegression()
lr.fit(x,y)
y_pred = lr.predict(x)
mse_lin = mean_squared_error(y, y_pred)
r2_lin = r2_score(y, y_pred) 
print('Train Fare:',y.tolist()[:10])
print('Linear Regression Predicted Fare: ',y_pred[:10])
print('Linear Regression Mean Square Error: ',mse_lin)
print('Linear Regression R2 Score: ',r2_lin)
print('\n')

x_mul = df_train[['time','distance']]
y_mul = df_train.fare_amount

#Multi Linear Regression
lr_mul = LinearRegression()
lr_mul.fit(x_mul,y_mul)
y_mul_pred = lr_mul.predict(x_mul)
mse_lin_mul = mean_squared_error(y_mul, y_mul_pred)
r2_lin_mul = r2_score(y_mul, y_mul_pred)
print('Train Fare:',y_mul.tolist()[:10])
print('Multi Linear Regression Predicted Fare: ',y_mul_pred[:10])
print('Multi Linear Mean square error: ',mse_lin_mul)
print('Multi Linear R2_score: ',r2_lin_mul)
fig=plt.figure()
mulreg=fig.add_subplot(1,1,1)
mulreg.scatter(y_mul_pred, y_mul, color='r')
mulreg.set_title('Multi Linear Regression')
print('\n')

#Multi Linear Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(x_mul,y_mul)
y_lasso = lasso.predict(x_mul)
mse_lasso = mean_squared_error(y_mul, y_lasso)
r2_lasso = r2_score(y_mul, y_lasso)
print('Train Fare:',y_mul.tolist()[:10])
print('Multi Linear Lasso Regression Predicted Fare: ',y_lasso[:10])
print('Multi Linear Lasso Mean square error: ',mse_lasso)
print('Multi Linear Lasso R2_score: ',r2_lasso)
fig=plt.figure()
mulreg=fig.add_subplot(1,1,1)
mulreg.scatter(y_lasso, y_mul, color='r')
mulreg.set_title('Multi Linear Lasso Regression')
print('\n')

#MLPRegressor
X_train, X_test, y_train, y_test = train_test_split(x_mul, y, test_size=0.4, random_state=0)
mlpReg = MLPRegressor(hidden_layer_sizes=(600, ), activation='tanh', 
                      solver='adam', alpha=0.0001, batch_size='auto', 
                      learning_rate='adaptive', learning_rate_init=0.001, 
                      power_t=0.5, max_iter=10000, shuffle=True, random_state=None, 
                      tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                      nesterovs_momentum=True, early_stopping=False, 
                      validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
                      epsilon=1e-08)

mlpReg = mlpReg.fit(x_mul, y_mul)
y_mlpReg_pred = mlpReg.predict(x_mul)
mse_mlpReg = mean_squared_error(y_mul, y_mlpReg_pred)
r2_mlp = r2_score(y_mul, y_mlpReg_pred)
print('Train Fare:',y_mul.tolist()[:10])
print('MLPRegressor Predicted Fare: ',y_mlpReg_pred[:10])
print('MLPRegressor Mean square error: ',mse_mlpReg)
print('MLPRegressor R2_score: ',r2_mlp)

#r2_mlp = mlpReg.score(X_test, y_test)

### Deep Neural Net with Keras
#kernel_initializer='lecun_uniform'
#bias_initializer='zeros'
#kernel_regularizer=None
#activation = "tanh"
#nb_epoch = 1000 # Кількість епох навчання
#alpha_zero = 0.001 # Коефіцієнт швидкості навчання
#batch_size = 64

#model = Sequential()
#model.add(Dense(10, input_dim = 2 , activation = activation))
#model.add(Dense(2, activation = activation))
#model.add(Dense(1,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer, activation = activation))
#optimizer = optimizers.Nadam(lr=alpha_zero, beta_1=0.9, beta_2=0.999,epsilon=None, schedule_decay=0.004)
#model.compile(loss = "mean_squared_error", optimizer = optimizer, metrics = ["accuracy"])
#history = model.fit(X_train, y_train, batch_size = batch_size,epochs = nb_epoch, verbose=2, validation_data = (X_test, y_test))
#score = model.evaluate(X_test, y_test,verbose = 0)
#model.summary()
#y_pred = model.predict(X_test)
#r2_dnn = r2_score(y_pred, y_test)
#mse_dnn = score[0]
#print('mse_dnn', mse_dnn)
#print('r2_dnn',r2_dnn)
