import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from subprocess import check_output
from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/crowdedness-at-the-campus-gym/data.csv')
df.head()
df.describe()
correlation = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
sns.distplot(df['temperature'], kde=True, rug=True)
sns.distplot(df['number_people'], kde=True, rug=True)
def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second
df = df.drop("date", axis=1)
noon = time_to_seconds(time(12, 0, 0))
df.timestamp = df.timestamp.apply(lambda t: abs(noon - t))
# one hot encoding
columns = ["day_of_week", "month", "hour"]
df = pd.get_dummies(df, columns=columns)
df.head()
data = df.values
X = data[:, 1:]  # all rows, no label
y = data[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler1 = StandardScaler()
scaler1.fit(X_train[:, 0:1])
X_train[:, 0:1] = scaler1.transform(X_train[:, 0:1])
X_test[:, 0:1] = scaler1.transform(X_test[:, 0:1])
scaler2 = StandardScaler()
scaler2.fit(X_train[:, 3:4])
X_train[:, 3:4] = scaler2.transform(X_train[:, 3:4])
X_test[:, 3:4] = scaler2.transform(X_test[:, 3:4])
print(X_train)
model = RandomForestRegressor(n_jobs=-1)
estimators = np.arange(40, 160, 20)
scores = []
for n in estimators:
    model.set_params(n_estimators=n)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
    print('score = ', model.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
import sklearn.metrics as metrics
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true + 1 - y_pred) / (y_true + 1)) * 100)
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('R^2: ', r2)
    print('MAE: ', mean_absolute_error)
    print('MSE: ', mse)
    print('RMSE: ', np.sqrt(mse))
    print('MAPE: ', mean_absolute_percentage_error(y_true, y_pred), '%')
regression_results(y_test, model.predict(X_test))
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
print(X_train.shape)
print(y_train.shape)
def create_network():
    model = Sequential()
    model.add(Dense(64, input_shape=(49,), activation='relu')) 
    model.add(Dense(1, activation='relu')) 
    #model.add(Dense(128, activation='relu'))
        
    model.compile(loss='mse', optimizer=Adam(), metrics=['accuracy']) 
    return model
earlyStopping = EarlyStopping(patience=30, verbose=1)
mcp_save = ModelCheckpoint('gym_model.h5', verbose=0, save_best_only=True, save_weights_only=True)
reduce_lr_loss = ReduceLROnPlateau(factor=0.1, patience=15, min_lr=0.000001, verbose=0)
callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
model = create_network()
results = model.fit(X_train, y_train, epochs = 500, batch_size = 64, 
                    validation_data = (X_test, y_test), verbose = 0)
model.evaluate(X_test, y_test)