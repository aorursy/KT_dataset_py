import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as skm

import os



from keras.models import Sequential

from keras.layers import Dense, Dropout, CuDNNLSTM



print(os.listdir("../input"))
def change_season(df, season='D'):

    if season != 'D':

        df["ds"] = pd.to_datetime(df["ds"])



        if season == 'W':

            df = df.resample("W", on="ds").sum()

            df["ds"] = df.index.strftime('%Y-%m-%d')

        elif season == 'M':

            df = df.resample("M", on="ds").sum()

    

        df["ds"] = df.index.strftime('%Y-%m-%d')



    return df



def construct_train_test_sets(dataset, window_size, forecast_horizon, scaler):    

    # Checked

    train = dataset[:-window_size - forecast_horizon]

    test = dataset[-forecast_horizon - window_size:]



    if window_size + forecast_horizon > len(train):

        print("Not enough training data.")

        return

    

    scaler.fit(dataset.reshape(-1, 1))

    X_train = list()

    y_train = list()



    # Checked

    for i in range(window_size, len(train) - forecast_horizon):

        X_train.append(train[i - window_size:i])

        y_train.append(train[i:i + forecast_horizon])



    # Checked

    X_train, y_train = scaler.transform(np.array(X_train).reshape(-1, 1)), scaler.transform(np.array(y_train).reshape(-1, 1))

    X_train, y_train = X_train.reshape(-1, window_size, 1), y_train.reshape(-1, forecast_horizon)



    print("X_train shape:", X_train.shape)

    print("y_train shape:", y_train.shape)



    # Checked

    X_test = scaler.transform(test[:-forecast_horizon].reshape(-1, 1))

    X_test = X_test.reshape(1, WINDOW_SIZE, 1)

    print("X_test shape:", X_test.shape)



    y_test = test[-forecast_horizon:].reshape(-1, 1)



    return X_train, y_train, X_test, y_test



def calc_error_rate(predictions, y_test):

    return ((np.sum(predictions) - np.sum(y_test)) / np.sum(y_test)) * 100



def compile_model(model):

    model.compile(optimizer="rmsprop", loss='mean_squared_error')

    print(model.summary())
WINDOW_SIZE = 180

FORECAST_HORIZON = 90
df = pd.read_csv("../input/120001_PH1.csv", names=['ds', 'km'], skiprows=1)

df[df.km > 1200] = np.nan

df.fillna(method="ffill", inplace=True)



#df = change_season(df, "M")



plt.figure(figsize=(16, 5))

df.plot()

plt.show()
scaler = MinMaxScaler()

X_train, y_train, X_test, y_test = construct_train_test_sets(df.km.values, WINDOW_SIZE, FORECAST_HORIZON, scaler)
std_model = Sequential()



std_model.add(CuDNNLSTM(50, input_shape=(WINDOW_SIZE, 1)))

std_model.add(Dropout(0.2))



std_model.add(Dense(FORECAST_HORIZON))
compile_model(std_model)
history = std_model.fit(X_train, y_train, epochs=400, batch_size=32)
plt.plot(history.history['loss'], label="Train")



plt.xlabel('Epoch')

plt.ylabel('Mean Squared Error Loss')

plt.title('Loss Over Time')

plt.legend()

plt.show()
predictions = std_model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
plt.plot(y_test, label="Actual")

plt.plot(predictions.reshape(-1, 1), label="Prediction")

plt.legend()

plt.show()
rmse = np.sqrt(skm.mean_squared_error(y_test, predictions.reshape(-1, 1)))

print("RMSE:", rmse)



print("Error rate:", calc_error_rate(predictions, y_test))