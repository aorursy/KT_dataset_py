import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense



sns.set_style("darkgrid")
df = pd.read_csv("/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv")



df = df.drop(columns=["Unnamed: 0"])

newColumnsNames = list(map(lambda c: c.split(" - ")[0] if "-" in c else "DATE", df.columns))

newColumnsNames

df.columns = newColumnsNames
# Fill ND values with previous and next values



df = df.replace("ND", np.nan)

df = df.bfill().ffill() 



# Make date wise indexing 



df = df.set_index("DATE")

df.index = pd.to_datetime(df.index)

df = df.astype(float)
print("Total number of records", len(df))

print("Total number of days between {} and {} are {}".format(df.index.min().date(), df.index.max().date(), (df.index.max() - df.index.min()).days+1))
# Prepare a full dataframe

num_records = 7303

data = {}

data["DATE"] = pd.date_range("2000-01-03", "2019-12-31", freq="D")



complete = pd.DataFrame(data=data)

complete = complete.set_index("DATE")

complete = complete.merge(df, left_index=True, right_index=True, how="left")

complete = complete.bfill().ffill()
complete.head()
toInspect = ["INDIA", "CHINA", "EURO AREA"]

rows, cols = 3, 2

fig, ax = plt.subplots(rows, cols, figsize=(20,rows*5))



for row in range(rows):

    sns.lineplot(data=df[[toInspect[row]]], ax=ax[row][0])

    sns.lineplot(data=complete[[toInspect[row]]], ax=ax[row][1])
sampled2d = complete.resample("2D").mean()
# Data Conversion Utility



def getTimeSeriesData(A, window=7):

    X, y = list(), list()

    for i in range(len(A)):

        end_ix = i + window

        if end_ix > len(A) - 1:

            break

        seq_x, seq_y = A[i:end_ix], A[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
window = 2

num_features = 1



X, y = getTimeSeriesData(list(sampled2d["INDIA"]), window=window)

print("X:", X.shape)

print("Y:", y.shape)



# We need to add one more dimension to X, i.e Num of features in 1 sample of time step. as we are doing a univariate prediction which means number of features are 1 only

X = X.reshape((X.shape[0], X.shape[1], num_features))  # For LSTM

print("-----------")

print("X:", X.shape)

print("Y:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("-----------")

print("X train:", X_train.shape)

print("y train:", y_train.shape)

print("X test:", X_test.shape)

print("y test:", y_test.shape)
# Define Model

model = Sequential()

model.add(LSTM(7, activation='relu', input_shape=(window, num_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=5, verbose=1)
plt.plot(history.history["loss"])

# plt.plot(history.history["val_loss"])

plt.title("Model Loss")

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')

plt.show()
yPred = model.predict(X_test, verbose=0)

yPred.shape = yPred.shape[0]
plt.figure(figsize=(30,5))

sns.set(rc={"lines.linewidth": 8})

sns.lineplot(x=np.arange(y_test.shape[0]), y=y_test, color="green")

sns.set(rc={"lines.linewidth": 3})

sns.lineplot(x=np.arange(y_test.shape[0]), y=yPred, color="coral")

plt.margins(x=0, y=0.5)

plt.legend(["Original", "Predicted"])
points = 200

plt.figure(figsize=(30,5))

sns.set(rc={"lines.linewidth": 8})

sns.lineplot(x=np.arange(points), y=y_test[:points], color="green")

sns.set(rc={"lines.linewidth": 3})

sns.lineplot(x=np.arange(points), y=yPred[:points], color="coral")

plt.margins(x=0, y=0.5)

plt.legend(["Original", "Predicted"])