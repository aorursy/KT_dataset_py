import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures



import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/abafall2020/train.csv")

df.head()
df_test = pd.read_csv("../input/abafall2020/test.csv")

df_test.head()
sns.distplot(df['Price'])
sns.distplot(df['Carat Weight'])
polish_map = {'G': 1, 

              'VG': 2, 

              'EX': 3, 

              'ID': 4}
clarity_map = {'SI1': 1, 

               'VS2': 2, 

               'VS1': 3,

               'VVS2': 4,

               'VVS1': 5,

               'IF': 6,

               'FL': 7}
symmetry_map = {'G': 1,

                'VG': 2,

                'EX': 3,

                'ID': 4}
color_map = {'J': 1,

                'I': 2,

                'H': 3,

                'G': 4,

                'F': 5,

                'E': 6,

                'D': 7}
cut_map = {'Poor': 1, 

               'Fair': 2, 

               'Good': 3,

               'Very Good': 4,

               'Ideal': 5,

               'Signature-Ideal': 6}
df['Cut'] = df['Cut'].map(cut_map)

df['Color'] = df['Color'].map(color_map)

df['Polish'] = df['Polish'].map(polish_map)

df['Symmetry'] = df['Symmetry'].map(symmetry_map)

df['Clarity'] = df['Clarity'].map(clarity_map)
df_test['Cut'] = df_test['Cut'].map(cut_map)

df_test['Color'] = df_test['Color'].map(color_map)

df_test['Polish'] = df_test['Polish'].map(polish_map)

df_test['Symmetry'] = df_test['Symmetry'].map(symmetry_map)

df_test['Clarity'] = df_test['Clarity'].map(clarity_map)
df = pd.get_dummies(df, drop_first=True)

df_test = pd.get_dummies(df_test, drop_first=True)



df.head()
df['Price'] = np.log(df['Price'])
df_X = df.drop(['Price', 'ID'], axis=1)

df_y = df['Price']
X_train, X_valid, y_train, y_valid = train_test_split(df_X, df_y, test_size=0.2)

X_test = df_test.drop(['Price', 'ID'], axis=1)
lm = LinearRegression()

lm.fit(X_train, y_train)



predictions = lm.predict(X_valid)
plt.scatter(y_valid,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
mean_squared_error(y_valid, predictions)
np.sqrt(mean_squared_error(y_valid, predictions) / 1200)
# MAPE



np.mean(np.abs((y_valid - predictions) / y_valid)) * 100
X_valid.shape
X_test.shape
poly_features = PolynomialFeatures(degree=3, include_bias=False)



X_train_poly = poly_features.fit_transform(X_train)

X_valid_poly = poly_features.transform(X_valid)

X_test_poly = poly_features.transform(X_test)
lm_poly = LinearRegression()
lm_poly.fit(X_train_poly, y_train)



predictions_poly = lm_poly.predict(X_valid_poly)
plt.scatter(y_valid,predictions_poly)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# MAPE



np.mean(np.abs((y_valid - predictions_poly) / y_valid)) * 100
import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train_nn = scaler.fit_transform(X_train)

X_valid_nn = scaler.transform(X_valid)

X_test_nn = scaler.transform(X_test)
model = keras.models.Sequential([

    keras.layers.Dense(28, activation = 'relu'),

    keras.layers.Dense(28, activation = 'relu'),

    keras.layers.Dense(14, activation = 'relu'),

    keras.layers.Dense(7, activation = 'relu'),

    keras.layers.Dense(1)

])



model.compile(loss="mean_squared_error", optimizer="adam")



early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train_nn, y_train, epochs=200, validation_data=(X_valid_nn, y_valid), callbacks=[early_stopping_cb])
predictions_nn = model.predict(X_valid_nn)



plt.scatter(y_valid,predictions_nn)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
mean_squared_error(y_valid, predictions_nn)
np.sqrt(mean_squared_error(y_valid, predictions_nn) / 1200)
predictions_nn.ravel().shape
# MAPE



np.mean(np.abs((y_valid - predictions_nn.ravel()) / y_valid)) * 100
test_predictions = np.exp(lm.predict(X_test))

test_predictions_nn = np.exp(model.predict(X_test))



test_predictions_poly = np.exp(lm_poly.predict(X_test_poly))



ensemble_predictions = (test_predictions_nn + test_predictions_poly) / 2



preds = {'lm':test_predictions, 'nn':test_predictions_nn, 'poly_lm': test_predictions_poly, 'ensemble': ensemble_predictions}
result_set = 'poly_lm'

df_test['Price'] = preds[result_set]

df_test[['ID', 'Price']]
df_test[['ID', 'Price']].to_csv('out.csv', index=False)