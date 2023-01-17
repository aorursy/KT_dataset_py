# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data = pd.read_csv("/kaggle/input/ciri422/training.csv", header=None)

test_data = pd.read_csv("/kaggle/input/ciri422/test.csv", header=None)

output = pd.read_csv("/kaggle/input/ciri422/sample.csv")



Input = raw_data.index

Columns = raw_data.columns

output.columns = ["Id", "Redshift"]

raw_data.columns = ["ID", "U", "G", "R", "I", "Z", "Y", "Red-shift"]

test_data.columns = ["ID", "U", "G", "R", "I", "Z", "Y"]



red_shift = raw_data["Red-shift"]

raw_data = raw_data.drop(columns=["Red-shift", "ID"])

test_data = test_data.drop(columns=["ID"])

print(red_shift)

print(raw_data)
raw_data.describe() # Stat description
import sklearn.linear_model as skl

import keras as ks
def rmse(y_true, y_pred):

	return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true)))
nn_model = ks.Sequential()



nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense01", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense02", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense03", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense04", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense05", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense06", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense07", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense08", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense09", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense10", kernel_regularizer=ks.regularizers.l2(1), activity_regularizer=ks.regularizers.l2(1)))

nn_model.add(ks.layers.Dense(1, activation=None, name="output"))

opt = ks.optimizers.Adam(lr=0.001, decay=5e-4)

nn_model.compile(optimizer=opt, loss=rmse, metrics=[rmse]) #mean_squared_logarithmic_error

X_train = np.asarray(raw_data)

Y_train = np.asarray(red_shift)

results = nn_model.fit(X_train, Y_train, batch_size=32, epochs=10000)
nn_model.summary()
pd.DataFrame(results.history).plot(figsize=(8, 5))

plt.grid(True)

plt.yscale("log")



plt.show()
print(nn_model.evaluate(X_train, Y_train))

train_prediction = nn_model.predict(np.asarray(X_train))



plt.gcf()

plt.scatter(train_prediction, Y_train, linewidths=0.01)

plt.show()

predictions = nn_model.predict(np.asarray(test_data))

print(predictions)
output["Redshift"] = predictions

formated_output = np.empty_like(output, dtype='O')



for index, row in output.iterrows():

    formated_output[index] = ("{:.18e}".format(row['Id']), "{:.18e}".format(row['Redshift']))



formated_output = pd.DataFrame(formated_output)

formated_output.columns = ["Id", "Redshift"]

display(formated_output)

formated_output.to_csv('\kaggle\output\predictions.csv', index=False)