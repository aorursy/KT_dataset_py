# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import keras as ks

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
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
def polyfeatures(X_train):

    poly = PolynomialFeatures(2)

    X_train = poly.fit_transform(X_train)

    X_train = X_train[:, 1:]

    return X_train
def rmse(y_true, y_pred):

	return ks.backend.sqrt(ks.backend.mean(ks.backend.square(y_pred - y_true)))
nn_model = ks.Sequential()



nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense01"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense02"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense03"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense04"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense05"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense06"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense07"))

nn_model.add(ks.layers.Dense(100, activation="selu", name="Dense08"))

nn_model.add(ks.layers.Dense(1, activation=None, name="output"))

opt = ks.optimizers.Adam(lr=0.001, decay=5e-4)

nn_model.compile(optimizer=opt, loss=rmse, metrics=[rmse])

Y_train = red_shift.to_numpy()

X_train = raw_data.to_numpy()

X_train = polyfeatures(X_train)

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
X_test = test_data.to_numpy()

X_test = polyfeatures(X_test)

predictions = nn_model.predict(X_test)



output["Redshift"] = predictions

formated_output = np.empty_like(output, dtype='O')



for index, row in output.iterrows():

    formated_output[index] = ("{:.18e}".format(row['Id']), "{:.18e}".format(row['Redshift']))



formated_output = pd.DataFrame(formated_output)

formated_output.columns = ["Id", "Redshift"]

display(formated_output)

formated_output.to_csv('\kaggle\output\predictions5.csv', index=False)