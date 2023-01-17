import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression 



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



train_data.dropna(inplace=True)

test_data.dropna(inplace=True)



train_data.describe()
x_train = train_data.x

y_train = train_data.y

x_test = test_data.x

y_test = test_data.y
np_x_train = np.array(x_train)

np_y_train = np.array(y_train)

np_x_test = np.array(x_test)

np_y_test = np.array(y_test)
n = len(np_x_train)

alpha = 0.000001



a = 0

b = 0



plt.scatter(np_x_test,np_y_test,color='red',label='GT')



for epoch in range(1000):

    y = a * np_x_train + b

    error = y - np_y_train

    mean_sq_er = np.sum(error**2)/n

    b = b - alpha * 2 * np.sum(error)/n 

    a = a - alpha * 2 * np.sum(error * np_x_train)/n

    if(epoch%10 == 0):

        print(mean_sq_er)
np_y_prediction = a * np_x_test + b

print('R2 Score:',r2_score(np_y_test,np_y_prediction))

plt.xkcd()

np_y_plot = []

for i in range(100):

    np_y_plot.append(a * i + b)

plt.figure(figsize=(10,10))

plt.scatter(np_x_test,np_y_test,color='red',label='GT')

plt.plot(range(len(np_y_plot)),np_y_plot,color='black',label = 'prediction')

plt.legend()

plt.show()
len(y_train)
new_x_train = np_x_train.reshape(-1,1)

new_x_test = np_x_test.reshape(-1,1)



regressor = LinearRegression()

regressor.fit(new_x_train,y_train)

y_pred = regressor.predict(new_x_test)

print(r2_score(y_test,y_pred))