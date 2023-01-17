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
%matplotlib inline

import matplotlib.pyplot as plt
df = pd.read_csv("../input/weight-height/weight-height.csv")
df.head()
X = df[['Height']]

y = df[['Weight']]
X.describe()
def line(X, m = 0, c = 0):  # 'm' can be 'w' and 'b' can be 'c'

    return ((m * X) + c) # y = mx + c





# Mean Squared Error

def mse(y_actual, y_pred):

    s = (y_actual - y_pred) ** 2

    return np.sqrt(s.mean())
plt.figure(figsize=(10, 5))

axes_1 = plt.subplot(121)

df.plot(

    kind = 'scatter',

    x = 'Height',

    y = 'Weight',

    title = 'Weight and Height in adults',

    ax = axes_1

)



bs = np.array([-100, -50, 0, 50, 100, 150])

mses =  []



for b in bs:

    y_pred = line(X, m = 2, c = b)

    mserror = mse(y, y_pred)

    mses.append(mserror)

    plt.plot(X, y_pred)

    

axes_2 = plt.subplot(122)

plt.plot(bs, mses, 'o-')

plt.title('Cost as a Function of b')

plt.xlabel('b')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam, SGD
X.shape
model = Sequential()

model.add(Dense(1, input_shape=(1, )))
model.summary()
model.compile(Adam(lr=0.8), 'mean_squared_error')
model.fit(X, y, epochs = 40)
y_pred = model.predict(X)
y_pred
df.plot(

    kind = 'scatter',

    x = 'Height',

    y = 'Weight',

    title = 'Weight and Height in Adults'

)

plt.plot(X, y_pred, color = 'red', linewidth = 3)
w, b = model.get_weights()
print("Weight: ", w)
model.predict([[200]])