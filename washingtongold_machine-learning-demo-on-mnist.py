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
train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
train.head()
# To get the dimension of the data

train.shape
# To get a column

train['label']
# To get multiple columns

# To get a column

train[['label','1x1','2x2']]
# To select a row

train.loc[0]
# To select several rows

train.loc[[0,1,2]]
# To get all the columns except one, use .drop. SPECIFY THE AXIS OR IT WILL DROP A ROW!

X = train.drop('label',axis=1)
y = train['label']
X
first_image = X.loc[0]

first_image
first_image = np.array(first_image)

first_image
first_image = first_image.reshape((28,28))

first_image
import matplotlib.pyplot as plt

plt.imshow(first_image)
import matplotlib.pyplot as plt

plt.imshow(first_image,cmap='gist_gray')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier()

dec.fit(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier

ran = RandomForestClassifier()

ran.fit(X_train,y_train)
#Prediction labels

log.predict(X_test)
from sklearn.metrics import mean_absolute_error as mae

mae(log.predict(X_test),y_test)
X_test
X_test.loc[1]
X_test = X_test.reset_index() #how can we address the issue of the old index? we don't need it anymore.

X_test
X_test = X_test.drop('index',axis=1)

X_test
image = np.array(X_test.loc[1])
display_image = image.reshape((28,28))
print("Label:",ran.predict(image.reshape(1, -1)))

plt.imshow(display_image)
index = 6



image = np.array(X_test.reset_index().drop('index',axis=1).loc[index])

display_image = np.array(X_test.reset_index().drop('index',axis=1).loc[index]).reshape((28,28))

print("Label:",ran.predict(image.reshape(1,-1)))

plt.imshow(display_image)

plt.show()
print("Label:",log.predict(image.reshape(1,-1)))

plt.imshow(display_image)