# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
train_data = pd.read_csv("/kaggle/input/predict-the-diabetes/train.csv")

test_data = pd.read_csv("/kaggle/input/predict-the-diabetes/test.csv")
train_data.head()
y = train_data.iloc[:, -2].values
y[2]
X = train_data.iloc[:, 0:8].values
X[2]
test_data.head()
X_test = test_data.iloc[:, 0:8].values
X_test[0]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X)

X_test = sc.transform(X_test)
X_train[2]
X_test[0]
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))

ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

ann.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y, batch_size = 32, epochs = 100)
predictions = ann.predict(X_test)
predictions
predictions = np.round(predictions)

predictions = predictions.astype(int)

print(predictions)
df = pd.DataFrame(predictions)

df_new = df.rename(columns={0 : 'Outcome'})
df_new.head()
df_new = df_new.applymap(str)

df_new.dtypes

df_new.head()
output = pd.DataFrame({'Id': test_data.Id, 'Outcome': df_new.Outcome})
print(output)
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")