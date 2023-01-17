import numpy as np

import pandas as pd

import tensorflow as tf
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train_data.iloc[:, 1].values

X = train_data.iloc[:, [2,4,6,7,9]].values

X_test = test_data.iloc[:, [1,3,5,6,8]].values
print(X)

print(X_test)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer



labelencoder_X_1 = LabelEncoder()



X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])



columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')



X = np.array(columnTransformer.fit_transform(X), dtype = np.str)



X = X[:, 1:]



X_test[:, 1] = labelencoder_X_1.fit_transform(X_test[:, 1])



columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')



X_test = np.array(columnTransformer.fit_transform(X_test), dtype = np.str)



X_test = X_test[:, 1:]
print(X[1])

print(X_test[1])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X)

X_test = sc.transform(X_test)
print(X_train[1])

print(X_test[1])

print(y[1])
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y, batch_size = 32, epochs = 100)
predictions = ann.predict(X_test)
predictions = np.round(predictions)

predictions = predictions.astype(int)

print(predictions)
df = pd.DataFrame(predictions)

df_new = df.rename(columns={0 : 'Survived'})
df_new.head()
df_new = df_new.applymap(str)

df_new.dtypes

df_new.head()
df_new.dtypes
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': df_new.Survived})
print(output)
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")