import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense



df = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')



# converting into useful numbers



df['banking_crisis'] = df['banking_crisis'].replace('crisis',np.nan)

df['banking_crisis'] = df['banking_crisis'].fillna(1)

df['banking_crisis'] = df['banking_crisis'].replace('no_crisis',np.nan)

df['banking_crisis'] = df['banking_crisis'].fillna(0)



# removing unneccesary data



df.drop(['cc3','country'], axis=1, inplace=True)



# scaling the data



df_scaled = preprocessing.scale(df)

df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled['banking_crisis'] = df['banking_crisis']

df = df_scaled



# defining the input data, X, and the desired results, y 



X = df.loc[:,df.columns != 'banking_crisis']

y = df.loc[:, 'banking_crisis']



# breaking data into training data, validation data, and test data



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2)



# constructing a simple Multilayer Perceptron



model = Sequential()

model.add(Dense(32,activation = 'relu', input_dim = 11))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# training the network



model.fit(X_train, y_train, epochs=200)



# scoring it on the data it trained on as well as test data



scores = model.evaluate(X_train, y_train)

print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))



scores = model.evaluate(X_test, y_test)

print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))



# plotting the confusion matrix



y_test_pred = model.predict_classes(X_test)

c_matrix = confusion_matrix(y_test,y_test_pred)

ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Banking Crisis','Banking Crisis'], yticklabels=['No Banking Crisis','Banking Crisis'], cbar=False, cmap='Blues')

ax.set_xlabel("Prediction")

ax.set_ylabel("Actual")