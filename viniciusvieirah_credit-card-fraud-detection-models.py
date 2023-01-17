import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import numpy 
from collections import Counter
data = pd.read_csv("../input/creditcard.csv")
credit_card = data
data['Class'] = pd.to_numeric(data['Class'])
data.info()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
Xtrain, Xtest, ytrain, ytest = train_test_split(data, data['Class'], test_size=0.30, random_state=42)
print(Xtrain.shape);
print(Xtest.shape);
clf_logistic = LogisticRegression(penalty='l2');
clf_logistic.fit(Xtrain, ytrain);
ypred = clf_logistic.predict(Xtest);
print(metrics.confusion_matrix(ytest,ypred));
print(metrics.classification_report(ytest,ypred));
print('Accuracy : %f' %(metrics.accuracy_score(ytest,ypred)));
print('Area under the curve : %f' %(metrics.roc_auc_score(ytest,ypred)));
from sklearn.metrics import classification_report


print('Total', data['Class'].count())

not_fraud = data[data['Class']==0]
print('not_fraud', not_fraud['Class'].count())

fraud = data[data['Class']==1]
print('Fraud', fraud['Class'].count())
# esse valor ta estranho
print('Proportion:', round(not_fraud['Class'].count() / fraud['Class'].count(), 2), ': 1')
from sklearn.utils import resample
y = data.Class
X = data.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)
X = pd.concat([X_train, y_train], axis=1)


not_fraud = X[X.Class == 0]
print('not_fraud', not_fraud['Class'].count())

fraud = X[X.Class == 1]
print('Fraud', fraud['Class'].count())


fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results
upsampled = pd.concat([not_fraud, fraud_upsampled])

upsampled.Class.value_counts()
y_train = upsampled.Class
X_train = upsampled.drop('Class', axis=1)
clf_logistic = LogisticRegression(penalty='l2');
clf_logistic.fit(X_train, y_train);
ypred = clf_logistic.predict(X_test);
print(metrics.confusion_matrix(y_test,ypred));
print(metrics.confusion_matrix(y_test,ypred));
print(metrics.classification_report(y_test,ypred));
print('Accuracy : %f' %(metrics.accuracy_score(y_test,ypred)));
print('Area under the curve : %f' %(metrics.roc_auc_score(y_test,ypred)));
import os
import numpy
import pandas
import matplotlib.pyplot as plt

from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


data = data.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

X_train = X_train.values
X_test = X_test.values
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))

# encoder
encoder = Dense(encoding_dim, activation='tanh')(input_layer)
encoder = Dense(int(encoding_dim / 2), activation='relu')(encoder)

# decoder
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)

# modelo
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = autoencoder.fit(X_train,
                          X_train,
                          epochs=5,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1).history
predictions = autoencoder.predict(X_test)

mse = numpy.mean(numpy.power(X_test - predictions, 2), axis=1)

error = pandas.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
error.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
normal = error[(error['true_class'] == 0) & (error['reconstruction_error'] < 10)]
_ = ax.hist(normal.reconstruction_error.values, bins=10)
fig = plt.figure()
ax = fig.add_subplot(111)
fraud = error[(error['true_class'] == 1)]
_ = ax.hist(fraud.reconstruction_error.values, bins=10)
threshold = 3.0
groups = error.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    label = 'Fraud' if name == 1 else 'Normal'
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='', label=label)
    
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, label='Threshold')
ax.legend()
plt.title('Erro de Reconstrução Para Diferentes Classes')
plt.ylabel('Erro de Reconstrução')
plt.xlabel('Index')
plt.show()
y_pred = [1 if e > threshold else 0 for e in error.reconstruction_error.values]
confusion_matrix(error.true_class, y_pred)
# ypred = clf_logistic.predict(Xtest);
print(metrics.confusion_matrix(error.true_class,y_pred));
print(metrics.classification_report(error.true_class,y_pred));
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim, ))

# encoder
encoder_binary = Dense(encoding_dim, activation='tanh')(input_layer)
encoder_binary = Dense(int(encoding_dim / 2), activation='relu')(encoder_binary)

# decoder
decoder_binary = Dense(int(encoding_dim / 2), activation='tanh')(encoder_binary)
decoder_binary = Dense(input_dim, activation='relu')(decoder_binary)

# modelo
autoencoder = Model(inputs=input_layer, outputs=decoder_binary)
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

history_binary = autoencoder.fit(X_train,
                          X_train,
                          epochs=5,
                          batch_size=32,
                          shuffle=True,
                          validation_data=(X_test, X_test),
                          verbose=1).history
predictions = autoencoder.predict(X_test)

mse = numpy.mean(numpy.power(X_test - predictions, 2), axis=1)

error = pandas.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
error.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
normal = error[(error['true_class'] == 0) & (error['reconstruction_error'] < 10)]
_ = ax.hist(normal.reconstruction_error.values, bins=10)
fig = plt.figure()
ax = fig.add_subplot(111)
fraud = error[(error['true_class'] == 1)]
_ = ax.hist(fraud.reconstruction_error.values, bins=10)
threshold = 3.0
groups = error.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    label = 'Fraud' if name == 1 else 'Normal'
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='', label=label)
    
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors='r', zorder=100, label='Threshold')
ax.legend()
plt.title('Erro de Reconstrução Para Diferentes Classes')
plt.ylabel('Erro de Reconstrução')
plt.xlabel('Index')
plt.show()
y_pred = [1 if e > threshold else 0 for e in error.reconstruction_error.values]
confusion_matrix(error.true_class, y_pred)
print(metrics.confusion_matrix(error.true_class,y_pred));
print(metrics.classification_report(error.true_class,y_pred));
