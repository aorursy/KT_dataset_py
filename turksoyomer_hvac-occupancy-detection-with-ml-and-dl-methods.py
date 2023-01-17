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



# Importing necessary libraries for this notebook.

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)

import plotly.express as px

import plotly.graph_objects as go



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.regularizers import l2, l1

from keras.metrics import BinaryAccuracy
datatest = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatest.txt")

datatest2 = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatest2.txt")

datatraining = pd.read_csv("/kaggle/input/occupancy-detection-data-set-uci/datatraining.txt")
print(datatest.info())

datatest.head()
print(datatest2.info())

datatest2.head()
print(datatraining.info())

datatraining.head()
datatest['date'] = pd.to_datetime(datatest['date'])

datatest2['date'] = pd.to_datetime(datatest2['date'])

datatraining['date'] = pd.to_datetime(datatraining['date'])

datatest.reset_index(drop=True, inplace=True)

datatest2.reset_index(drop=True, inplace=True)

datatraining.reset_index(drop=True, inplace=True)
datatraining.describe()
scaler = MinMaxScaler()

columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']

scaler.fit(np.array(datatraining[columns]))

datatest[columns] = scaler.transform(np.array(datatest[columns]))

datatest2[columns] = scaler.transform(np.array(datatest2[columns]))

datatraining[columns] = scaler.transform(np.array(datatraining[columns]))
plt.figure(figsize=(10,10))

plt.title('Box Plot for Features', fontdict={'fontsize':18})

ax = sns.boxplot(data=datatraining.drop(['date', 'Occupancy'],axis=1), orient="h", palette="Set2")

print(datatraining.drop(['date', 'Occupancy'],axis=1).describe())
plt.figure(figsize=(10,8))

plt.title('Correlation Table for Features', fontdict={'fontsize':18})

ax = sns.heatmap(datatraining.corr(), annot=True, linewidths=.2)
data = datatraining.copy()

data.Occupancy = data.Occupancy.astype(str)

fig = px.scatter_3d(data, x='Temperature', y='Humidity', z='CO2', size='Light', color='Occupancy', color_discrete_map={'1':'red', '0':'blue'})

fig.update_layout(scene_zaxis_type="log", title={'text': "Features and Occupancy",

                                                'y':0.9,

                                                'x':0.5,

                                                'xanchor': 'center',

                                                'yanchor': 'top'})

iplot(fig)
sns.set(style="darkgrid")

plt.title("Occupancy Distribution", fontdict={'fontsize':18})

ax = sns.countplot(x="Occupancy", data=datatraining)
hours_1 = []

hours_0 = []

for date in datatraining[datatraining['Occupancy'] == 1]['date']:

    hours_1.append(date.hour)

for date in datatraining[datatraining['Occupancy'] == 0]['date']:

    hours_0.append(date.hour)
plt.figure(figsize=(8,8))

ax = sns.distplot(hours_1)

ax = sns.distplot(hours_0)
datatest['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest['date']]

datatest2['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest2['date']]

datatraining['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatraining['date']]

datatraining.sample(10)
X_train = datatraining.drop(columns=['date', 'Occupancy'], axis=1)

y_train = datatraining['Occupancy']

X_validation = datatest.drop(columns=['date', 'Occupancy'], axis=1)

y_validation = datatest['Occupancy']

X_test = datatest2.drop(columns=['date', 'Occupancy'], axis=1)

y_test = datatest2['Occupancy']
# parameter-tuning for knn

n_neighbors_list = [7,15,45,135]

weights_list = ['uniform', 'distance']

metric_list = ['euclidean', 'manhattan']

accuracies = {}

for n in n_neighbors_list:

    for weight in weights_list:

        for metric in metric_list:

            knn_model = KNeighborsClassifier(n_neighbors=n, weights=weight, metric=metric)

            knn_model.fit(X_train, y_train)

            accuracy = knn_model.score(X_validation, y_validation)

            accuracies[str(n)+"/"+weight+"/"+metric] = accuracy
plotdata = pd.DataFrame()

plotdata['Parameters'] = accuracies.keys()

plotdata['Accuracy'] = accuracies.values()

fig = px.line(plotdata, x="Parameters", y="Accuracy")

fig.update_layout(title={'text': "Accuracies for Different Hyper-Parameters",

                                                'x':0.5,

                                                'xanchor': 'center',

                                                'yanchor': 'top'})

iplot(fig)
knn_model = KNeighborsClassifier(n_neighbors=135)

knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_validation)

plt.title("KNN Confusion Matrix for Validation Data", fontdict={'fontsize':18})

ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")
svm_model = SVC()

svm_model.fit(X_train, y_train)

print("Accuracy for SVM on validation data: {}%".format(round((svm_model.score(X_validation, y_validation)*100),2)))
y_pred = svm_model.predict(X_validation)

plt.title("SVM Confusion Matrix for Validation Data", fontdict={'fontsize':18})

ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")
# NN without regularization

model1 = Sequential()

model1.add(Dense(32, activation='relu', input_dim=6))

model1.add(Dense(16, activation='relu'))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history1 = model1.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))
# NN with 0.2 dropout ratio before the hidden layer.

model2 = Sequential()

model2.add(Dense(32, activation='relu', input_dim=6))

model2.add(Dropout(0.2))

model2.add(Dense(16, activation='relu'))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history2 = model2.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))
# NN with L1(Lasso) regularization

model3 = Sequential()

model3.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l1(l=0.01)))

model3.add(Dense(16, activation='relu', kernel_regularizer=l1(l=0.01)))

model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history3 = model3.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))
# NN with L2(Ridge) Regularization

model4 = Sequential()

model4.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l2(l=0.01)))

model4.add(Dense(16, activation='relu', kernel_regularizer=l2(l=0.01)))

model4.add(Dense(1, activation='sigmoid'))

model4.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

history4 = model4.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_validation, y_validation))
loss1 = history1.history['loss']

val_loss1 = history1.history['val_loss']

loss2 = history2.history['loss']

val_loss2 = history2.history['val_loss']

loss3 = history3.history['loss']

val_loss3 = history3.history['val_loss']

loss4 = history4.history['loss']

val_loss4 = history4.history['val_loss']





fig = go.Figure()

fig.add_trace(go.Scatter(x=np.arange(len(loss1)), y=loss1,

                    name='Training Loss without Regularization', line=dict(color='royalblue')))

fig.add_trace(go.Scatter(x=np.arange(len(val_loss1)), y=val_loss1,

                    name='Validation Loss without Regularization', line = dict(color='firebrick')))



fig.add_trace(go.Scatter(x=np.arange(len(loss2)), y=loss2,

                    name='Training Loss with Dropout', line=dict(color='royalblue', dash='dash')))

fig.add_trace(go.Scatter(x=np.arange(len(val_loss2)), y=val_loss2,

                    name='Validation Loss with Dropout', line = dict(color='firebrick', dash='dash')))



fig.add_trace(go.Scatter(x=np.arange(len(loss3)), y=loss3,

                    name='Training Loss with L1 Regularization', line=dict(color='royalblue', dash='dot')))

fig.add_trace(go.Scatter(x=np.arange(len(val_loss3)), y=val_loss3,

                    name='Validation Loss with L1 Regularization', line = dict(color='firebrick', dash='dot')))



fig.add_trace(go.Scatter(x=np.arange(len(loss4)), y=loss4,

                    name='Training Loss with L2 Regularization', line=dict(color='royalblue', dash='longdashdot')))

fig.add_trace(go.Scatter(x=np.arange(len(val_loss4)), y=val_loss4,

                    name='Validation Loss with L2 Regularization', line = dict(color='firebrick', dash='longdashdot')))





fig.update_layout(xaxis_title='Epochs',

                  yaxis_title='Loss',

                  title={'text': "Training and Validation Losses for Different Models",

                                                'x':0.5,

                                                'xanchor': 'center',

                                                'yanchor': 'top'})

iplot(fig)
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=6, kernel_regularizer=l2(l=0.01)))

model.add(Dropout(0.3))

model.add(Dense(32, activation='relu', kernel_regularizer=l2(l=0.01)))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',

                loss='binary_crossentropy',

                metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32)
print("Accuracy for SVM on test data: {}%\n".format(round((svm_model.score(X_test, y_test)*100),2)))

print("Accuracy for Neural Network model on test data: {}%".format(round((model.evaluate(X_test, y_test)[1]*100),2)))
y_pred = svm_model.predict(X_test)

plt.title("SVM Confusion Matrix for Test Data", fontdict={'fontsize':18})

ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
y_pred = model.predict(X_test)

threshold = 0.6

y_pred = [1 if i >= threshold else 0 for i in y_pred]

plt.title("Neural Network Confusion Matrix for Test Data", fontdict={'fontsize':18})

ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")