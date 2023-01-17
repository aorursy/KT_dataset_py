import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

sns.set_style('darkgrid')
qcm3 = pd.read_csv('/kaggle/input/qcm-sensor-alcohol-dataset/QCM3.csv', sep = ';')

qcm6 = pd.read_csv('/kaggle/input/qcm-sensor-alcohol-dataset/QCM6.csv', sep = ';')

qcm7 = pd.read_csv('/kaggle/input/qcm-sensor-alcohol-dataset/QCM7.csv', sep = ';')

qcm10 = pd.read_csv('/kaggle/input/qcm-sensor-alcohol-dataset/QCM10.csv', sep = ';')

qcm12 = pd.read_csv('/kaggle/input/qcm-sensor-alcohol-dataset/QCM12.csv', sep = ';')
print("Shape of qcm3: ", qcm3.shape)

print("Shape of qcm6: ", qcm6.shape)

print("Shape of qcm7: ", qcm7.shape)

print("Shape of qcm10: ", qcm10.shape)

print("Shape of qcm12: ", qcm12.shape)
dataset = pd.concat([qcm3, qcm6, qcm7, qcm10, qcm12])

print("Shape of dataset: ", dataset.shape)
dataset.head()
corr = dataset.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
dataset.loc[dataset["1-Octanol"] == 1, 'alcohol'] = '1-Octanol'

dataset.loc[dataset["1-Propanol"] == 1, 'alcohol'] = '1-Propanol'

dataset.loc[dataset["2-Butanol"] == 1, 'alcohol'] = '2-Butanol'

dataset.loc[dataset["2-propanol"] == 1, 'alcohol'] = '2-propanol'

dataset.loc[dataset["1-isobutanol"] == 1, 'alcohol'] = '1-isobutanol'
dataset['alcohol'].value_counts()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.799_0.201'], ax = axes[0])

axes[0].set_xlabel('0.799 - 0.201 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.799_0.201', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('0.799 - 0.201 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].set_ylabel('Fixed Acidity', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.799_0.201.1'], ax = axes[0])

axes[0].set_xlabel('0.799 - 0.201.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.799_0.201.1', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.799 - 0.201.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.700_0.300'], ax = axes[0])

axes[0].set_xlabel('0.700 - 0.300 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.700_0.300', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.700 - 0.300 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.700_0.300.1'], ax = axes[0])

axes[0].set_xlabel('0.700 - 0.300.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.700_0.300.1', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.700 - 0.300.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.600_0.400'], ax = axes[0])

axes[0].set_xlabel('0.600 - 0.400 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.600_0.400', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.600 - 0.400 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.600_0.400.1'], ax = axes[0])

axes[0].set_xlabel('0.600 - 0.400.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.600_0.400.1', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.600 - 0.400.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.501_0.499'], ax = axes[0])

axes[0].set_xlabel('0.501 - 0.499 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.501_0.499', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.501 - 0.499 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.501_0.499.1'], ax = axes[0])

axes[0].set_xlabel('0.501 - 0.499.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.501_0.499.1', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.501 - 0.499.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.400_0.600'], ax = axes[0])

axes[0].set_xlabel('0.400 - 0.600 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.400_0.600', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.400 - 0.600 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
f, axes = plt.subplots(1,2,figsize=(14,4))



sns.distplot(dataset['0.400_0.600.1'], ax = axes[0])

axes[0].set_xlabel('0.400 - 0.600.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[0].set_ylabel('Count', fontsize=14)

axes[0].yaxis.tick_left()



sns.swarmplot(x = 'alcohol', y = '0.400_0.600.1', data = dataset, hue = 'alcohol',ax = axes[1])

axes[1].set_xlabel('Quality', fontsize=14)

axes[1].set_ylabel('0.400 - 0.600.1 (Air Ratio - Gas Ratio) ml', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)



plt.show()
dataset.drop('alcohol', axis = 1, inplace = True)
dataset.head()
X = dataset.iloc[:, 0:10].values

y = dataset.iloc[:, [10,11,12,13,14]].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the second hidden layer

classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

history = classifier.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = 100, epochs = 3000, verbose = 1)
f, axes = plt.subplots(1,2,figsize=(14,4))



axes[0].plot(history.history['loss'])

axes[0].plot(history.history['val_loss'])

axes[0].set_xlabel('Loss', fontsize=14)

axes[0].set_ylabel('Epuch', fontsize=14)

axes[0].yaxis.tick_left()

axes[0].legend(['Train', 'Test'], loc='upper left')



axes[1].plot(history.history['acc'])

axes[1].plot(history.history['val_acc'])

axes[1].set_xlabel('Accuracy', fontsize=14)

axes[1].set_ylabel('Epoch', fontsize=14)

axes[1].yaxis.set_label_position("right")

axes[1].yaxis.tick_right()

axes[1].legend(['Train', 'Test'], loc='upper left')



plt.show()
from sklearn.metrics import confusion_matrix

from sklearn import metrics

y_pred = classifier.predict_classes(X_test)

cm = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred)
sns.heatmap(cm, annot=True, cmap = 'viridis')

plt.show()