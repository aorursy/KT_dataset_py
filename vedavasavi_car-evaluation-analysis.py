import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir('../input'))
%time data = pd.read_csv('../input/car_evaluation.csv', header = None)

print("Shape of the Data: ", data.shape )
# Assigning names to the columns in the dataset

data.columns = ['Price', 'Maintenance Cost', 'Number of Doors', 'Capacity', 'Size of Luggage Boot', 'safety', 'Decision']

data.head()
data.sample(5)
data.tail()
data.info()
data.describe()
data.isnull().any()
data.columns
price = pd.crosstab(data['Price'], data['Decision'])
price.div(price.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each Price Category', fontsize = 20)
plt.xlabel('Price Range in Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
mc = pd.crosstab(data['Maintenance Cost'], data['Decision'])
mc.div(mc.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each Maintenance Cost Category', fontsize = 20)
plt.xlabel('Maintenance Cost in Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
safety = pd.crosstab(data['safety'], data['Decision'])
safety.div(safety.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each Safety Category', fontsize = 20)
plt.xlabel('Safety Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
doors = pd.crosstab(data['Number of Doors'], data['Decision'])
doors.div(doors.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each doors Category', fontsize = 20)
plt.xlabel('No. of Doors in Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
capacity = pd.crosstab(data['Capacity'], data['Decision'])
capacity.div(capacity.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each capacity Category', fontsize = 20)
plt.xlabel('Capacity in Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
luggage = pd.crosstab(data['Size of Luggage Boot'], data['Decision'])
luggage.div(safety.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (10, 7))

plt.title('Stacked Bar Graph to Depict portions of Decisions taken on each luggage size Category', fontsize = 20)
plt.xlabel('Luggage Size in Increasing Order', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.legend()
plt.show()
data['Decision'].value_counts().sort_index()
labels = ['acc', 'good', 'unacc', 'vgood']
colors = ['pink', 'lightblue', 'lightgreen', 'magenta']
size = [384, 69, 1210, 65]
explode = [0.1, 0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = "%.2f%%")
plt.title('A Pie Chart Representing Different Decisions', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# Label Encoding

data.Decision.replace(('unacc', 'acc', 'good', 'vgood'), (0, 1, 2, 3), inplace = True)

data['Decision'].value_counts()
data['Size of Luggage Boot'].value_counts().sort_index()
labels = ['big', 'medium', 'small',]
colors = ['red', 'cyan', 'orange']
size = [576, 576, 576]
explode = [0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('A Pie Chart Representing Different Luggage boot Sizes', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['Size of Luggage Boot'].replace(('small', 'med', 'big'), (0, 1, 2), inplace = True)

data['Size of Luggage Boot'].value_counts()
data['safety'].value_counts().sort_index()
labels = ['high', 'low', 'med']
colors = ['purple', 'cyan', 'lightblue']
size = [576, 576, 576]
explode = [0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('A Pie Chart Representing Different Safety Levels', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['safety'].replace(('low', 'med', 'high'), (0, 1, 2), inplace = True)

data['safety'].value_counts()
data['Price'].value_counts().sort_index()
labels = ['high', 'low', 'med', 'vhigh']
colors = ['crimson', 'blue', 'lightpink', 'maroon']
size = [432, 432, 432, 432]
explode = [0.1, 0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 120, autopct = '%1.1f%%')
plt.title('A Pie Chart Representing Different Price Ranges', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['Price'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)

data['Price'].value_counts()
data['Maintenance Cost'].value_counts().sort_index()
labels = ['high', 'low', 'med', 'vhigh']
colors = ['crimson', 'orange', 'red', 'pink']
size = [432, 432, 432, 432]
explode = [0.1, 0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, autopct='%1.1f%%', shadow = True, startangle = 120)
plt.title('A Pie Chart Representing Different Maintenance Cost Ranges', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['Maintenance Cost'].replace(('low', 'med', 'high', 'vhigh'), (0, 1, 2, 3), inplace = True)

data['Maintenance Cost'].value_counts()
data['Number of Doors'].value_counts().sort_index()
labels = ['high', 'low', 'med', 'vhigh']
colors = ['yellow', 'orange', 'green', 'pink']
size = [432, 432, 432, 432]
explode = [0.1, 0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, autopct='%1.1f%%', shadow = True, startangle = 120)
plt.title('A Pie Chart Representing Different no. of doors', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['Number of Doors'].replace('5more', 5, inplace = True)

data['Number of Doors'].value_counts()
data['Capacity'].value_counts().sort_index()
labels = ['2', '3', 'more']
colors = ['crimson', 'lightblue', 'lightgreen']
size = [576, 576, 576]
explode = [0.1, 0.1, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, autopct='%1.1f%%', shadow = True, startangle = 120)
plt.title('A Pie Chart Representing Different Capacities', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
# label Encoding

data['Capacity'].replace('more', 5, inplace = True)

data['Capacity'].value_counts()
data.sample(5)
plt.rcParams['figure.figsize'] = (12, 8)
ax = sns.violinplot(x = data['Price'], y = data['Decision'], color = 'g')
ax.set_title('Violin Plot to show relation between Price and Decision', fontsize = 20)
ax.set_xlabel('Price in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.violinplot(x = data['Maintenance Cost'], y = data['Decision'], color = 'y')
ax.set_title('Violin Plot to show relation between Maintenance Cost and Decision', fontsize = 20)
ax.set_xlabel('Maintenance Cost in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()

sns.set(style = 'dark', palette = 'muted', color_codes = True)
plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.violinplot(x = data['Number of Doors'], y = data['Decision'], color = 'r')
ax.set_title('Violin Plot to show relation between Number of Doors and Decision', fontsize = 20)
ax.set_xlabel('Number of Doors in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.violinplot(x = data['Capacity'], y = data['Decision'], color = 'b')
ax.set_title('Violin Plot to show relation between Capacity and Decision', fontsize = 20)
ax.set_xlabel('Capcity in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.violinplot(x = data['Size of Luggage Boot'], y = data['Decision'], color = 'm')
ax.set_title('Violin Plot to show relation between Size of Luggage Boot and Decision', fontsize = 20)
ax.set_xlabel('Size of Luggage Boot in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
plt.rcParams['figure.figsize'] = (12, 8)

ax = sns.boxplot(x = data['safety'], y = data['Decision'], color = 'w')
ax.set_title('Box Plot to show relation between safety and Decision', fontsize = 20)
ax.set_xlabel('safety in Increasing range', fontsize = 15)
ax.set_ylabel('Decision in Positive vibe')
plt.show()
data.shape
# splitting the dataset into dependent and independent variables

x = data.iloc[:,:6]
y = data.iloc[:, 6]

print("Shape of x: ", x.shape)
print("Shape of y: ", y.shape)
# Label Encoding once more so that we get higher accuracy
# we have 4 classes namely 0, 1, 2, 3
# It would be very beneficial if we combine 0 and 1 as 0 and 1 and 2 as 1

data.Decision.replace((0, 1, 2, 3), (0, 0, 1, 1), inplace = True)

data['Decision'].value_counts()
# splitting the dataset into train and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 0)

print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_test: ", y_test.shape)
# standardization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#importing libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# creating a model
model = LogisticRegression(C = 1)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 3)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 2)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 4)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 5)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 6)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 7)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 8)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 9)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier(max_depth = 10)
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = DecisionTreeClassifier()
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#  plotting the graph for performance of decision trees with different max_depth values

max_depth = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
Training_Accuracy = np.array([93.4, 93.5, 97.3, 97.3, 94.6, 97.3, 97.3, 98.4, 98.07])
Testing_Accuracy = np.array([92.02, 92.02, 95.9, 95.9, 97.8, 98.4, 99.3, 99.7, 100.0])

plt.rcParams['figure.figsize'] = (10, 7)
plt.plot(max_depth, Training_Accuracy, 'r--')
plt.plot(max_depth, Testing_Accuracy, 'b-*')
plt.title('Performance of Decision Tree with Different Values of max depth', fontsize = 20)
plt.xlim([1, 11])
plt.ylim([90, 100])
plt.xlabel('Max Depth ', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.legend()
plt.show()
#importing libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# creating a model
model = RandomForestClassifier()
# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# checking the feature importance

imp = model.feature_importances_

print(imp)

import matplotlib.pyplot as plt
import seaborn as sns

Acc = np.array([97.6, 98.4, 99.2])
count = np.array([1, 2, 3])
plt.scatter(count, Acc, color = 'green')
plt.title('1 Represents Logistic Regression, 2 Represents Decision Tree and 3 Represents Random Forest \
It plots a graph to compare performance of these Classifiers', fontsize = 20)
plt.xlabel('Classifiers', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)

plt.show()
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 1)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 2)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 3)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 4)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 5)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 6)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 7)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 8)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 9)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.neighbors import KNeighborsClassifier

# creating a model
model = KNeighborsClassifier(n_neighbors = 10)

# feeding the training data into the model
model.fit(x_train, y_train)

# predicting the values for x-test
y_pred = model.predict(x_test)

# finding the training and testing accuracy
print("Training Accuracy: ",model.score(x_train, y_train))
print("Testing Accuracy: ", model.score(x_test, y_test))

# printing the confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
#  plotting the graph for performance of decision trees with different max_depth values

max_depth = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Training_Accuracy = np.array([100, 98.9, 99.7, 99.7, 99.8, 99.1, 99.5, 99.2, 99.3, 99.2])
Testing_Accuracy = np.array([98.8, 96.9, 99.2, 98.4, 98.8, 97.6, 98.8, 98.8, 98.8, 98])

plt.rcParams['figure.figsize'] = (10, 7)
plt.plot(max_depth, Training_Accuracy, 'y--')
plt.plot(max_depth, Testing_Accuracy, 'g-*')
plt.title('Performance of KNN Classifiers with Different Values of neighbors', fontsize = 20)
plt.xlim([0, 11])
plt.ylim([95, 100])
plt.xlabel('Max Depth ', fontsize = 15)
plt.ylabel('Accuracy', fontsize = 15)
plt.legend()
plt.show()
data.columns
data.head()
'''
predicting if the costumer's Decision for buying a car with following specifications will be good or not

Price : 2
Maintenance Cost = 1
Number of Doors = 2
Capacity = 3
Size of Luggage Boot = 1
safety = 2

'''

new_prediction = model.predict(sc.transform(np.array([[2, 1, 2, 3, 1, 2]])))

new_prediction = (new_prediction > 0.5 )
print(new_prediction)
'''
predicting if the costumer's Decision for buying a car with following specifications will be good or not

Price : 1
Maintenance Cost = 1
Number of Doors = 3
Capacity = 3
Size of Luggage Boot = 2
safety = 3

'''
new_prediction = model.predict(sc.transform(np.array([[0, 2, 3, 3, 2, 3]])))

new_prediction = (new_prediction > 0.5 )
print(new_prediction)