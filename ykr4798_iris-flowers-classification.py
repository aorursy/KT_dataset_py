# First, we'll import pandas, a data processing and CSV file I/O library

import pandas as pd



# We'll also import seaborn, a Python graphing library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)
# Next, we'll load the Iris flower dataset, which is in the "../input/" directory

iris = pd.read_csv("../input/iris/Iris.csv") # the iris dataset is now a Pandas DataFrame



# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do

iris.head()
iris.info()
iris.shape
iris.describe()
# Let's see how many examples we have of each species

iris["Species"].value_counts()
# Drop Unwanted Columns

iris.drop('Id',axis=1,inplace=True)
# The first way we can plot things is using the .plot extension from Pandas dataframes

# We'll use this to make a scatterplot of the Iris features.

# For Sepal 

fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='royalblue',label='Iris-setosa')

iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='darkorange',label='Iris-versicolor',ax=fig)

iris[iris.Species == 'Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='forestgreen',label='Iris-virginica',ax=fig)



fig.set_xlabel('Sepal Length')

fig.set_ylabel('Sepal Width')

fig.set_title('Sepal Length vs Width');
# The first way we can plot things is using the .plot extension from Pandas dataframes

# We'll use this to make a scatterplot of the Iris features.

# For Sepal 

fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='royalblue',label='Iris-setosa')

iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='darkorange',label='Iris-versicolor',ax=fig)

iris[iris.Species == 'Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='forestgreen',label='Iris-virginica',ax=fig)



fig.set_xlabel('Petal Length')

fig.set_ylabel('Petal Width')

fig.set_title('Petal Length vs Width');
sns.pairplot(iris, hue='Species',palette='Set1');
# importing alll the necessary packages to use the various classification algorithms

X = iris.drop('Species', axis=1)

y = iris['Species']



from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Logistic Regression (LR)

# for Logistic Regression algorithm

# Training

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train,y_train)

y_pred= classifier.predict(X_test)



# Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  

print(confusion_matrix(y_test, y_pred))  

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test,y_pred))
# Support Vector Machine (SVM)

#for Support Vector Machine (SVM) Algorithm

# Training

from sklearn import svm

classifier1 = svm.SVC()

classifier1.fit(X_train,y_train)

y_pred1 = classifier1.predict(X_test)



# Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  

print(confusion_matrix(y_test, y_pred1))  

print(classification_report(y_test, y_pred1))

print(accuracy_score(y_test,y_pred1))
# Decision Tree

#for using Decision Tree Algoithm

# Training

from sklearn.tree import DecisionTreeClassifier

classifier2 = DecisionTreeClassifier()

classifier2.fit(X_train,y_train)

y_pred2 = classifier2.predict(X_test)



# Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  

print(confusion_matrix(y_test, y_pred2))  

print(classification_report(y_test, y_pred2))

print(accuracy_score(y_test,y_pred2))
# K-Nearest Neighbors (KNN)

#for using K-Nearest Neighbors Algoithm

# Training

from sklearn.neighbors import KNeighborsClassifier

classifier3 = KNeighborsClassifier(n_neighbors=5)

classifier3.fit(X_train,y_train)

y_pred3 = classifier3.predict(X_test)



# Evaluating the Algorithm

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  

print(confusion_matrix(y_test, y_pred3))  

print(classification_report(y_test, y_pred3))

print(accuracy_score(y_test,y_pred3))
petal=iris[['PetalLengthCm','PetalWidthCm','Species']]

sepal=iris[['SepalLengthCm','SepalWidthCm','Species']]
train_p,test_p=train_test_split(petal,test_size=0.3,random_state=0)  #petals

train_x_p=train_p[['PetalWidthCm','PetalLengthCm']]

train_y_p=train_p.Species

test_x_p=test_p[['PetalWidthCm','PetalLengthCm']]

test_y_p=test_p.Species





train_s,test_s=train_test_split(sepal,test_size=0.3,random_state=0)  #Sepal

train_x_s=train_s[['SepalWidthCm','SepalLengthCm']]

train_y_s=train_s.Species

test_x_s=test_s[['SepalWidthCm','SepalLengthCm']]

test_y_s=test_s.Species
from sklearn import metrics #for checking the model accuracy
# Logistic Regression (LR)

# for Logistic Regression algorithm

# Training

model = LogisticRegression()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

# Evaluating the Algorithm

print('The accuracy of the Logistic Regression using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

# Evaluating the Algorithm

print('The accuracy of the Logistic Regression using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
# Support Vector Machine (SVM)

#for Support Vector Machine (SVM) Algorithm

# Training

model=svm.SVC()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

# Evaluating the Algorithm

print('The accuracy of the SVM using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model=svm.SVC()

model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

# Evaluating the Algorithm

print('The accuracy of the SVM using Sepal is:',metrics.accuracy_score(prediction,test_y_s))
# Decision Tree

#for using Decision Tree Algoithm

# Training

model=DecisionTreeClassifier()

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

# Evaluating the Algorithm

print('The accuracy of the Decision Tree using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

# Evaluating the Algorithm

print('The accuracy of the Decision Tree using Sepals is:',metrics.accuracy_score(prediction,test_y_s))
# K-Nearest Neighbors (KNN)

#for using K-Nearest Neighbors Algoithm

# Training

model=KNeighborsClassifier(n_neighbors=3) 

model.fit(train_x_p,train_y_p) 

prediction=model.predict(test_x_p) 

# Evaluating the Algorithm

print('The accuracy of the KNN using Petals is:',metrics.accuracy_score(prediction,test_y_p))



model.fit(train_x_s,train_y_s) 

prediction=model.predict(test_x_s) 

# Evaluating the Algorithm

print('The accuracy of the KNN using Sepals is:',metrics.accuracy_score(prediction,test_y_s))