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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df=pd.read_csv("E:\DATASET\diabetes.csv")
df.head()
df.shape
df.isnull().sum()

df_n=df[['Glucose','Age','DiabetesPedigreeFunction','BMI','Insulin','SkinThickness','BloodPressure']]
sns.pairplot(df_n , height=7, kind="reg",markers="+")
corr = df.corr()
print(corr)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=cmap, vmax=.3,square=True,linewidths=6, cbar_kws={"shrink": .5})
colormap = plt.cm.viridis
sns.pairplot(df.dropna(), hue='Outcome', palette="husl")
X = df.iloc[:,:-1].values
y = df.iloc[:,[-1]].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 0)
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
# make predictions for the testing set
y_pred = logReg.predict(X_test)
# check for accuracy
from sklearn import metrics
cm=metrics.confusion_matrix(y_test,y_pred)
cm
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predict')
plt.ylabel('Truth')
plt.show()
print( metrics.accuracy_score(y_test, y_pred))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 6, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

print( metrics.accuracy_score(y_test, y_pred))

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('predict')
plt.ylabel('Truth')
plt.show()




#Split dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


#Predict the response for test dataset
y_pred = clf.predict(X_test)
y_pred

# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1= confusion_matrix(y_test, y_pred)
cm1

plt.figure(figsize=(10,7))
sns.heatmap(cm1,annot=True)
plt.xlabel('predict')
plt.ylabel('Truth')
plt.show()



#Split dataset into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Create SVM model and fit the model
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)

#Making Predictions
y_pred = model.predict(X_test)
y_pred

#Evaluting the algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm2= confusion_matrix(y_test, y_pred)
cm2

plt.figure(figsize=(10,7))
sns.heatmap(cm1,annot=True)
plt.xlabel('predict')
plt.ylabel('Truth')
plt.show()