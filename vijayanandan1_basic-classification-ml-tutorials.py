features = [[140,'smooth'], [130,'smooth'], [150,'bumpy'], [170, 'bumpy']]

labels = ['apple','apple','orange','orange']
#PreProcessing - Label Enconder

features = [[140,1], [130,1], [150,2], [170, 2]]

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

y = le.fit(['apple','apple','orange','orange'])

labels = le.transform(['apple','apple','orange','orange'])

labels

from sklearn import tree

clf = tree.DecisionTreeClassifier()
#Train the Model:
clf = clf.fit(features, labels)
#Making Predictions
print( clf.predict([[140,0]]))
# Machine learning-Iris classification
from  sklearn import  datasets

iris=datasets.load_iris()

iris 

x=iris.data

y=iris.target

import pandas as pd

df = pd.DataFrame(x,y)

df



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
#Using Desission Tree

from sklearn import tree

classifier=tree.DecisionTreeClassifier()
#Using K-Neart Classifier

from sklearn import neighbors

classifierkn=neighbors.KNeighborsClassifier()
#Traing the Model usind Decision Tree

classifier.fit(x_train,y_train)
#Training the Model Using KN

classifierkn.fit(x_train,y_train)
predictions=classifier.predict(x_test)
#Getting Accuracy 

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))