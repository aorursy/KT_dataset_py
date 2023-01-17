import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





maindf = pd.read_csv('../input/train.csv')

testdf = pd.read_csv('../input/test.csv')

maindf.head()
#Check columns informations

maindf.info()

print("----------------------------")

testdf.info()
# Create a train set descriptors and result



#Assumes that the PassengerID,name,ticket,cabin,and embarked do not matter

#Assumes that Fare is correlated to pclass

X = maindf.drop(['PassengerId','Survived','Name','Ticket','Cabin','Embarked','Fare'],axis=1)

Xtest = testdf.drop(['PassengerId','Name','Ticket','Cabin','Embarked','Fare'],axis=1)

y = maindf['Survived']
X.head()
X.shape
X.describe()
#notice that the count of age is below than 714, indicate that there are empty values

#What are the empty values in each column of X

X.apply(lambda x: sum(x.isnull()),axis=0) 
#Refill empty values with Mean

X['Age'].fillna(maindf['Age'].mean(), inplace=True)

Xtest['Age'].fillna(testdf['Age'].mean(), inplace=True)
age_bins = [0, 2, 10, 17, 40, 65, 100]

age_group = [0,1,2,3,4,5]

X['Age']= pd.cut(X['Age'], age_bins, labels=age_group)

# age_group = ['baby', 'child', 'adolescence', 'young adult','adult','elderly']

Xtest['Age']= pd.cut(Xtest['Age'], age_bins, labels=age_group)



# fare_bins = [0,7.910400, 14.454200, 31.000000, 512.329200]

# fare_group = ['low', 'med', 'high', 'very high']

# X['Fare']= pd.cut(X['Fare'], fare_bins, labels=fare_group)
#Map Sex to 0,1

X['Sex'] = X['Sex'].map({'male':0,'female':1})

Xtest['Sex'] = Xtest['Sex'].map({'male':0,'female':1})
#SibSp would only care if the person brings spouse or sibling

#Parch would only care if the person brings parent or children



X['SibSp'][X['SibSp']>0]=1

X['Parch'][X['Parch']>0]=1

Xtest['SibSp'][Xtest['SibSp']>0]=1

Xtest['Parch'][Xtest['Parch']>0]=1



# X['WithSomebody'] = X['SibSp']+X['Parch']

X.head()
X.shape
y.shape
kidsorwoman = y[(X['Age']<3) | (X['Sex'] == 1)]

kidsorwoman.value_counts()

#From this result we know that kids or women are more likely to survive than die.
nosiblingorparent = y[X['SibSp']+ X['Parch']<1]

hassiblingorparent = y[X['SibSp']+ X['Parch']>=1]

print(nosiblingorparent.value_counts())

print('____________________')

print(hassiblingorparent.value_counts())



#From here we can see that the likelihood to survive is more if a person has anyone with him/her
import numpy as np

from sklearn import preprocessing,cross_validation

from sklearn.tree import DecisionTreeClassifier
#splitting the train and test sets

X_train, X_test, y_train,y_test= cross_validation.train_test_split(X,y,test_size=0.2)
clf = DecisionTreeClassifier()

clf.fit(X_train,y_train)
pd.DataFrame(X_train,y_train).head()

accuracy = clf.score(X_test,y_test)

print(accuracy)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
# Predict Test Set



favorite_clf = RandomForestClassifier()

favorite_clf.fit(X_train, y_train)

y_pred = pd.DataFrame(favorite_clf.predict(Xtest))
# Tidy and Export Submission

submission = pd.DataFrame({

        "PassengerId": testdf["PassengerId"]    

    })

submission['Survived'] = y_pred



submission.to_csv('submission.csv', index = False)

submission.tail()
submission.shape