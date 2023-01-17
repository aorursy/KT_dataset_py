# import first

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

# change the style from the very beging

plt.style.use('ggplot')

%matplotlib inline

dataset=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/census.csv')

test_dataset=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')

ex=pd.read_csv('/kaggle/input/udacity-mlcharity-competition/example_submission.csv')

dataset.head()
dataset.dtypes
dataset.describe()
dataset.isna().sum()
#dataset=dataset.loc[:,dataset.isin([0]).mean() < .6]
features = dataset.drop(['income'],axis=1)

# solve the categorical data

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

features['workclass'] = encoder.fit_transform(features['workclass'])

features['education_level'] = encoder.fit_transform(features['education_level'])

features['marital-status'] = encoder.fit_transform(features['marital-status'])

features['occupation'] = encoder.fit_transform(features['occupation'])

features['relationship'] = encoder.fit_transform(features['relationship'])

features['race'] = encoder.fit_transform(features['race'])

features['sex'] = encoder.fit_transform(features['sex'])

features['native-country'] = encoder.fit_transform(features['native-country'])

dataset['income'] = encoder.fit_transform(dataset['income'])

goal=dataset['income']
# standrize the values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features = scaler.fit_transform(features)



# split into train and test

from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(random_state = 0)

logistic.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(goal_test, logistic.predict(test_set))

pd.DataFrame(cm)
print(logistic.score(train_set, goal_train))

print(logistic.score(test_set, goal_test))

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# split into train and test

from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)
accs_test = []

accs_train = []

ks = np.linspace(1, 20, 20)

for K in ks:

    classifier = KNeighborsClassifier(n_neighbors=int(K))

    classifier.fit(train_set, goal_train)

    cm = confusion_matrix(goal_test, classifier.predict(test_set))

    accs_train.append(classifier.score(train_set, goal_train))

    accs_test.append(classifier.score(test_set, goal_test))
plt.plot(ks, accs_train, label='train_acc')

plt.plot(ks, accs_test, label='test_acc')

plt.legend()

plt.title("accuracy versus K")
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

classifier = KNeighborsClassifier(n_neighbors = 2)

classifier.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(goal_test, classifier.predict(test_set))

pd.DataFrame(cm)
print("model accuracy on train: {:.4f}".format(classifier.score(train_set, goal_train)))

print("model accuracy on test: {:.4f}".format(classifier.score(test_set, goal_test)))
# split into train and test

from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(goal_test, classifier.predict(test_set))

print(classifier.score(train_set, goal_train))



print(classifier.score(test_set, goal_test))



pd.DataFrame(cm)
# split into train and test

from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()

clf.fit(train_set, goal_train)

clf.predict(test_set)





print(clf.score(train_set, goal_train))

clf.score(test_set, goal_test)
test_dataset.head()
test=test_dataset.drop(['Unnamed: 0'],axis=1)
test.isna().sum()
test=test.fillna(test.mean()) 

test.fillna(method='ffill', inplace=True)

test.isna().sum()
# solve the categorical data

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

test['workclass'] = encoder.fit_transform(test['workclass'])

test['education_level'] = encoder.fit_transform(test['education_level'])

test['marital-status'] = encoder.fit_transform(test['marital-status'])

test['occupation'] = encoder.fit_transform(test['occupation'])

test['relationship'] = encoder.fit_transform(test['relationship'])

test['race'] = encoder.fit_transform(test['race'])

test['sex'] = encoder.fit_transform(test['sex'])

test['native-country'] = encoder.fit_transform(test['native-country'])







# standrize the values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

test = scaler.fit_transform(test)
# split into train and test

from sklearn.model_selection import train_test_split

train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

classifier = KNeighborsClassifier(n_neighbors = 2)

classifier.fit(train_set, goal_train)

classifier.predict(test_set)

classifier.predict(test)
ex.head()
goal_test=ex[['id']]
goal_test['income']=classifier.predict(test)
goal_test.head()
goal_test.to_csv('submission.csv')