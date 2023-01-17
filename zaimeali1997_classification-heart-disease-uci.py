import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head(10)
print("Total Rows in a Data: ", data.shape[0])

print("Total Columns in a Data: ", data.shape[1])
print("\t****************")

print("\tData Information:")

print("\t****************\n")

data.info()
print("\t****************")

print("\tData Describe:")

print("\t****************\n")

data.describe()
print("Null Values in each column:")

print(data.isna().any())
print("Unique Values in each column:")

print("----------------------------\n")

cols = list(data.columns)

for c in cols:

    print(c.upper(), ":", data[c].unique(), "\n")
print("The most younger person in a data:")

print(data.iloc[data.age.idxmin()])
print("The most older person in a data:")

print(data.iloc[data.age.idxmax()])
fig, ax = plt.subplots(figsize=(14, 8))

sns.countplot(y=data.age)

# sns.countplot(y=data['age'], order=data['age'].value_counts().index)

plt.title("Age Count")

plt.xlabel("Count")

plt.ylabel("Age")

ax.set(xticks=range(0, 21))

plt.show()
sns.boxplot(data.age)

plt.show()
fig, ax = plt.subplots(figsize=(14, 5))

ax = plt.plot(data.groupby('age')['target'].mean())

plt.xticks(range(min(data.age)-1, max(data.age)+1))

plt.show()
data[data.age == 61]
data[data.age > 70]
print("Number of Unique Values in Gender Column: ", data.sex.nunique(), "\n")

print("Unique Values count are:")

print(data.sex.value_counts())
ax = sns.countplot(data.sex, palette='Set3')

plt.xlabel("Sex")

plt.title("Gender Occuring in a Dataset")

plt.show()
data.groupby('sex')['target'].value_counts()
fig, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x='sex', hue='target', data=data, palette='Set2')

plt.xlabel("Gender (0=Female, 1=Male)")

plt.show()
ax = sns.catplot(x='target', col='sex', data=data, kind='count')
print("Number of Unique Values in Chest Pain Type Column: ", data.cp.nunique(), "\n")

print("Unique Values count are:")

print(data.cp.value_counts())
sns.countplot(data.cp)

plt.title("Chest Pain Count")

plt.xlabel("Chest Pain Type")

plt.show()
data.groupby('cp')['target'].value_counts()
ax = sns.countplot(x='cp', data=data, hue='target')

plt.xlabel("Chest Pain Type (lowest to highest)")

plt.ylabel("Count")

plt.show()
print("Number of Unique Values in Fasting Blood Sugar Column: ", data.fbs.nunique(), "\n")

print("Unique Values count are:")

print(data.fbs.value_counts())
data.groupby('fbs')['target'].value_counts()
sns.countplot(x='fbs', hue='target', data=data, palette='Set2')

plt.xlabel("Fasting Blood Sugar (0=False, 1=True)")

plt.ylabel("Count")

plt.show()
print("Number of Unique Values in Resting Electrocardiographic Column: ", data.restecg.nunique(), "\n")

print("Unique Values count are:")

print(data.restecg.value_counts())
data.groupby('restecg')['target'].value_counts()
sns.countplot(x=data['restecg'])

plt.xlabel("Resting Electrocardiographic Results")

plt.show()
print("Number of Unique Values in Thalach Column: ", data.thalach.nunique(), "\n")

print("Unique Values count are:")

print(data.thalach.value_counts())
sns.distplot(data.thalach)

plt.xlabel("Heart Rate")

plt.show()
sns.scatterplot(x='age', y='thalach', data=data)

plt.xlabel('Age')

plt.ylabel('Heart Rate')

plt.show()
print("Number of Unique Values in Exercise Induced Angina Column: ", data.exang.nunique(), "\n")

print("Unique Values count are:")

print(data.exang.value_counts())
data.groupby('exang')['target'].value_counts()
sns.countplot(y=data.exang)

plt.xlabel("Count")

plt.ylabel("Exercised Induced Angina")

plt.show()
ax = sns.countplot(x='exang', data=data, hue='target')

plt.xlabel("Exercised Induced Angina (0=No, 1=Yes)")

plt.ylabel("Count")

plt.show()
plt.title("The Slope of the Peak Exercise ")

sns.countplot(x=data['slope'])

plt.xlabel("Slope")

plt.ylabel("Count")

plt.show()
plt.title("The Slope of the Peak Exercise ")

sns.countplot(x=data['slope'], hue=data['target'], data=data)

plt.xlabel("Slope")

plt.ylabel("Count")

plt.show()
sns.countplot(x=data['ca'])

plt.title("Number of Major Vessels")

plt.show()
sns.countplot(x=data['thal'])

plt.title("Blood Disorder called Thalassemia")

plt.show()
sns.countplot(x=data['target'])

plt.title("Target Count")

plt.show()
data.corr()['target'].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(data.corr(), annot=True)

plt.show()
sns.pairplot(data[['cp', 'restecg', 'thalach', 'slope', 'target']])

plt.show()
X1 = data.iloc[:, :-1]

y1 = data.iloc[:, -1]



df = data[['cp', 'restecg', 'thalach', 'slope', 'target']]

X2 = df.iloc[:, :-1]

y2 = df.iloc[:, -1]



print("Columns in Data1: ", list(data.columns))

print("Data1 Shape: ", data.shape)

print("X1 Shape: ", X1.shape)

print("y1 Shape: ", y1.shape)

print()



print("Columns in Data2: ", list(df.columns))

print("Data2 Shape: ", df.shape)

print("X2 Shape: ", X2.shape)

print("y2 Shape: ", y2.shape)
from sklearn.model_selection import train_test_split

train_X1, test_X1, train_y1, test_y1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

train_X2, test_X2, train_y2, test_y2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(train_X1, train_y1)

pred_dtc1 = DTC.predict(test_X1)

score_dtc1 = round(DTC.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with Decision Tree Classifier is: ", score_dtc1, "%")

print()

DTC.fit(train_X2, train_y2)

pred_dtc2 = DTC.predict(test_X2)

score_dtc2 = round(DTC.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with Decision Tree Classifier is: ", score_dtc2, "%")
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier() # n_estimators = 100

RFC.fit(train_X1, train_y1)

pred_rfc1 = RFC.predict(test_X1)

score_rfc1 = round(RFC.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with Random Forest Classifier is: ", score_rfc1, "%")

print()

RFC.fit(train_X2, train_y2)

pred_rfc2 = RFC.predict(test_X2)

score_rfc2 = round(RFC.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with Random Forest Classifier is: ", score_rfc2, "%")
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(train_X1, train_y1)

pred_lr1 = LR.predict(test_X1)

score_lr1 = round(LR.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with Logistic Regression is: ", score_lr1, "%")

print()

LR.fit(train_X2, train_y2)

pred_lr2 = LR.predict(test_X2)

score_lr2 = round(LR.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with Logistic Regression is: ", score_lr2, "%")
from sklearn.svm import SVC

SC = SVC()

SC.fit(train_X1, train_y1)

pred_sc1 = SC.predict(test_X1)

score_sc1 = round(SC.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with Support Vector Classifier is: ", score_sc1, "%")

print()

SC.fit(train_X2, train_y2)

pred_sc2 = SC.predict(test_X2)

score_sc2 = round(SC.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with Support Vector Classifier is: ", score_sc2, "%")
from sklearn.linear_model import SGDClassifier

SGD = SGDClassifier()

SGD.fit(train_X1, train_y1)

pred_sgd1 = SGD.predict(test_X1)

score_sgd1 = round(SGD.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with Stochastic Gradient Descent Classifier is: ", score_sgd1, "%")

print()

SGD.fit(train_X2, train_y2)

pred_sgd2 = SGD.predict(test_X2)

score_sgd2 = round(SGD.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with Stochastic Gradient Descent Classifier is: ", score_sgd2, "%")
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=10) # default n_neighbors = 5

KNN.fit(train_X1, train_y1)

pred_knn1 = KNN.predict(test_X1)

score_knn1 = round(KNN.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with K-Nearest Neighbor Classifier is: ", score_knn1, "%")

print()

KNN.fit(train_X2, train_y2)

pred_knn2 = KNN.predict(test_X2)

score_knn2 = round(KNN.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with K-Nearest Neighbor Classifier is: ", score_knn2, "%")
from sklearn.ensemble import AdaBoostClassifier

ABC = AdaBoostClassifier() # n_estimators = 50 default

ABC.fit(train_X1, train_y1)

pred_abc1 = ABC.predict(test_X1)

score_abc1 = round(ABC.score(test_X1, test_y1)*100, 2)

print("Accuracy of Data 1 with AdaBoost Classifier is: ", score_abc1, "%")

print()

ABC.fit(train_X2, train_y2)

pred_abc2 = ABC.predict(test_X2)

score_abc2 = round(ABC.score(test_X2, test_y2)*100, 2)

print("Accuracy of Data 2 with AdaBoost Classifier is: ", score_abc2, "%")
model1 = pd.DataFrame(

    {

        'Models': [

            'Decision Tree Classifier',

            'Random Forest Classifier',

            'Logistic Regression',

            'Support Vector Machine',

            'Stochastic Gradient Descent',

            'K-Nearest Neighbors',

            'AdaBoost Classifier'

        ],

        'Scores': [

            score_dtc1,

            score_rfc1,

            score_lr1,

            score_sc1,

            score_sgd1,

            score_knn1,

            score_abc1

        ],

    }

)

model2 = pd.DataFrame(

    {

        'Models': [

            'Decision Tree Classifier',

            'Random Forest Classifier',

            'Logistic Regression',

            'Support Vector Machine',

            'Stochastic Gradient Descent',

            'K-Nearest Neighbors',

            'AdaBoost Classifier'

        ],

        'Scores': [

            score_dtc2,

            score_rfc2,

            score_lr2,

            score_sc2,

            score_sgd2,

            score_knn2,

            score_abc2

        ],

    }

)
print("Models who are Train on Data 1:")

model1
print("Models who are Train on Data 2:")

model2