# Import Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



%matplotlib inline
# Import Dataset

df = pd.read_csv("../input/train.csv", delimiter = ",")

df.head(2)
# Missing Values



fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
# we can delete "Name, ID, Ticket and Cabin" parameters

df.drop(["PassengerId", "Name","Ticket", "Cabin"], axis = 1, inplace = True) # Drop 
df.head()
list(df.columns)
df.info() # we have missing values
df.describe().T
df['Survived'].value_counts() * 100 / len(df)
df['Survived'].value_counts().plot(kind='bar')

plt.show()
df['Pclass'].value_counts().plot(kind='pie')

plt.show()
df['Sex'].value_counts()
df['Embarked'].value_counts()
df.Age.plot(kind='hist', bins=20);
# Embarked - Age

sns.catplot(x="Embarked",y="Age", data = df); 
%config InlineBackend.figure_format = "retina"

sns.catplot(x="Embarked",y="Age", data = df, hue = "Sex");
sns.catplot(x = "Sex", y = "Age", hue = "Survived", kind = "point", data = df);
sns.catplot(x = "Sex", y = "SibSp", hue = "Survived", kind = "point", data = df);
sns.boxplot("SibSp","Age", data = df);
sns.boxplot("SibSp","Fare", data = df);
sns.boxplot("SibSp","Fare", hue = "Survived",data = df);
sns.catplot("Survived","SibSp", kind = "violin", hue = "Sex", data = df);
g = sns.FacetGrid(data = df, row = "Sex", col = "Survived")

g.map(plt.hist, "Parch");
print(pd.isnull(df).sum())
df.hist(bins=10,figsize=(8,5),grid=False);
g = sns.FacetGrid(df, hue="Survived", col="Pclass", margin_titles=True,palette={1:"blue", 0:"red"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
df.corr()["Survived"]
g = sns.FacetGrid(df, hue = 'Survived', aspect=3)

g.map(sns.kdeplot, 'Age', shade= True )

g.set(xlim=(0 , df['Age'].max()))

g.add_legend();
from sklearn.preprocessing import LabelEncoder



labelencoder_Sex = LabelEncoder()

df.Sex = labelencoder_Sex.fit_transform(df.Sex)
df.Embarked = df["Embarked"].astype("str")



labelencoder_Embarked = LabelEncoder()

df.Embarked = labelencoder_Embarked.fit_transform(df.Embarked)
# Missing Value Imputation for "Age" Parameter

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

df["Age"] = imputer.fit_transform(df["Age"].values.reshape(-1,1))

df = df.dropna(how = "all")
XX= df.iloc[:,1:].values

XX.shape
y = df.iloc[:,0].values

y.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

XX = scaler.fit_transform(XX)
XX[:5]
y[:12]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(XX, y, test_size=0.25, random_state=14)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



# KNN model



classifier = KNeighborsClassifier(n_neighbors=9, metric='minkowski', p=2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

predictions = pd.DataFrame(data=y_pred,    # values

                index=range(len(y_pred)),    # 1st column as index

                   columns=['y_pred'])  # 1st row as the column names



predictions['y_test'] = y_test

predictions.head()
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(cm)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: % {:10.2f}".format(accuracy*100)) 
accuracy_list = []

k_list = []



for k in range(2,10):

    classifier = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_list.append(accuracy)

    k_list.append(k)

    print("{} için accuracy: {:10.2f}".format(k,accuracy))
test = pd.read_csv("../input/test.csv", delimiter = ",")
test.head(2)
test.drop(["PassengerId", "Name","Ticket", "Cabin"], axis = 1, inplace = True)

test.info()
imputer = SimpleImputer(strategy="median")

test["Age"] = imputer.fit_transform(test["Age"].values.reshape(-1,1))

test = test.dropna(how = "all")





imputer = SimpleImputer(strategy="median")

test["Fare"] = imputer.fit_transform(test["Fare"].values.reshape(-1,1))

test = test.dropna(how = "all")
labelencoder_Sex = LabelEncoder()

test.Sex = labelencoder_Sex.fit_transform(test.Sex)





test.Embarked = df["Embarked"].astype("str")



labelencoder_Embarked = LabelEncoder()

test.Embarked = labelencoder_Embarked.fit_transform(test.Embarked)
XY_test= test.values

scaler = StandardScaler()

XY_test = scaler.fit_transform(XY_test)

XY_test[:10]
classifier.predict(XY_test)
# test = pd.read_csv("../input/test.csv", delimiter = ",")

# test.drop(["Name","Ticket", "Cabin", "Pclass","Sex","Age","SibSp","Parch","Ticket","Fare","Embarked"], axis = 1, inplace = True)

# test["Survived"] = classifier.predict(XY_test)

# test.to_csv("sonuc.csv", index = False, header = True)


