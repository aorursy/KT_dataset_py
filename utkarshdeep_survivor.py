import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
path = '../input/'
pd.options.display.max_columns = None
df_train = pd.read_csv(path + 'train.csv') 
df_train.columns
df_train.head()
df_train = df_train.drop(['Name', 'Ticket'], axis = 1)
df_train['Survived'].astype('category').describe()
#histogram
sns.countplot(x="Survived", data=df_train)
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.barplot(x="Survived", y = "Survived", data=df_train,  estimator=lambda x: len(x) / len(df_train) * 100, hue = "Pclass")
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.stripplot(x = "Survived", y = "Fare", data = df_train)
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.swarmplot(x = "Survived", y = "Fare", data = df_train)
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.boxplot(x="Survived", y="Fare", hue="Pclass", data=df_train)
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.barplot(x="Survived", y = "Survived", data=df_train,  estimator=lambda x: len(x) / len(df_train) * 100, hue = "Sex")
plt.show()
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.set(font_scale=1.25)
sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, linewidths=.9)
plt.show()
sns.set()
cols = ['Pclass', 'Fare', 'Sex', 'Survived']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
missing_df = df_train.copy()
missing_df['Age'] = np.where(missing_df['Age'].isnull(), "Missing", "There" )
missing_df['Cabin'] = np.where(missing_df['Cabin'].isnull(), "Missing", "There" )
print(df_train['Age'].head(10))
print(missing_df['Age'].head(10))
f, ax = plt.subplots(figsize=(8, 10))
sns.barplot(x="Pclass", y = "Pclass", data=missing_df,  estimator=lambda x: len(x) / len(missing_df) * 100, hue = "Age")
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.barplot(x="Pclass", y = "Pclass", data=missing_df,  estimator=lambda x: len(x) / len(missing_df) * 100, hue = "Cabin")
plt.show()
f, ax = plt.subplots(figsize=(8, 10))
sns.swarmplot(x = "Survived", y = "Fare", data = df_train)
plt.show()
df_train['Sex'] = np.where(df_train['Sex'] == 'male', 1.0, 0.0 )
df_train['Embarked'] = df_train['Embarked'].map(lambda x : 1.0 if x == 'C' else x)
df_train['Embarked'] = df_train['Embarked'].map(lambda x : 2.0 if x == 'Q' else x)
df_train['Embarked'] = df_train['Embarked'].map(lambda x : 3.0 if x == 'S' else x)
df_train['Embarked'] = df_train['Embarked'].fillna(3.0)
df_train['Relatives'] = df_train['Parch'] + df_train['SibSp']
df_train_filtered = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Relatives', 'Embarked']]
X_train, X_test, y_train, y_test = train_test_split(df_train_filtered[['Pclass', 'Sex', 'Fare', 'Relatives', 'Embarked']], df_train_filtered['Survived'], random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression(solver = 'lbfgs')
model = model.fit(X_train, y_train)
model.score(X_train, y_train)
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train, y_train)
rfc.score(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy is %f" % (accuracy_score(y_test,y_pred)*100))
df_test = pd.read_csv(path + 'test.csv')
#missing data
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
#print(df_test['Fare'])
df_test['Sex'] = np.where(df_test['Sex'] == 'male', 1.0, 0.0 )
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].mean())
df_test['Relatives'] = df_test['Parch'] + df_test['SibSp']
df_test['Embarked'] = df_test['Embarked'].map(lambda x : 1.0 if x == 'C' else x)
df_test['Embarked'] = df_test['Embarked'].map(lambda x : 2.0 if x == 'Q' else x)
df_test['Embarked'] = df_test['Embarked'].map(lambda x : 3.0 if x == 'S' else x)
x = df_test[['Pclass', 'Sex', 'Fare', 'Relatives', 'Embarked']]
y_pred = pd.DataFrame(clf.predict(x))
y_pred.to_csv('ans.csv', sep='\t')
y_pred = pd.DataFrame(model.predict(x))
y_pred.to_csv('ans.csv', sep='\t')
y_pred = pd.DataFrame(rfc.predict(x))
y_pred.to_csv('ans.csv', sep='\t')