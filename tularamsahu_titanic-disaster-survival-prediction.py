%matplotlib inline
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
titanic_train = pd.read_csv("../input/titanic/train.csv")
titanic_test = pd.read_csv("../input/titanic/test.csv")
titanic_gen = pd.read_csv("../input/titanic/gender_submission.csv")

df1 = pd.concat([titanic_test, titanic_gen['Survived']],  axis=1)
df2 = titanic_train
df = pd.concat([df1, df2])
df.info()
titanic_train.info()

print('****'*30)

titanic_test.info()
titanic_train.head()
titanic_test.head()
print(titanic_train.shape)
print(titanic_test.shape)
titanic_train.info()
titanic_test.info()
titanic_train.describe()
titanic_test.describe()
# function to identify missing value from all features

def missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
    return missing_data

print(' # Missing values from titanic_train dataset')

missing_data_train = missing_data(titanic_train)
print(missing_data_train)



print(' # Missing values from titanic_test dataset')

missing_data_test1 = missing_data(titanic_test)
print(missing_data_test1)
# function to drop missing value 
def drop_missing(dataset, missing, values):
    dataset = dataset.drop((missing[missing['Percent'] > values]).index, axis = 1)
    print(dataset.isnull().sum().sort_values(ascending = False))
    return dataset

titanic_train = drop_missing(titanic_train, missing_data_train, 0.60)
titanic_test = drop_missing(titanic_test, missing_data_test1, 0.60)
# fuction to identify numeric feature
def numeric_features(dataset):
    numeric_col = dataset.select_dtypes(include=np.number).columns.tolist()
    return dataset[numeric_col].head()

numeric_col= numeric_features(titanic_train)
print("Numeric features for train:")
print(numeric_col)
numeric_col1= numeric_features(titanic_test)
print("Numeric features for test:")
print(numeric_col1)
# fuction to identify categorial feature
def categorial_features(dataset):
    categorial_col = dataset.select_dtypes(exclude=np.number).columns.tolist()
    return dataset[categorial_col].head()


categorial_col = categorial_features(titanic_train)
print("categorial features for train:")
print(categorial_col)
categorial_col1 = categorial_features(titanic_test)
print("categorial features for test:")
print(categorial_col1)
def check_datatypes(dataset):
    return dataset.dtypes

check_datatypes(titanic_train)
check_datatypes(titanic_test)
# to impute null value
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].mean() )
titanic_test['Age'] = titanic_train['Age'].fillna(titanic_test['Age'].mean() )
titanic_test['Fare'] = titanic_train['Fare'].fillna(titanic_test['Fare'].mean() )
titanic_train['Embarked'].value_counts(normalize = True)
titanic_train['Embarked'].fillna('s', inplace= True)
# find the relationship on titanic_train dataset
plt.figure(figsize=(15,6))
sns.heatmap(titanic_train.corr(), annot=True)
sns.pairplot(titanic_train)
titanic_train.drop(["Name","Ticket"], axis=1, inplace=True)
titanic_test.drop(["Name",  "Ticket"], axis=1, inplace=True)
from sklearn import preprocessing 
lb = preprocessing.LabelEncoder()
titanic_train['Embarked'] = lb.fit_transform(titanic_train['Embarked'])
titanic_train['Sex'] = lb.fit_transform(titanic_train['Sex'])
titanic_train.head()
titanic_test['Embarked'] = lb.fit_transform(titanic_test['Embarked'])
titanic_test['Sex'] = lb.fit_transform(titanic_test['Sex'])
titanic_test.head()
df1 = pd.concat([titanic_test, titanic_gen['Survived']],  axis=1)
df2 = titanic_train
df = pd.concat([df1, df2])
df.info()
df['Age'].plot.hist()
sns.countplot('Sex', data=df).set_title('male_female_count')

sns.countplot('SibSp', data=df).set_title('SibSp')
sns.countplot('Pclass', data=df).set_title('Pclass')
sns.countplot('Parch', data=df).set_title('Parch')
sns.countplot('Embarked', data=df).set_title('Embarked')
sns.countplot('Survived', data=df).set_title('survived_count')
sns.jointplot(x='Age',y='Survived',data=df,color='red',kind='kde');
titanic_train = pd.DataFrame(titanic_train)
titanic_test = pd.DataFrame(titanic_test)

# dependent and independent variable
x_titanic_train = titanic_train.drop(['PassengerId', 'Survived'], axis = 1)
y_titanic_train = titanic_train['Survived']

x_titanic_test = titanic_test.drop(['PassengerId'], axis=1)
y_test = titanic_gen.drop(['PassengerId'], axis =1)




x_titanic_train.shape, y_titanic_train.shape, x_titanic_test.shape, y_test.shape
classifier = LogisticRegression(solver = 'liblinear', random_state=0)
classifier.fit(x_titanic_train,y_titanic_train)
y_pred = classifier.predict(x_titanic_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
lr = classifier
rf = RandomForestClassifier()
sv = SVC()

vc = VotingClassifier(estimators=[('lr', classifier), ('rf' ,rf), ('sv', sv), ('dt', dt)], voting='hard')
vc.fit(x_titanic_train,y_titanic_train)
# here we calculate accuracy rate of our model
for clf in ( lr,rf, sv, dt, vc):
    clf.fit(x_titanic_train,y_titanic_train)
    y_pred = clf.predict(x_titanic_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_pred, y_test))
print('MSE:', metrics.mean_squared_error(y_pred, y_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_pred, y_test)))
confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
ids = titanic_gen[['PassengerId']]
ids2 = pd.DataFrame({ 'Survived' : y_pred})
ids['Survived'] = ids2
submission_file = ids

submission_file.to_csv('./submission_file.csv' , index=False)
