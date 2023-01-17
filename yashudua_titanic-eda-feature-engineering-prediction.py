
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
plt.figure(figsize=(14,6))
sns.countplot(x='Survived',hue='Sex',data=train)
plt.title('Number of people category by Sex and Survival');
plt.figure(figsize=(14,6))
sns.violinplot(x='Sex',y='Fare',hue='Survived',data=train)
plt.title('Number of people Fare Distribution category by Sex and Survival');
plt.figure(figsize=(14,6))
sns.countplot(x='Pclass',hue='Survived',data=train)
plt.title('Number of people category by Pclass and Survival');
plt.figure(figsize=(14,6))
sns.countplot(x='Embarked',hue='Survived',data=train)
plt.title('Number of people category by Embarked and Survival');
plt.figure(figsize=(14,6))
sns.distplot(a=train[train['Survived']==0]['Fare'])
sns.distplot(a=train[train['Survived']==1]['Fare'])
plt.title("Distribution of fares by survival");
plt.figure(figsize=(14,6))
sns.distplot(a=train[train['Survived']==0]['Age'])
sns.distplot(a=train[train['Survived']==1]['Age'])
plt.title("Distribution of Age by survival");
plt.figure(figsize=(14,6))
sns.heatmap(train.corr(),annot=True)
plt.title("Correlation Matrix");
def visualNA(df,perc=0):
    #Percentage of NAN Values 
    NAN = [(c, df[c].isna().mean()*100) for c in df]
    NAN = pd.DataFrame(NAN, columns=["column_name", "percentage"])
    NAN = NAN[NAN.percentage > perc]
    print(NAN.sort_values("percentage", ascending=False))
visualNA(train)
train['Embarked'].value_counts()
train['Embarked'].fillna(value='S',inplace=True)
train[['Ticket_Class','Ticket_Number']]=train['Ticket'].str.split(" ",expand=True,n=1)
train['Ticket_Class'],train['Ticket_Number'] = zip(*train[['Ticket_Class','Ticket_Number']].apply((lambda x : (None,x['Ticket_Class']) if x['Ticket_Number'] is None else (x['Ticket_Class'],x['Ticket_Number'])),axis=1))
train['Title'] = train['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
plt.figure(figsize=(16,6))
sns.countplot(x=train['Title'],hue=train['Survived'])
plt.xticks(rotation=45)
plt.title('Distribution of People by Title');
train['Cabin'].fillna(value='0',inplace=True)
train['Ticket_Class'].fillna(value='0',inplace=True)
les = LabelEncoder()
train['Sex'] = les.fit_transform(train['Sex'])


let = LabelEncoder()
train['Title'] = let.fit_transform(train['Title'].astype(str))

letn = LabelEncoder()
train['Ticket_Number'] = letn.fit_transform(train['Ticket_Number'].astype(str))


letc = LabelEncoder()
train['Ticket_Class'] = letc.fit_transform(train['Ticket_Class'].astype(str))


lec = LabelEncoder()
train['Cabin'] = lec.fit_transform(train['Cabin'].astype(str))


lee = LabelEncoder()
train['Embarked'] = lee.fit_transform(train['Embarked'].astype(str))

train['Cabin'].replace(0,np.NaN,inplace=True)
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
       'Fare', 'Cabin', 'Embarked', 'Ticket_Class','Ticket_Number','Title']]

y = train['Survived']
imputer = KNNImputer(n_neighbors=5)


X = imputer.fit_transform(X)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
model_bc = BaggingClassifier(RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),n_estimators=100)

model_bc.fit(X_res,y_res)
feature_importances = np.mean([
    tree.feature_importances_ for tree in model_bc.estimators_
], axis=0)
coeff_df = pd.DataFrame(feature_importances,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
       'Fare', 'Cabin', 'Embarked', 'Ticket_Class','Ticket_Number','Title'],columns=['Coefficient'])

fig, ax = plt.subplots(1,1,figsize=(12,8))
coeff_df.sort_values(by='Coefficient',ascending=True).plot(kind='barh',ax=ax)
plt.xlabel('Features')
plt.title('Top Features');
visualNA(test)
test[['Ticket_Class','Ticket_Number']]=test['Ticket'].str.split(" ",expand=True,n=1)
test['Ticket_Class'],test['Ticket_Number'] = zip(*test[['Ticket_Class','Ticket_Number']].apply((lambda x : (None,x['Ticket_Class']) if x['Ticket_Number'] is None else (x['Ticket_Class'],x['Ticket_Number'])),axis=1))
test['Title'] = test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
for i,lechange in [('Ticket_Class',letc),('Cabin',lec),('Embarked',lee),
                   ('Ticket_Number',letn),('Title',let)]:
    test[i] = test[i].map(lambda s: '<unknown>' if s not in lechange.classes_ else s)
    le_classes = lechange.classes_.tolist()
    le_classes.insert(len(le_classes), '<unknown>')
    lechange.classes_ = le_classes
test['Ticket_Class'].fillna(value='0',inplace=True)


test['Sex'] = les.transform(test['Sex'])
test['Ticket_Class'] = letc.transform(test['Ticket_Class'].astype(str))
test['Cabin'] = lec.transform(test['Cabin'].astype(str))
test['Embarked'] = lee.transform(test['Embarked'].astype(str))
test['Ticket_Number'] = letn.transform(test['Ticket_Number'].astype(str))
test['Title'] = let.transform(test['Title'].astype(str))

ID = test['PassengerId']

X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
       'Fare', 'Cabin', 'Embarked', 'Ticket_Class','Ticket_Number','Title']]
X_test = imputer.transform(X_test)
y_pred_bc = model_bc.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": ID,
        "Survived": y_pred_bc
    })
submission.to_csv('submission.csv', index=False)