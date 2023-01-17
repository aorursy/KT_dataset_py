import numpy as np
import pandas as pd
import matplotlib as plt
%matplotlib inline
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
train_csv="../input/train.csv"#"Z:\\db\\titanic\\train.csv"
test_csv="../input/test.csv"#"Z:\\db\\titanic\\test.csv"
prediction="../input/prediction.csv"#'Z:\\db\\titanic\\prediction.csv'


df=pd.read_csv(train_csv)
df_test=pd.read_csv(test_csv)
features = pd.DataFrame()
index=list(df.head(0))
print(index)
df.describe()

print((df['Age']).median())        # median = 28.0  # mean_age=29.69 
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Age'].hist(bins=50)               
print(df.head())
targets = df.Survived
df['Died'] = 1 - df['Survived']
df.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', figsize=(25, 7),stacked=True, colors=['g', 'r']);

df.boxplot("Fare","Survived")      #mixed fare better get embarkment
print(df['Embarked'])
#map(int,df['Embarked'])

#df.boxplot("Embarked","Survived")
#ax = plt.subplot()
#ax.set_ylabel('Average fare')
#data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(25, 7), ax = ax);
#age_count=df['Age'].value_counts(True)
#pivot=df.pivot_table('Survived',['Pclass'],aggfunc= lambda x: x.map({'Y':1,'N':0}).mean())
#pivot  #not working
#df['Age'].fillna(df['Age'].median(), inplace=True)
df.apply(lambda x: sum(x.isnull()),axis=0)
df.head()
print(df.head())
#targets = df.Survived   done above
df.drop(['Survived'], 1, inplace=True)
combined = df.append(df_test)
combined.reset_index(inplace=True)
combined.drop(['index', 'PassengerId','Name','Ticket','Died','Cabin'], inplace=True, axis=1)    #A10 #data without tag
print(combined.head())
combined.describe()           # remaining  Embarked  parch+Sibsp sex 
combined['Age'].fillna(combined['Age'].median(), inplace=True)  # should have done once
combined['Fare'].fillna(combined['Fare'].median(), inplace=True)
combined['Embarked'].fillna("S", inplace=True)
combined.apply(lambda x: sum(x.isnull()),axis=0)
embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')  # remaining  Embarked  parch+Sibsp sex 
combined = pd.concat([combined, embarked_dummies], axis=1)
combined.drop('Embarked', axis=1, inplace=True)

combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

combined.head()
targets = pd.read_csv(train_csv, usecols=['Survived'])['Survived'].values              #reset val in target
train = combined.iloc[:891]
test = combined.iloc[891:]
print(test.head())
print(test.describe())
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
              
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
              
features.plot(kind='barh', figsize=(25, 25))

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print( train_reduced.shape)

test_reduced = model.transform(test)
print (test_reduced.shape)
xval = cross_val_score(RandomForestClassifier(), X=train_reduced, y=targets, cv = 5, scoring='accuracy')
np.mean(xval)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
model = RandomForestClassifier(**parameters)
model.fit(train, targets)

output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv(test_csv)
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
#df_output[['PassengerId','Survived']].to_csv(prediction, index=False)