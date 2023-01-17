import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import manifold
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.describe()
train_df.head(5)
test_df.head(5)
train_df.isnull().sum()
test_df.isnull().sum()
dataset = [train_df,test_df]

for data in dataset:
    cabin_map = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8}
    data['Deck'] = data.Cabin.str.extract('([A-Za-z]+)',expand=False)
    data['Deck'] = data['Deck'].map(cabin_map)
    data['Deck'] = data.Deck.fillna(0)
    data['Deck'] = data.Deck.astype(int)
train_df = train_df.drop(['Cabin'],axis=1)
test_df = test_df.drop(['Cabin'],axis=1)
dataset = [train_df,test_df]

for data in dataset:
    data['Title'] =  data.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
    #train_data['Title']
    title_map = {"Mr":1,"Mrs":2,"Miss":3,"Ms":4,"Rare":5}
    data['Title'] = data['Title'].replace(['Master','Don','Rev','Dr','Major','Lady','Sir',
                           'Col','Capt','Countess','Jonkheer'],'Rare')
    data['Title'] = data['Title'].replace(['Mme'],'Mrs')
    data['Title'] = data['Title'].replace(['Mlle'],'Miss')
    data['Title'] = data['Title'].map(title_map)
    data['Title'] = data['Title'].fillna(0)
    data['Title'] = data.Title.astype(int)
train_df = train_df.drop(['Name'],axis=1)
test_df = test_df.drop(['Name'],axis=1)
ports = {"S":1,"C":2,"Q":3}
dataset = [train_df,test_df]

for data in dataset:
    data['Embarked'] = data['Embarked'].map(ports)
    data['Embarked'] = data.Embarked.fillna(0)
    data['Embarked'] = data.Embarked.astype(int)
gender_map = {"male":1 , "female":2}
dataset = [train_df,test_df]

for data in dataset:
    data['Sex'] = data['Sex'].map(gender_map)

dataset = [train_df,test_df]

for data in dataset:
    mean = data['Age'].mean()
    std = data['Age'].std()
    is_null = data['Age'].isnull().sum()
    rand_age = np.random.randint(mean-std,mean+std,size=is_null)
    #rand_age
    age_slice = data['Age'].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data['Age'] = age_slice
    data['Age'] = data.Age.astype(int)
    #train_data['Age'] = train_data.Age.astype(int)
dataset = [train_df,test_df]

for data in dataset:
    data.loc[ data['Age'] <= 11, 'Age'] = 0
    data.loc[ (data['Age'] >11) & (data['Age']<=18),'Age'] = 1
    data.loc[ (data['Age'] >18) & (data['Age']<=22),'Age'] = 2
    data.loc[ (data['Age'] >22) & (data['Age']<=27),'Age'] = 3
    data.loc[ (data['Age'] >27) & (data['Age']<=33),'Age'] = 4
    data.loc[ (data['Age'] >33) & (data['Age']<=40),'Age'] = 5
    data.loc[ (data['Age'] >40) & (data['Age']<=66),'Age'] = 6
    data.loc[ data['Age'] >66 ,'Age'] = 7
dataset = [train_df,test_df]

for data in dataset:
    data.loc[ data['Fare'] <= 7.91 , 'Fare'] = 0
    data.loc[ (data['Fare'] > 7.91) & (data['Fare'] <= 14.454) , 'Fare'] = 1
    data.loc[ (data['Fare'] > 14.454) & (data['Fare'] <= 31.00) , 'Fare'] = 2
    data.loc[ (data['Fare'] > 31.00) & (data['Fare'] <= 100) , 'Fare'] = 3
    data.loc[ (data['Fare'] > 100) & (data['Fare'] <= 250) , 'Fare'] = 4
    data.loc[ data['Fare'] > 250 , 'Fare'] = 5
    data['Fare'] = data['Fare'].fillna(0)
    data['Fare'] = data.Fare.astype(int)
train_df = train_df.drop(['Ticket'],axis=1)
test_df = test_df.drop(['Ticket'],axis=1)
train_df = train_df.drop(['PassengerId'],axis=1)
test_df = test_df.drop(['PassengerId'],axis=1)
dataset = [train_df,test_df]

for data in dataset:
    data['relatives'] = data['SibSp']+data['Parch']
    data.loc[data['relatives']>0,'not_alone'] = 0
    data.loc[data['relatives'] == 0,'not_alone'] = 1
    data['not_alone'] = data['not_alone'].astype(int)
train_df['Age'].isnull().sum()
test_df['Age'].value_counts()
train_df['Fare'].value_counts()
train_df.head(5)
test_df.head(5)
x_train = train_df.drop(['Survived'],axis=1)
y_train = train_df['Survived']
x_test = test_df
params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)
clf = RandomForestClassifier()
rand_clf_cv = GridSearchCV(estimator=clf,param_grid=params, cv=5) 
rand_clf_cv.fit(x_train,y_train)
y_pred = rand_clf_cv.predict(x_test)
rand_clf_cv.best_params_
rand_clf_cv.score(x_train,y_train)
dec_clf = DecisionTreeClassifier()
dec_clf.fit(x_train,y_train)
y_pred = dec_clf.predict(x_test)
dec_clf.score(x_train,y_train)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

knn.score(x_train,y_train)
pid = list(range(892,1310))
#pid
submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": y_pred
    })
submission.to_csv('submission9.csv', index=False)