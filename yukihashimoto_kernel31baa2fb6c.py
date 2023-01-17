import numpy as np

import pandas as pd

import sklearn 





import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
y = test

train.head()

test.info()
train.columns
test.columns
test.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].mean())

test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())

test.info()
sns.countplot('Embarked',data=train)

plt.title('Number of Passengers Boarded')

plt.show()
train["Embarked"] = train["Embarked"].fillna("S")
train.head()
train['Title'] = train['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])

train['Title'].value_counts()
test['Title'] = test['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])

test['Title'].value_counts()
y = test
train['Title'].replace(['Col','Dr', 'Rev','Capt','Don','Jonkheer','Lady','Mlle','Major','Mile','Mme','Sir','the Countess'], 'Rare',inplace=True)

train['Title'].replace('Mlle', 'Miss',inplace=True) 

train['Title'].replace('Ms', 'Miss',inplace=True) 
test['Title'].replace(['Col','Dr', 'Rev','Dona'], 'Rare',inplace=True)

test['Title'].replace('Mlle', 'Miss',inplace=True) 

test['Title'].replace('Ms', 'Miss',inplace=True) 
train["Ticket_v"] = train["Ticket"].map(lambda x: len(x.split()))

train["Ticket_v"].value_counts()



test["Ticket_v"] = test["Ticket"].map(lambda x: len(x.split()))

test["Ticket_v"].value_counts()

train = pd.get_dummies(train, columns=['Sex', 'Embarked','Title','Ticket_v'])

test = pd.get_dummies(test, columns=['Sex', 'Embarked','Title','Ticket_v'])

#test["Fare"]



y_train = train["Survived"]



train.drop(['PassengerId','Cabin','Ticket','Name','Survived'], axis=1, inplace=True)

test.drop(['PassengerId','Cabin','Ticket','Name'], axis=1, inplace=True)
X_train = train
#Sex, Age, Embarked

#from sklearn.model_selection import train_test_split

#from sklearn.metrics import accuracy_score



#train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

#train_drop.info()

#X_test.info()

#age_bins = [10,20,30,40,50,60,70,80,90]





#age_cut_data = pd.cut(X_train["age_cut_data"].age, age_bins)

#age_cut_data
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold



#train_x, valid_x, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.3, random_state=0)





features, test_features = train.align(test, join = 'inner', axis = 1)



features = np.array(features)

test_features = np.array(test_features)



k_fold = KFold(n_splits = 5, shuffle = True, random_state = 50)



for train_indices, valid_indices in k_fold.split(features):



    # Training data for the fold

    train_features, train_labels = features[train_indices], y_train[train_indices]

    # Validation data for the fold

    valid_features, valid_labels = features[valid_indices], y_train[valid_indices]

# #X_train = pd.get_dummies(X_train)

# #X_test["Embarked"]=test["Embarked"]

# import lightgbm as lgb

# gbm = lgb.LGBMClassifier(objective='binary',

#                         min_data_in_leaf=23,

                         

                        

#                         )



# print(gbm)





# gbm.fit(

#         train_x, train_y, 

#         eval_set = [(valid_x, valid_y)],

#         early_stopping_rounds=100, 

#         verbose=10

# ) ;





import lightgbm as lgb





dtrain = lgb.Dataset(train_features, label=train_labels)

dval = lgb.Dataset(valid_features, label=valid_labels)



params = {'objective': 'binary', 'lambda_l1': 6.200507325959984e-05, 'lambda_l2': 8.718800463884586, 'num_leaves': 31, 'feature_fraction': 0.4, 'bagging_fraction': 0.8951875481012406, 'bagging_freq': 7, 'min_child_samples': 10}



gbm = lgb.train(

    params, dtrain, valid_sets=[dtrain, dval]

)

#0.378299

#0.377291









# oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

# oof2 = gbm.predict(train_x, num_iteration=gbm.best_iteration_)



# print('score', round(accuracy_score(train_y, oof2)*100,2), '%')

# print('score', round(accuracy_score(valid_y, oof)*100,2), '%')

y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)





#y_pred = np.argmax(y_pred)



y_pred
from sklearn.metrics import roc_auc_score





#if>=0.5 ---> 1

#else ---->0

#rounding the values

y_pred=y_pred.round(0)

#converting from float to integer

y_pred=y_pred.astype(int)





#roc_auc_score metric

#roc_auc_score(y_pred,valid_labels)





y["Survived"] = y_pred



y = y[["PassengerId","Survived"]]

y.info()





y.to_csv("submission.csv", index=False, encoding='cp932')