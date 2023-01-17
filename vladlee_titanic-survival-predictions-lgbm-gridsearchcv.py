# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sklearn.metrics  as sklm

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.neighbors import LocalOutlierFactor



import lightgbm as lgbm
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset = dataset.fillna(np.nan)



dataset.head(15)
g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")
def set_age(row):

    age = row.Age

    if np.isnan(age):

        if row.Sex == 'male':

            age = 28

        else:

            age = 27

    return age



dataset.Age = dataset.apply(lambda row: set_age(row), axis=1)

dataset.head(15)
# parse names

known_titles = ['Miss.','Mrs.','Mr.','Master.']

def build_title(name):

    title = name.find('Miss.')

    if title >=0:

        return 1

    title = name.find('Mrs.')

    if title >=0:

        return 2

    title = name.find('Mr.')

    if title >=0:

        return 3

    title = name.find('Master.')

    if title >=0:

        return 4

    return 0



dataset['Title'] = dataset.Name.apply(build_title)

dataset.head()
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset.drop(['Name','SibSp','Parch','Ticket'], axis=1, inplace=True)  

dataset.head()
fare_midean = dataset['Fare'].median()

values = {'Cabin':'Z','Embarked':'S','Fare':fare_midean}

dataset.fillna(value=values,inplace=True)

dataset['Cabin1'] = dataset['Cabin'].str[0]    

dataset.drop(['Cabin'], axis=1, inplace=True)    

    

#encode labels

dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'].values)

dataset['Cabin1'] = LabelEncoder().fit_transform(dataset['Cabin1'].values)

dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'].values)



dataset.head()
working_columns = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','Cabin1','Title']



scaler = StandardScaler()

dataset[working_columns] = scaler.fit_transform(dataset[working_columns])

dataset.head(10)
train_set = dataset.iloc[:train.shape[0]]

test_set = dataset.iloc[train.shape[0]:]



train_set.tail()

test_set.head()
columns = [c for c in dataset.columns if c not in ["Survived", "PassengerId", "Name", "Sex", "Cabin1", "Embarked"]]



# fit the model for outlier detection (default)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# use fit_predict to compute the predicted labels of the training samples

# (when LOF is used for outlier detection, the estimator has no predict,

# decision_function and score_samples methods).

y_pred = clf.fit_predict(train_set[columns].values)

X_scores = clf.negative_outlier_factor_



clf_df = pd.DataFrame( columns=['X_scores', 'y_pred'])

clf_df.X_scores = X_scores

clf_df.y_pred = y_pred

outlier_index = clf_df.index[clf_df['y_pred'] == -1].tolist()



train_set.drop(outlier_index)
g = sns.heatmap(dataset[working_columns].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
from sklearn.model_selection import GridSearchCV





cv_params = {

    'num_leaves': [10, 15, 20],

    'min_data_in_leaf':[15, 17, 19], 

    'learning_rate': [0.03,0.04,0.05]

}



ind_params = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'verbose': 0

}



optimized_GBM = GridSearchCV(lgbm.LGBMClassifier(**ind_params), cv_params, scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_GBM.fit(X_train, y_train)
print( 'best_score' % optimized_GBM.best_score_)

print( 'best_params' % optimized_GBM.best_params_)

X = train_set[working_columns].values

y = train_set['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



lgb_train_set = lgbm.Dataset(X_train, label=y_train)

lgb_test_set = lgbm.Dataset(X_test, label=y_test)



## {'learning_rate': 0.03, 'min_data_in_leaf': 17, 'num_leaves': 15}



#setting parameters for lightgbm

parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 15,

    'min_data_in_leaf':17, 

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.03,

    'verbose': 0

}





#training our model using light gbm

model = lgbm.train(parameters, lgb_train_set, valid_sets=lgb_test_set, num_boost_round=5000, early_stopping_rounds=100)

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score



y_pred = model.predict(X_test)



fpr, tpr, _ = roc_curve(y_test,  y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
X_sub = test_set[working_columns].values

predictions = model.predict(X_sub)



submission = pd.DataFrame({

        "PassengerId": test_set["PassengerId"],

        "Survived": predictions.ravel() 

    })



submission.Survived = submission.Survived.apply(lambda x : 1 if x > 0.49 else 0)

submission.to_csv('submission.csv', index=False)