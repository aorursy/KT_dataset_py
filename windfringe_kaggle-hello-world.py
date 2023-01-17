import pandas as pd

from pandas import Series,DataFrame



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
########################

# 数据处理 & 特征工程 #

########################

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.info()

print("----------------------------")

test_df.info()



# 剔除无用的字段

train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

test_df.drop(['Name', 'Ticket'], axis=1, inplace=True)



# Embarked



# 填充缺Embarked的数据

# train_df['Embarked'] = train_df['Embarked'].fillna('S')



# Embarked（登船点）和生还的关系

# sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)

# sns.plt.show()



train_df.drop(['Embarked'], axis=1, inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)



# Fare

# test样本填充

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# float to int

train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)



# Age

# 找出train中的mean，std，# of null

mean_age_train = train_df['Age'].mean()

std_age_train = train_df['Age'].std()

null_age_train = train_df['Age'].isnull().sum()



# 找出test中的mean，std，# of null

mean_age_test = test_df['Age'].mean()

std_age_test = test_df['Age'].std()

null_age_test = test_df['Age'].isnull().sum()



# 生成随机age

def random_age_array(mean, std, num):

    return np.random.randint(mean - std, mean + std, size=num)



rand1 = random_age_array(mean_age_train, std_age_train, null_age_train)

rand2 = random_age_array(mean_age_test, std_age_test, null_age_test)



# 找出所有为null的age并赋予随机值

# train_df['Age'].dropna().astype(int)

train_df['Age'][np.isnan(train_df['Age'])] = rand1

test_df['Age'][np.isnan(test_df['Age'])] = rand2

# 转为int

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



# print(train_df['Age'], test_df['Age'])



# Cabin a lot of null, so useless, drop it

train_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)



# Family

def combineSibSpAndParch(df) :

    df['Family'] = df['SibSp'] + df['Parch']

    df['Family'].loc[df['Family'] > 0] = 1

    df['Family'].loc[df['Family'] == 0] = 0

    df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    return df



train_df = combineSibSpAndParch(train_df)

test_df = combineSibSpAndParch(test_df)



#Sex

def getPerson(passenger) :

    age, sex = passenger

    return 'child' if age < 16 else sex



train_df['Person'] = train_df[['Age', 'Sex']].apply(getPerson, axis=1)

test_df['Person'] = test_df[['Age', 'Sex']].apply(getPerson, axis=1)



train_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



person_dummies_train  = pd.get_dummies(train_df['Person'])

person_dummies_train.columns = ['Child','Female','Male']

person_dummies_train.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train_df = train_df.join(person_dummies_train)

test_df = test_df.join(person_dummies_test)



train_df.drop(['Person'], axis=1, inplace=True)

test_df.drop(['Person'], axis=1, inplace=True)



# Pclass

pclass_dummies_train  = pd.get_dummies(train_df['Pclass'])

pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



train_df = train_df.join(pclass_dummies_train)

test_df  = test_df.join(pclass_dummies_test)
########################

# 建立数据集

########################



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df['Survived']



X_test = test_df.drop('PassengerId', axis=1).copy()

# Y_test = pd.read_csv('../input/gender_submission.csv')

# Y_test.drop('PassengerId', axis=1, inplace=1)
########################

# 各种算法预测 #

########################



def predicAndScore(clf) :

    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    score = clf.score(X_train, Y_train)

    # predict_score = accuracy_score(Y_test, Y_pred)

    # print(predict_score)

    return score



# Gaussian Native Bayes

nb = GaussianNB()

score = predicAndScore(nb)

print('GaussianNB score:', score)



# SVM

svc = SVC()

score = predicAndScore(svc)

print('Support Vector Machine score:', score)



# Random Forests

rf = RandomForestClassifier(n_estimators=100)

score = predicAndScore(rf)

print('Random Forests:', score)



# Logistic Regression

logreg = LogisticRegression()

score = predicAndScore(logreg)

print('Logistic Regression:', score)



# knn

knn = KNeighborsClassifier(n_neighbors=3)

score = predicAndScore(knn)

print('knn:', score)



# decision tree

dt = tree.DecisionTreeClassifier()

score = predicAndScore(dt)

print('decision tree:', score)

# score: 0.957351290685



# Adaboost

parameters = { 'n_estimators': [10],

               'base_estimator': [DecisionTreeClassifier(max_depth=1, min_samples_split=3),

                                  DecisionTreeClassifier(max_depth=1, min_samples_split=10)],

               'learning_rate':[1.5, 10]}

clf = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=parameters)

clf.fit(X_train, Y_train)

# Y_pred = clf.predict(X_test)

# predict_score = accuracy_score(Y_test, Y_pred)

# print(predict_score)

print('Adaboost best params:', clf.best_params_)

print('Adaboost:', clf.score(X_train, Y_train))