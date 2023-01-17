import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import make_scorer, roc_auc_score   
from time import time
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier,BaggingClassifier

%matplotlib inline

import warnings  
warnings.filterwarnings("ignore") 
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
fullData = [trainData, testData]
originTrainData = trainData.copy()
originTestData = testData.copy()
trainData.head()
trainData.info()
print('='*40)
testData.info()
trainData.describe()
trainData.describe(include='object')
# Pclass, Sex and Embarked
cols = ['Pclass', 'Sex', 'Embarked']

for col in cols:
    curColumn =  trainData[col]
    Survived_0 = curColumn[trainData['Survived'] == 0].value_counts()
    Survived_1 = curColumn[trainData['Survived'] == 1].value_counts()
    df = pd.DataFrame({"Survived=1":Survived_1, "Survived=0":Survived_0})
    ax = df.plot(kind='bar', title='Survived or Not')
    ax.set_xlabel(col)
    ax.set_ylabel('people')
# SibSp and Parch
survived1 = trainData[['SibSp','Survived']].groupby('SibSp').sum()
survived1.plot(y='Survived', kind='pie', title='SibSp distribution among survived people')

survived2 = trainData[['Parch','Survived']].groupby('Parch').sum()
survived2.plot(y='Survived', kind='pie', title='Parch distribution among survived people')
# Age and Fare
cols = {'Age':[0,85], 'Fare':[0,515]}

for col in cols:
    curColumn =  trainData[col]
    Survived_0 = curColumn[trainData['Survived'] == 0]
    Survived_1 = curColumn[trainData['Survived'] == 1]
    df = pd.DataFrame({"Survived=1":Survived_1, "Survived=0":Survived_0})
    ax = df.plot(kind='kde',title='Survived or Not', xlim=cols[col])
    ax.set_xlabel(col)
# Ticket
trainData[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sample(20)
# Cabin
print(trainData.isnull().sum())
print("="*40)
print(trainData['Cabin'].value_counts())
# Name
trainData['Name'].head(20)
drop_column = ['Ticket']
trainData.drop(drop_column, axis=1, inplace = True)
testData.drop(drop_column, axis=1, inplace = True)
trainData.head()
for data in fullData:
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    data['Fare'].fillna(data['Fare'].median(), inplace = True)
    data['Cabin'].fillna('N', inplace = True)
    
print(trainData.isnull().sum())
print('='*40)
print(testData.isnull().sum())
# isAlone, familySize, hasCabin, Title
for data in fullData:
    data['familySize'] = data['SibSp'] + data['Parch'] + 1
    data['isAlone'] = 1
    data['isAlone'].loc[data['familySize'] > 1] = 0
    data["hasCabin"] = 1
    data['hasCabin'].loc[data['Cabin'] == 'N'] = 0
    data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

trainData.head()
print(trainData['Title'].value_counts())
print('='*40)
print(testData['Title'].value_counts())
miniCount = 10

for data in fullData:
    title_names = (data['Title'].value_counts() < miniCount)
    data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(trainData['Title'].value_counts())
print('='*40)
print(testData['Title'].value_counts())
cols = ['isAlone', 'familySize', 'hasCabin', 'Title']

for col in cols:
    curColumn =  trainData[col]
    Survived_0 = curColumn[trainData['Survived'] == 0].value_counts()
    Survived_1 = curColumn[trainData['Survived'] == 1].value_counts()
    df = pd.DataFrame({"Survived=1":Survived_1, "Survived=0":Survived_0})
    ax = df.plot(kind='bar', title='Survived or Not')
    ax.set_xlabel(col)
    ax.set_ylabel('people')
# Age, Fare, Sex, Embarked, Title
for data in fullData:
    data['AgeBins'] = pd.cut(data['Age'], 5).cat.codes
    data['FareBins'] = pd.qcut(data['Fare'], 5).cat.codes

trainDummy = pd.get_dummies(trainData, columns=["Sex", "Embarked", "Title"], prefix=["Sex", "Embarked",  "Title"])
testDummy = pd.get_dummies(testData, columns=["Sex", "Embarked", "Title"], prefix=["Sex", "Embarked",  "Title"])

trainDummy.head(1)
print(originTrainData.columns)
print('='*40)
print(trainData.columns)
print('='*40)
print(trainDummy.columns)
featureLabels = ['Pclass', 'SibSp', 'Parch','familySize', 'isAlone', 'hasCabin', 'AgeBins', 'FareBins', \
                 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master',\
                'Title_Misc', 'Title_Miss', 'Title_Mr', 'Title_Mrs']
targetLabels = ['Survived']
train_x = trainDummy[featureLabels]
train_y = trainDummy[targetLabels]
test_x = testDummy[featureLabels]
# Initializing some machine learning algorithms
lr = LogisticRegressionCV()
knn = KNeighborsClassifier(n_neighbors = 3)
gb = GaussianNB()
svc = SVC()
nusvc = NuSVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
gbdt = GradientBoostingClassifier(n_estimators=100)
xgb = XGBClassifier(max_depth=3, n_estimators=100)

mlas = [lr, knn,gb, svc, nusvc, dt, rf, gbdt, xgb]
# Evaluating algorithms' performance by cross-validation
kfold = 10
scores = []
names = []
roc_auc_scorer=make_scorer(roc_auc_score)

for mla in mlas:
    names.append(mla.__class__.__name__)
    scores.append(cross_val_score(mla, train_x, y=train_y, scoring = roc_auc_scorer, cv = kfold, n_jobs=4).mean())
# Plotting
scoreDf = pd.DataFrame({'names':names, 'scores':scores}).sort_values(by='scores', ascending=True)
print(scoreDf)
ax = scoreDf.plot(x=['names'], y=['scores'], kind='barh', title='Basic Algorithm Performance', xlim=[0.5,1])
ax.set_ylabel('Algorithms')
ax.set_xlabel('Accuracy')
def fit_model(mla,parameters, X, y):  
    grid = GridSearchCV(mla,parameters,scoring=roc_auc_scorer,cv=10)
    start=time()
    grid=grid.fit(X,y)
    end=time()  
    t=round(end-start,3)  
    print('best parameters: %s' % grid.best_params_)
    print('best score: %s' % grid.best_score_)
    print('searching time for {} is {} s'.format(mla.__class__.__name__,t))
    return grid 
# 1 SVM
parameterSVC = {"C":range(1,20),"gamma": [0.05,0.1,0.15,0.2,0.25]}
classifierSVC = fit_model(svc, parameterSVC, train_x, train_y.values.ravel())  
# 2 XGBoost
parameterXGB1 = {'n_estimators':range(10,200,10)}  
parameterXGB2 = {'max_depth':range(1,10),'min_child_weight':range(1,10)}  
parameterXGB3 = {'subsample':[i/10.0 for i in range(1,10)], 'colsample_bytree':[i/10.0 for i in range(1,10)]}
classifierXGB = fit_model(xgb, parameterXGB1, train_x, train_y.values.ravel())  
xgb = XGBClassifier(n_estimators=10)
classifierXGB = fit_model(xgb, parameterXGB2, train_x, train_y.values.ravel())  
xgb = XGBClassifier(n_estimators=10, max_depth=3, min_child_weight=3)
classifierXGB = fit_model(xgb, parameterXGB3, train_x, train_y.values.ravel())
# 3 Random Forest
parameterRF1 = {'n_estimators':range(10,200,10)}  
parameterRF2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}
classifierRF = fit_model(rf, parameterRF1, train_x, train_y.values.ravel())  
rf = RandomForestClassifier(n_estimators=120)
classifierRF = fit_model(rf, parameterRF2, train_x, train_y.values.ravel())
# 4 KNN
parameterKNN = {'n_neighbors':range(2,10),'leaf_size':range(10,80,20)}
classifierKNN = fit_model(knn, parameterKNN, train_x, train_y.values.ravel())
# 5 LR and 6 GBDT can also be tuned, but I chose not to do too much adjustment here.
lr = lr.fit(train_x, train_y)
gb = gb.fit(train_x, train_y)
def save(classifier,i):  
    prediction = classifier.predict(test_x)  
    sub = pd.DataFrame({ 'PassengerId': originTestData['PassengerId'], 'Survived': prediction })  
    sub.to_csv("prediction_{}.csv".format(i), index=False) 
    
i = 0
classifiers = [classifierSVC, classifierXGB, classifierRF, lr, gb]
for classifier in classifiers:
    save(classifier, i)
    i += 1

print("done!")
kaggleScores = [0.79425, 0.78947, 0.78947, 0.76076, 0.77033]
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
X, y = train_x, train_y

clas = {'svc':classifierSVC, 'xgb':classifierXGB, 'rf':classifierRF, 'lr':lr, 'gb':gb, 'knn':classifierKNN}
for claName in clas:
    title = "Learning Curves -" + claName 
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    estimator = clas[claName]
    plot_learning_curve(estimator, title, X, y, (0.5, 1.01), cv=cv, n_jobs=4)
    plt.show()
ntrain = train_x.shape[0]
ntest = test_x.shape[0]
SEED = 0 
NFOLDS = 5 
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)


def get_out_fold(clf, x_train, y_train, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0]))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


x_train = train_x.values 
x_test = test_x.values 
y_train = train_y.values
# First level classifiers
basicSVC = SVC(C=7,gamma=0.05)
basicRF = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_split=7)
basicLR = LogisticRegressionCV()
basicGB = GaussianNB()
basicKNN = KNeighborsClassifier(leaf_size=30, n_neighbors=7)
basicXGB = XGBClassifier( n_estimators= 30, max_depth= 3, min_child_weight= 3,  subsample=0.9, \
                         colsample_bytree=0.7)
clfs = [basicSVC, basicRF, basicLR, basicGB, basicKNN]

trainPredictions = np.zeros((ntrain, len(clfs)))
testPredictions = np.zeros((ntest, len(clfs)))
for i,clf in enumerate(clfs):
    trainPredictions[:,i], testPredictions[:,i] = get_out_fold(clf, x_train, y_train, x_test)
# Second level classifier-XBGoost
stackXGB = basicXGB.fit(trainPredictions, y_train)
predictions = stackXGB.predict(testPredictions)
title = "Learning Curves - stackXGB"
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
plot_learning_curve(stackXGB, title, X, y, (0.5, 1.01), cv=cv, n_jobs=4)
plt.show()

vot = VotingClassifier(estimators=[('svc', basicSVC), ('rf', basicRF), ('lr', basicLR), ( 'gb', basicGB), ( 'knn', basicKNN), ('xgb', basicXGB)])
vot = vot.fit(X, y)
predictions = vot.predict(test_x)
#Submission
sub = pd.DataFrame({ 'PassengerId': originTestData['PassengerId'], 'Survived': predictions })  
sub.to_csv("prediction_voting.csv", index=False) 
title = "Learning Curves - voting"
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
plot_learning_curve(vot, title, X, y, (0.5, 1.01), cv=cv, n_jobs=4)
plt.show()
bag = BaggingClassifier(base_estimator=basicSVC, n_estimators=20, max_samples=0.8 )
bag = bag.fit(X, y)
predictions = bag.predict(test_x)
#Submission
sub = pd.DataFrame({ 'PassengerId': originTestData['PassengerId'], 'Survived': predictions })  
sub.to_csv("prediction_bagging.csv", index=False) 
title = "Learning Curves - bagging"
cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
plot_learning_curve(bag, title, X, y, (0.5, 1.01), cv=cv, n_jobs=4)
plt.show()