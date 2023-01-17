# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

#from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
columns = ['bone_length','rotting_flesh','hair_length','has_soul']



sns.pairplot(train[columns] , diag_kind='kde',vars=columns,kind='scatter')
sns.pairplot(train,vars=columns,kind='scatter',diag_kind='kde',hue='type')
le = LabelEncoder()

y = le.fit_transform(train.type.values)

#y = train.type

test_ids = test.id

train.drop(['type','id'],axis=1,inplace=True)

test.drop(['id'],axis=1,inplace=True)

all_data = pd.concat((train,test),axis=0).reset_index()

all_data['hair_bone'] = all_data['hair_length'] * all_data['bone_length']

all_data['hair_rotting'] = all_data['hair_length'] * all_data['rotting_flesh']

all_data['hair_soul'] = all_data['hair_length'] * all_data['has_soul']

all_data['hair_soul_bone'] = all_data['hair_length'] * all_data['has_soul'] * all_data['bone_length']

all_data['hair_rotting_soul'] = all_data['hair_length'] * all_data['rotting_flesh'] * all_data['has_soul']

all_data['hair_bone_rotting'] = all_data['hair_length'] * all_data['bone_length'] *all_data['rotting_flesh']

all_data['rotting_hair_soul'] = all_data['rotting_flesh'] * all_data['hair_length'] * all_data['has_soul']

all_data['rotting_bone_soul'] = all_data['rotting_flesh'] * all_data['bone_length'] * all_data['has_soul']

all_data['bone_rotting_soul'] = all_data['bone_length'] * all_data['rotting_flesh'] * all_data['has_soul']

all_data['bone_rotting'] = all_data['bone_length'] * all_data['rotting_flesh']

all_data['bone_soul'] = all_data['bone_length'] * all_data['has_soul']

all_data['rotting_soul'] = all_data['rotting_flesh'] * all_data['has_soul']

#all_data['color'] = pd.factorize(all_data['color'],sort=True)[0]

all_data.drop(['index','color'],axis=1,inplace=True)
#train['color'] = pd.factorize(train['color'],sort=True)[0]

#test['color'] = pd.factorize(test['color'],sort=True)[0]
all_data.head()
columns = all_data.columns

n_trains = train.shape[0]

#all_data = pd.concat((train,test),axis=0).reset_index()

scaler = StandardScaler().fit(all_data)

x_train = scaler.transform(all_data.iloc[:n_trains])

x_test = scaler.transform(all_data.iloc[n_trains:])
clf = RandomForestClassifier(n_estimators=300)

clf.fit(x_train,y)

indices = np.argsort(clf.feature_importances_)[::-1] ## reverse





for idx in range(x_train.shape[1]):

    print('%d feature %d %s (%f)' %(idx + 1,

                                    indices[idx],

                                    columns[indices[idx]],

                                    clf.feature_importances_[indices[idx]]))



#sfs = StratifiedShuffleSplit(y,n_iter=10,test_size=0.2,random_state=42)

X_train,X_test,y_train,y_test = train_test_split(x_train,y,test_size=0.2,random_state=42)

#for train_idx,test_idx in sfs:

#    X_train,X_test = train.values[train_idx],train.values[test_idx]

#    y_train,y_test = y[train_idx],y[test_idx]





classifiers = [

    

    RandomForestClassifier(n_estimators=500),

    AdaBoostClassifier(random_state=42,n_estimators=500,learning_rate=0.01),

    DecisionTreeClassifier(),

    GradientBoostingClassifier(subsample=0.8,learning_rate=0.1,n_estimators=500),

    SVC(kernel='rbf',C=0.02,probability=True),

    KNeighborsClassifier(n_neighbors=3),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()

    

]

columns = ['Classifier','Accuracy','Log Loss']

result = pd.DataFrame(columns = columns)

for classifier in classifiers:

    print('-'*50)

    print(classifier.__class__.__name__)

    name = classifier.__class__.__name__

    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    print('Accuracy : %0.2f' %(accuracy))

    print(classification_report(y_pred,y_test))

    y_pred = classifier.predict_proba(X_test)

    log = log_loss(y_test,y_pred)

    print('Log Loss : %0.2f' %(log))

    entry = pd.DataFrame([[name,accuracy,log]],columns = columns)

    result = result.append(entry)

print('-'*50)
clf = RandomForestClassifier()

param = {'n_estimators':[150,250,350],

         'criterion' : ['gini', 'entropy'],

         'max_features' : ['auto', 'sqrt', 'log2', None]

         }

optimzation = GridSearchCV(clf,param_grid = param , cv=5 , scoring='accuracy')

optimzation.fit(x_train,y)

print('Best score : ' , optimzation.best_score_)

print('Best parameter : ' , optimzation.best_params_)
clf = RandomForestClassifier(n_estimators=150,criterion='gini',max_features='log2')

param = {'max_depth':[None,5,7,9,12],

         'min_samples_split' : [3,5,7],

         'min_weight_fraction_leaf' : [0.0,0.1],

         'max_leaf_nodes' : [40,60,80,100]

         }

optimzation = GridSearchCV(clf,param_grid = param , cv=5 , scoring='accuracy')

optimzation.fit(x_train,y)

print('Best score : ' , optimzation.best_score_)

print('Best parameter : ' , optimzation.best_params_)
clf = RandomForestClassifier(n_estimators=150,

                             criterion='gini',

                             max_features='log2',

                             min_samples_split=5,

                             min_weight_fraction_leaf=0.1,

                             max_leaf_nodes=100,

                             max_depth=None)

clf.fit(x_train,y)

y_pred = clf.predict(x_test)

print(clf.classes_)

print(le.classes_)

pred = []

for yped in y_pred:

    if yped ==0:

        pred.append('Ghost')

    elif yped == 1:

        pred.append('Ghoul')

    else:

        pred.append('Goblin')

result = pd.DataFrame({'id':test_ids,'type':pred})

result.to_csv('randomforest_ghost.csv',index=False)
X_train,X_test,y_train,y_test = train_test_split(x_train,y,test_size=0.2,random_state=42)



clf = SVC(kernel='rbf',probability=True)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print('Accuracy %f : ' %(accuracy_score(y_test,y_pred)))
dtrain =  xgb.DMatrix(x_train,label=y)

dtest = xgb.DMatrix(x_test)

xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.075,

    "objective": "multi:softmax",

    'max_depth': 7,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'num_class':3

    #'eval_metric': 'mae',

}

clf = xgb.train(xgb_params,dtrain,num_boost_round=350)

y_pred = clf.predict(dtest)

pred = []

for yped in y_pred:

    if yped ==0:

        pred.append('Ghost')

    elif yped == 1:

        pred.append('Ghoul')

    else:

        pred.append('Goblin')

result = pd.DataFrame({'id':test_ids,'type':pred})

result.to_csv('xgb_ghost.csv',index=False)