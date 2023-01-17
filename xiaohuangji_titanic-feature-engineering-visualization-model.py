# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(df_train.shape)

print(df_test.shape)
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score



# Figures inline and set visualization style

%matplotlib inline

sns.set()
df_train.info()
df_train.head()
df_train.describe()
sns.countplot(x='Survived', data=df_train);

sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);

df_train.groupby(['Sex']).Survived.sum()

sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);

sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);

sns.distplot(df_train.Fare, kde=False);

df_train.groupby('Survived').Fare.hist(alpha=0.6);

sns.distplot(df_train.dropna().Age, kde=False);

sns.swarmplot(x='Survived', y='Fare', data=df_train);

df_train.info()
df_train['Title']=df_train.Name.str.extract(r'(\w+)\.')



def get_features(df):

    

    #Age Normalize

    df['Age'].fillna(df['Age'].median(),inplace=True)

    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)



    #FamilySize SibSp+Parch

    df['FamilySize']=df['SibSp']+df['Parch']+1

    

    #Extract Title from Name

    df['Title']=df.Name.str.extract(r'(\w+)\.')

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

    #Embarked

    df['Embarked'] = df['Embarked'].fillna('mode')

    

    #FareBin

    df['Fare'].fillna(df['Fare'].median(),inplace=True)

    df['FareBin'] = pd.qcut(df['Fare'], 4)

    

    #keep Pclass

    return df.loc[:,['Pclass','Sex','FamilySize','Title','Embarked','AgeBin','FareBin']]

    

                   

                   
X_train=get_features(df_train)

X_test=get_features(df_test)
X_train['Sex'].values.reshape(-1,1).shape




from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import MinMaxScaler



#one_hot encoding

for col in ['Sex','Title','Embarked']:

    encoder = OneHotEncoder()

    X_train= pd.concat([X_train,  pd.DataFrame(encoder.fit_transform(X_train[col].values.reshape(-1,1)).toarray(),columns=encoder.get_feature_names())],axis=1)

    X_train.drop(col,axis=1,inplace=True)

    X_test= pd.concat([X_test, pd.DataFrame(encoder.transform(X_test[col].values.reshape(-1,1)).toarray(),columns=encoder.get_feature_names())],axis=1)

    X_test.drop(col,axis=1,inplace=True)



for col in ['AgeBin','FareBin']: 

    label = LabelEncoder()

    label.fit(X_train[col].append(X_test[col]))

    X_train[col]=label.transform(X_train[col])

    X_test[col]=label.transform(X_test[col])





#scaler

# scaler = MinMaxScaler()

# X_train[['Age']]=scaler.fit_transform(X_train[['Age']])

# X_test[['Age']]=scaler.transform(X_test[['Age']])

    

print(X_train.columns)





y_train=df_train['Survived']


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import KFold, GridSearchCV,train_test_split

from xgboost import XGBClassifier

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]



models=[

    {

        "name":ensemble.AdaBoostClassifier(),

        "param_grid":

        {

            #AdaBoostClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

            'n_estimators': grid_n_estimator, #default=50

            'learning_rate': grid_learn, #default=1

            'random_state': grid_seed

        }

    },

    {

        "name":XGBClassifier(),

        "param_grid":

            {

            #XGBClassifier - http://xgboost.readthedocs.io/en/latest/parameter.html

            'learning_rate': grid_learn, #default: .3

            'max_depth': [1,2,4,6,8,10], #default 2

            'n_estimators': grid_n_estimator, 

            'seed': grid_seed  

             }

    },

    {

        "name":svm.SVC(),

        "param_grid":{

            #SVC - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r

            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],

            'C': [1,2,3,4,5], #default=1.0

            'gamma': grid_ratio, #edfault: auto

            'decision_function_shape': ['ovo', 'ovr'], #default:ovr

            'probability': [True],

            'random_state': grid_seed

             }

    },

    {

        "name":ensemble.RandomForestClassifier(),

        "param_grid":{

            #RandomForestClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

            'n_estimators': grid_n_estimator, #default=10

            'criterion': grid_criterion, #default=”gini”

            'max_depth': grid_max_depth, #default=None

            'oob_score': [True], #default=False -- 12/31/17 set to reduce runtime 

             ##-- The best parameter for RandomForestClassifier is {'criterion': 'entropy', 'max_depth': 6, 'n_estimators': 100, 'oob_score': True, 'random_state': 0} with a runtime of 146.35 seconds.

            'random_state': grid_seed

             }

    },

    {

        "name":ensemble.BaggingClassifier(),

        "param_grid":{

            #BaggingClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier

            'n_estimators': grid_n_estimator, #default=10

            'max_samples': grid_ratio, #default=1.0

            'random_state': grid_seed

            }

    },

    {

        "name": neighbors.KNeighborsClassifier(),

        "param_grid":{

            #KNeighborsClassifier - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

            'n_neighbors': [1,2,3,4,5,6,7], #default: 5

            'weights': ['uniform', 'distance'], #default = ‘uniform’

            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']

            }

    },

    {

        "name": naive_bayes.GaussianNB(),

        "param_grid":{

            

        }

    },

    {

        "name": naive_bayes.BernoulliNB(),

        "param_grid":{

            #BernoulliNB - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

            'alpha': grid_ratio, #default: 1.0

             }

    },

    {

        "name": linear_model.LogisticRegressionCV(),

        "param_grid":{

            #LogisticRegressionCV - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV

            'fit_intercept': grid_bool, #default: True

            #'penalty': ['l1','l2'],

            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

            'random_state': grid_seed

           }

    }

]



train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, random_state = 0)



trained_model={}



for item in models:

    grid_search = GridSearchCV(

                                estimator = item['name'], param_grid = item['param_grid'], 

                                cv = KFold(n_splits=5, shuffle=True, random_state=555),

                                scoring = 'roc_auc'

                            )

    

    grid_search.fit(X_train,y_train)

    

    best_param = grid_search.best_params_

    clf = item['name'].set_params(**best_param) 



    model=clf.fit(train_x,train_y)

    

    trained_model[clf.__class__.__name__]=model

    

    pred_y=model.predict(val_x)

    #metrics

    print("{}:{}".format(clf.__class__.__name__,accuracy_score(train_y,model.predict(train_x))))

    print("{}:{}".format(clf.__class__.__name__,accuracy_score(val_y,pred_y)))

    

    

submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = trained_model['XGBClassifier'].predict(X_test)

submission.head()

submission.to_csv('submission.csv', index=False)