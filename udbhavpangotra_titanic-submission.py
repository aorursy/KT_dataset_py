import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

#from catboost import CatBoostClassifier

import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV



import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

%matplotlib inline

import numpy as np

import pandas as pd

from sklearn.preprocessing import Imputer
#import data

test = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df=train

test_df=test

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

data_df = train_df.append(test_df) # The entire data: train + test.

passenger_id=test_df['PassengerId']



## We will drop PassengerID and Ticket since it will be useless for our data. 

train_df.drop(['PassengerId'], axis=1, inplace=True)

test_df.drop(['PassengerId'], axis=1, inplace=True)

test_df.shape
print (train_df.isnull().sum())

print (''.center(20, "*"))

print (test_df.isnull().sum())

sns.boxplot(x='Survived',y='Fare',data=train_df)
train_df=train_df[train_df['Fare']<400]
train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)

test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)
train_df.head()
pd.options.display.max_columns = 99

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

train_df.head()
for name_string in data_df['Name']:

    data_df['Title']=data_df['Name'].str.extract('([A-Za-z]+)\.',expand=True)

    

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)



data_df['Title'].value_counts()

train_df['Title']=data_df['Title'][:891]

test_df['Title']=data_df['Title'][891:]



titles=['Mr','Miss','Mrs','Master','Rev','Dr']

for title in titles:

    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]

    #print(age_to_impute)

    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

data_df.isnull().sum()







train_df['Age']=data_df['Age'][:891]

test_df['Age']=data_df['Age'][891:]

test_df.isnull().sum()
## Family_size seems like a good feature to create

train_df['family_size'] = train_df.SibSp + train_df.Parch+1

test_df['family_size'] = test_df.SibSp + test_df.Parch+1
def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a



train_df['family_group'] = train_df['family_size'].map(family_group)

test_df['family_group'] = test_df['family_size'].map(family_group)
train_df['child'] = [1 if i<16 else 0 for i in train_df.Age]

test_df['child'] = [1 if i<16 else 0 for i in test_df.Age]

train_df.child.value_counts()
train_df['calculated_fare'] = train_df.Fare/train_df.family_size

test_df['calculated_fare'] = test_df.Fare/test_df.family_size
def fare_group(fare):

    a= ''

    if fare <= 5:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a
train_df['fare_group'] = train_df['calculated_fare'].map(fare_group)

test_df['fare_group'] = test_df['calculated_fare'].map(fare_group)
#comment after use

train_df = pd.get_dummies(train_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

test_df = pd.get_dummies(test_df, columns=['Title',"Pclass",'Embarked', 'family_group', 'fare_group'], drop_first=True)

train_df.drop(['Cabin', 'family_size','Ticket','Name', 'Fare'], axis=1, inplace=True)

test_df.drop(['Ticket','Name','family_size',"Fare",'Cabin'], axis=1, inplace=True)
def age_group_fun(age):

    a = ''

    if age <= 2:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 15:

        a = 'child'

    elif age <= 19:

        a = 'teenager'

    elif age <= 32:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a
train_df['age_group'] = train_df['Age'].map(age_group_fun)

test_df['age_group'] = test_df['Age'].map(age_group_fun)
train_df = pd.get_dummies(train_df,columns=['age_group'], drop_first=True)

test_df = pd.get_dummies(test_df,columns=['age_group'], drop_first=True)

#Lets try all after dropping few of the column.

train_df.drop(['Age','calculated_fare'],axis=1,inplace=True)

test_df.drop(['Age','calculated_fare'],axis=1,inplace=True)
train_df.head()

train_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)

test_df.drop(['Title_Rev','age_group_old','age_group_teenager','age_group_senior_citizen','Embarked_Q'],axis=1,inplace=True)
X = train_df.drop('Survived', 1)

y = train_df['Survived']

#testing = test_df.copy()

#testing.shape
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    svm.SVC(probability=True),

    DecisionTreeClassifier(),

    CatBoostClassifier(),

    XGBClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

    





log_cols = ["Classifier", "Accuracy"]

log= pd.DataFrame(columns=log_cols)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit



SSplit=StratifiedShuffleSplit(test_size=0.2,random_state=7)

acc_dict = {}



for train_index,test_index in SSplit.split(X,y):

    X_train,X_test=X.iloc[train_index],X.iloc[test_index]

    y_train,y_test=y.iloc[train_index],y.iloc[test_index]

    

    for clf in classifiers:

        name = clf.__class__.__name__

          

        clf.fit(X_train,y_train)

        predict=clf.predict(X_test)

        acc=accuracy_score(y_test,predict)

        if name in acc_dict:

            acc_dict[name]+=acc

        else:

            acc_dict[name]=acc
log['Classifier']=acc_dict.keys()

log['Accuracy']=acc_dict.values()

#log.set_index([[0,1,2,3,4,5,6,7,8,9]])

%matplotlib inline

sns.set_color_codes("muted")

ax=plt.subplots(figsize=(10,8))

ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')

ax.set_xlabel('Accuracy',fontsize=20)

plt.ylabel('Classifier',fontsize=20)

plt.grid(color='r', linestyle='-', linewidth=0.5)

plt.title('Classifier Accuracy',fontsize=20)




## Necessary modules for creating models. 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

testframe = std_scaler.fit_transform(test_df)

testframe.shape
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_score,recall_score,confusion_matrix

logreg = LogisticRegression(solver='liblinear', penalty='l1')

logreg.fit(X_train,y_train)

predict=logreg.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]

penalties = ['l1','l2']



param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=10,shuffle=True), n_jobs=1,scoring='accuracy')
grid.fit(X_train,y_train)

print (grid.best_params_)

print (grid.best_score_)

print(grid.best_estimator_)
#grid.best_estimator_.fit(X_train,y_train)

#predict=grid.best_estimator_.predict(X_test)

#print(accuracy_score(y_test,predict))

logreg_grid=LogisticRegression(C=0.33, class_weight=None, dual=False, fit_intercept=True,

                   intercept_scaling=1, l1_ratio=None, max_iter=100,

                   multi_class='warn', n_jobs=None, penalty='l1',

                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,

                   warm_start=False)

#logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])

logreg_grid.fit(X_train,y_train)

y_pred = logreg_grid.predict(X_test)

logreg_accy = round(accuracy_score(y_test, y_pred), 3)

print (logreg_accy)

print(confusion_matrix(y_test,y_pred))

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))
AdaC=AdaBoostClassifier()



AdaC.fit(X_train,y_train)

predict=AdaC.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))
from sklearn.tree import DecisionTreeClassifier

n_estimator=[50,60,100,150,200,300]

learning_rate=[0.001,0.01,0.1,0.2,0.3]

hyperparam={'n_estimators':n_estimator,'learning_rate':learning_rate}

gridBoost=GridSearchCV(AdaC,param_grid=hyperparam,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1,scoring='accuracy')
gridBoost.fit(X_train,y_train)

print(gridBoost.best_score_)

print(gridBoost.best_estimator_)
gridBoost.params = gridBoost.best_params_

gridBoost.params
gridBoost.best_estimator_.fit(X_train,y_train)

predict_grid=gridBoost.best_estimator_.predict(X_test)

print(accuracy_score(y_test,predict))
xgb=XGBClassifier(max_depth=2, n_estimators=700, learning_rate=0.009,nthread=-1,subsample=1,colsample_bytree=0.8)

xgb.fit(X_train,y_train)

predict=xgb.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
paramsxgb = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }
folds = 3

param_comb = 5

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
grid = GridSearchCV(estimator=xgb, param_grid=paramsxgb, scoring='recall', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3 )

grid.fit(X_train, y_train)
xgb = XGBClassifier(base_score=0.5, booster='gbtree',

                                     colsample_bylevel=1, colsample_bynode=1,

                                     colsample_bytree=1, gamma=0,

                                     learning_rate=0.02, max_delta_step=0,

                                     max_depth=3, min_child_weight=1,

                                     missing=None, n_estimators=600, n_jobs=1,

                                     nthread=1,

                                     random_state=0, reg_alpha=0, reg_lambda=1,

                                     scale_pos_weight=1, seed=None, silent=True,

                                     subsample=1, verbosity=1)
xgb.fit(X_train,y_train)

predict=xgb.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
lda=LinearDiscriminantAnalysis()

lda.fit(X_train,y_train)

predict=lda.predict(X_test)

print(accuracy_score(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_split=20,max_features=0.2, min_samples_leaf=8,random_state=20)

#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

randomforest.fit(X_train, y_train)

y_pred = randomforest.predict(X_test)

random_accy = round(accuracy_score(y_pred, y_test), 3)

print (random_accy)

print(precision_score(y_test,y_pred))

print(recall_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
from sklearn.ensemble import VotingClassifier
# Prediction with catboost algorithm.

from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=False, one_hot_max_size=3)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

acc = round(accuracy_score(y_pred, y_test), 3)

print(acc)
y_predict=lda.predict(testframe)
temp = pd.DataFrame(pd.DataFrame({

        "PassengerId": passenger_id,

        "Survived": y_predict

    }))
temp.to_csv('submission_3.csv',index = False)
temp.head()
# df_test.head()
# df_train.describe()
# df_train.isna().sum()
# df_train=df_train.drop(columns=['Cabin'])
# df_train.isna().sum()
# df_train['Age'].mean()
# sns.boxplot(df_train['Age'])
# df_train['Age'].describe()
# df_train['Age']=df_train.fillna(df_train['Age'].mean())
# df_train.describe()
# from sklearn.preprocessing import LabelEncoder
# X_test = df_train['Survived']

# X_train = df_train.drop(columns=['Survived'])