# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
##prepare datasets

#read train and test data sets

train0=pd.read_csv('../input/titanic/train.csv',index_col="PassengerId")

test0=pd.read_csv('../input/titanic/test.csv',index_col="PassengerId")



#create copies of training and test sets for feature engineering

train1=train0.copy()

test1=test0.copy()



df_all=[train1,test1]
test0
##Create new columns - Title,Fam_size,Cabin_new



#define a function to clean up title values

def clean_title(dataset):

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    return dataset



for df in df_all:

    #create title

    df['Title'] = df.Name.str.extract(' (\w+)\.', expand=False)

    df=clean_title(df)

    #family size

    df['fam_size']=df.SibSp+df.Parch

    #cabin - missing or not

    df['Cabin_new']=df.Cabin.notnull().astype(int)
#combine two datasets into one

all_data=pd.concat([train1,test1],sort=False)

#create new column- ave fare

all_data=all_data.join(all_data.groupby('Ticket')['Ticket'].size(), on='Ticket', rsuffix='_count')
all_data['Ave_Fare']=all_data['Fare']/all_data['Ticket_count']
#split between test and train

train=all_data.iloc[:len(train0)]

test=all_data.iloc[-len(test0):]
#identify correlation between ticket_count/Cabin_new and survival rate

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



sns.barplot(x=all_data.Cabin_new, y=all_data.Survived)
sns.barplot(x=all_data.Ticket_count, y=all_data.Survived)
##feature selection for baseline model

feature1=['Pclass','Sex','Age','Title','fam_size','Cabin_new','Ticket_count','Ave_Fare']

X=train[feature1]

y=train['Survived']



#Ave_Fare has some 0's which indicate that these values are missing

X['Ave_Fare'].replace(0, np.nan, inplace= True)

#Split training and validation sets

from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

##Code categorcal features

cat_feature = [cname for cname in X_train.columns if 

                    X_train[cname].dtype == "object"]

X_train=pd.get_dummies(X_train)

X_test=pd.get_dummies(test[feature1])

X_valid=pd.get_dummies(X_valid)
##Handling missing values

# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
#filling in missing values with KNNImputer

from sklearn.impute import KNNImputer

my_imputer=KNNImputer()

imputed_X_train=pd.DataFrame(my_imputer.fit_transform(X_train),index=X_train.index,columns=X_train.columns)

imputed_X_valid=pd.DataFrame(my_imputer.transform(X_valid),index=X_valid.index,columns=X_valid.columns)

imputed_X_test=pd.DataFrame(my_imputer.transform(X_test),index=X_test.index,columns=X_test.columns)

X_total=pd.concat([imputed_X_train,imputed_X_valid]).sort_index()

y_total=train0.Survived
X_total
##Trainiing LightGBM model as the baseline model

#define score function

from sklearn import metrics

def score(model,feature):

    valid_pred=model.predict(imputed_X_valid[feature])

    valid_score=metrics.roc_auc_score(y_valid,valid_pred)

    print(f"Validation AUC score:{valid_score:.4f}")
#train XGBoost model

from sklearn.ensemble import GradientBoostingClassifier

baseline=GradientBoostingClassifier()

baseline.fit(imputed_X_train,y_train)
#univariate feature selection

from sklearn.feature_selection import SelectKBest,f_classif

selector=SelectKBest(f_classif,k=6)

X_new=selector.fit_transform(imputed_X_train,y_train)

X_new
# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=X_train.index, 

                                 columns=X_train.columns)

# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]
#feature selection using Gradient booster feature importance

feat_importances = pd.Series(baseline.feature_importances_, index=X_train.columns)

feat_importances.plot(kind='bar')
feat_importances2 = pd.Series(selector.scores_, index=X_train.columns)

feat_importances2.plot(kind='bar')
feature2=['Pclass','Cabin_new','Ave_Fare','Sex_female','Sex_male','Title_Miss','Title_Mrs','Title_Mr']

model1=GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,min_samples_split=8,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10)

model1.fit(imputed_X_train[feature2],y_train)
score(model1,feature2) 
#Tuning hyperparameters

from sklearn.model_selection import GridSearchCV

param_test1={'n_estimators':range(50,150,10)}

gsearch1=GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.1,min_samples_split=8,max_depth=8,max_features='sqrt',

                                                           subsample=0.8,random_state=10),param_grid=param_test1,scoring='roc_auc',n_jobs=-1,cv=5)

gsearch1.fit(X_total[feature2],y_total)
gsearch1.best_params_, gsearch1.best_score_
# #train LightGBM model

# import lightgbm as lgb



# dtrain=lgb.Dataset(X_train,label=y_train)

# dvalid=lgb.Dataset(X_valid,label=y_valid)

# param={'num_leaves':64,'objective':'binary'}

# param['metric']='auc'

# num_round=1000

# baseline=lgb.train(param,dtrain,num_round,valid_sets=[dvalid],early_stopping_rounds=10,verbose_eval=False)
##predict on test set

model1.fit(X_total[feature2],y_total)

pred=model1.predict(imputed_X_test[feature2]).astype(int)

submission=pd.DataFrame({'PassengerId':X_test.index,

                          'Survived':pred})

submission.to_csv('submission.csv',index=False)
submission.shape