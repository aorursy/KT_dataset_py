## import all libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, cross_val_score
titanic = pd.read_csv("../input/train.csv")
titanic.head()
## get the length and shape of train data
dataframe_len = len(titanic)
titanic.shape
titanic.info()
titanic_test = pd.read_csv("../input/test.csv")
test = titanic_test.copy()
titanic_test.head()
titanic.describe()
### null check on train data
titanic.isnull().sum()
titanic_test.info()
#### null check on test data
titanic_test.isnull().sum()
print("Proportion of Survived:",sum(titanic['Survived']==1)/len(titanic['Survived']))
print("Proportion of NOT Survived:",sum(titanic['Survived']==0)/len(titanic['Survived']))
titanic['Sex'].value_counts()
titanic.groupby('Sex').Survived.value_counts()
titanic['Pclass'].value_counts()
titanic.groupby('Pclass').Survived.value_counts()
titanic[['Pclass','Survived']].groupby('Pclass',as_index=False).mean()
titanic[['Sex','Survived']].groupby('Sex',as_index=False).mean()
sns.barplot('Sex','Survived',data=titanic)
sns.barplot('Pclass','Survived',data=titanic)
titanic['Embarked'].value_counts()
titanic.groupby('Embarked').Survived.value_counts()
titanic[['Embarked','Survived']].groupby('Embarked',as_index=False).mean()
sns.barplot('Embarked','Survived',data=titanic)
titanic['Parch'].value_counts()
titanic.groupby('Parch').Survived.value_counts()
titanic[['Parch','Survived']].groupby('Parch',as_index=False).mean()
sns.barplot('Parch','Survived',data=titanic,ci=None)
titanic['SibSp'].value_counts()
titanic.groupby('SibSp').Survived.value_counts()
titanic[['SibSp','Survived']].groupby('SibSp',as_index=False).mean()
sns.barplot('SibSp','Survived',data=titanic)
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.violinplot('Sex','Age',data=titanic,hue='Survived',split=True)
plt.subplot(1,3,2)
sns.violinplot('Pclass','Age',data=titanic,hue='Survived',split=True)
plt.subplot(1,3,3)
sns.violinplot('Embarked','Age',data=titanic,hue='Survived',split=True)
dataframe = pd.concat([titanic,titanic_test])
## map gender fields
dataframe['Sex'] = dataframe['Sex'].map({'female':1,'male':0})
dataframe.head()
dataframe['Embarked'].unique()
dataframe['Embarked'].value_counts()
### fill with mode value
dataframe['Embarked'] = dataframe['Embarked'].fillna('S')
dataframe['Embarked'].unique()
### mapping 
dataframe['Embarked'] = dataframe['Embarked'].map({'S':0,'C':1,'Q':2})
dataframe.head()
sns.boxplot(dataframe['Age'])
sns.boxplot(dataframe['Fare'])
## extract title from name
dataframe['Title'] = dataframe['Name'].str.extract('([A-Za-z]+\.)')
dataframe.head()
dataframe['Title'].unique()
### mapping
dataframe['Title'] = dataframe['Title'].replace(['Don.','Major.','Sir.', 'Col.', 'Capt.','Jonkheer.'],'Mr.')
dataframe['Title'] = dataframe['Title'].replace(['Dona.','Lady.', 'Countess.'],'Mrs.')
dataframe['Title'] = dataframe['Title'].replace(['Ms.', 'Mlle.', 'Mme.'],'Miss.')
dataframe['Title'].unique()
### mapping
dataframe['Title'] = dataframe['Title'].map({'Mr.':1,'Mrs.':2,'Miss.':3,'Master.':4,'Rev.':5,'Dr.':6})
dataframe.isnull().sum()
dataframe.head()
### new family size feature creation
dataframe['FamilySize'] = dataframe['SibSp'] +  dataframe['Parch'] + 1
dataframe.isnull().sum()
### new feature creation
dataframe['Ticket_N'] = dataframe['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
dataframe['Ticket_L'] = dataframe['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
dataframe.head()
### missing value treatment
dataframe['Age']=dataframe['Age'].fillna(dataframe['Age'].median())
### handling age columns and creating brackets

dataframe.loc[(dataframe['Age']<=10),'Age'] = 0
dataframe.loc[(dataframe['Age']>10) & (dataframe['Age']<=15) ,'Age'] = 1
dataframe.loc[(dataframe['Age']>15) & (dataframe['Age']<=20) ,'Age'] = 2
dataframe.loc[(dataframe['Age']>20) & (dataframe['Age']<=25) ,'Age'] = 3
dataframe.loc[(dataframe['Age']>25) & (dataframe['Age']<=30) ,'Age'] = 4
dataframe.loc[(dataframe['Age']>30) & (dataframe['Age']<=35) ,'Age'] = 5
dataframe.loc[(dataframe['Age']>35) & (dataframe['Age']<=40) ,'Age'] = 6
dataframe.loc[(dataframe['Age']>40) & (dataframe['Age']<=45) ,'Age'] = 7
dataframe.loc[(dataframe['Age']>45),'Age'] = 8
## handling fare column and creating brackets
dataframe['Fare'] = dataframe['Fare'].fillna(dataframe['Fare'].median()).astype(int)

dataframe.loc[(dataframe['Fare']<=2),'Fare'] = 0
dataframe.loc[(dataframe['Fare']>2) & (dataframe['Fare']<=5) ,'Fare'] = 2
dataframe.loc[(dataframe['Fare']>5) & (dataframe['Fare']<=8) ,'Fare'] = 3
dataframe.loc[(dataframe['Fare']>8) & (dataframe['Fare']<=15) ,'Fare'] = 4
dataframe.loc[(dataframe['Fare']>15) & (dataframe['Fare']<=50) ,'Fare'] = 5
dataframe.loc[(dataframe['Fare']>50),'Fare'] = 6
#handling cabin column for missing value
dataframe['Cabin'] = dataframe['Cabin'].fillna('Missing')
dataframe['Cabin'] = dataframe['Cabin'].str[:1]

dataframe.head()
dataframe['Cabin'].unique()
### and creating brackets
Cabin_mapping = {"A": 1, "B": 2, "C": 3, "D": 4,"E": 5,"F": 6,"G": 7,"M": 8,"T":9 }
dataframe['Cabin'] = dataframe['Cabin'].map(Cabin_mapping)
#create IsAlone column
family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
dataframe['FamilySize_Bracket'] = dataframe['FamilySize'].map(family_map)
dataframe['IsAlone'] = dataframe['FamilySize'].map(lambda s: 1 if s == 1 else 0)
# Extracting surnames from Name
dataframe['Last_Name'] = dataframe['Name'].apply(lambda x: str.split(x, ",")[0])
dataframe.head()
DEFAULT_SURVIVAL_VALUE = 0.5
dataframe['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in dataframe[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                dataframe.loc[dataframe['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                dataframe.loc[dataframe['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in dataframe.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    dataframe.loc[dataframe['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    dataframe.loc[dataframe['PassengerId'] == passID, 'Family_Survival'] = 0
                        

sns.catplot(x="FamilySize",y="Survived",data = dataframe.iloc[:dataframe_len],kind="bar")
#### encoding 
encoder = LabelEncoder()
dataframe.FamilySize_Bracket=encoder.fit_transform(dataframe.FamilySize_Bracket)
#remove unwanted columns from dataframe
dataframe.drop(labels=['Name', 'PassengerId', 'Ticket', 'Last_Name','Ticket_L'], axis=1, inplace=True)
### now time to separate train and test data
titanic = dataframe[:891]
titanic_test = dataframe[891:]
titanic_test.tail()
titanic.head()
X_train = titanic.drop('Survived',axis=1)
y_train = titanic['Survived']
X_test = titanic_test.drop(['Survived'],axis=1).copy()
print(X_train.shape)   
print(y_train.shape)    
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
score = lr.score(X_train, y_train)
score
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
score = svc.score(X_train, y_train)
score
lnsvc = LinearSVC()
lnsvc.fit(X_train,y_train)
y_pred_lnsvc = lnsvc.predict(X_test)
score = lnsvc.score(X_train, y_train)
score
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_dtc = dtc.predict(X_test)
score = dtc.score(X_train, y_train)
score
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
score = rfc.score(X_train, y_train)
score
#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))
rf = RandomForestClassifier(random_state = 1)
kfold = StratifiedKFold(n_splits=8)
param_grid =  {'n_estimators': [100],
               'criterion':['gini'],
                                  'bootstrap': [True],
                                  'max_depth': [4],
                                  'max_features': ['sqrt'],
                                  'min_samples_leaf': [4],
                                  'min_samples_split': [2]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, cv = kfold, verbose = True, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train,y_train)
y_hat_rf_grid = best_clf_rf.best_estimator_.predict(X_test).astype(int)    ### predict
clf_performance(best_clf_rf,'Random Forest')
import xgboost
classifier = xgboost.XGBClassifier(booster='gbtree',verbose=0,learning_rate=0.1,max_depth=10,objective='binary:logistic',
                  n_estimators=1000,seed=2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_XGB = classifier.predict(X_test)
acc_XGB = round( classifier.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_XGB) + '%')
y_pred_XGB.shape
#### using kfold gridserch
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
kfold = StratifiedKFold(n_splits=8)

xgb_param_grid_best = {'learning_rate':[0.1], 
                  'reg_lambda':[0.3],
                  'gamma': [1],
                  'subsample': [0.8],
                  'max_depth': [2],
                  'n_estimators': [300]
              }

gs_xgb = GridSearchCV(xgb, param_grid = xgb_param_grid_best, cv=kfold, n_jobs= -1, verbose = 1)

gs_xgb.fit(X_train,y_train)
y_hat_gs_xgb = gs_xgb.predict(X_test).astype(int)   ## predict

xgb_best = gs_xgb.best_estimator_
print(f'XGB GridSearch best params: {gs_xgb.best_params_}')
print(f'XGB GridSearch best score: {gs_xgb.best_score_}')
from lightgbm import LGBMClassifier

LGB=LGBMClassifier(boosting_type='gbdt', max_depth=10, learning_rate=0.1, objective='binary', reg_alpha=0,
                  reg_lambda=1, n_jobs=-1, random_state=100, n_estimators=1000)

model = LGB.fit(X_train,y_train)
y_pred_LGB = model.predict(X_test)
#rounding the values
y_pred_LGB = y_pred_LGB.round(0)
#converting from float to integer
y_pred_LGB = y_pred_LGB.astype(int)
acc_LGB = round( model.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_LGB) + '%')
from catboost import CatBoostClassifier

CATB=CatBoostClassifier(learning_rate=0.05,depth=8,boosting_type='Plain',eval_metric='Accuracy',n_estimators=1000,random_state=294)
CATB.fit(X_train,y_train)

# Predicting the Test set results
y_pred_CATB = CATB.predict(X_test)
acc_CATB = round(CATB.score(X_train, y_train) * 100, 2)
print ("Train Accuracy: " + str(acc_CATB) + '%')
d=pd.DataFrame()
d=pd.concat([d,pd.DataFrame(gs_xgb.predict(X_test).astype(int)),pd.DataFrame(best_clf_rf.best_estimator_.predict(X_test).astype(int)),pd.DataFrame(CATB.predict(X_test).astype(int))],axis=1)
d.columns=['1','2','3']

re=d.mode(axis=1)[0]
re.head()
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_hat_rf_grid              ### prediting using random forest model
    })

submission.to_csv('Titanic_submission.csv', index=False)
submission.Survived.value_counts()
# votingC = VotingClassifier(estimators=[('XGB_1',classifier1),('XGB_2',classifier2),('XGB_3',classifier3)], voting='soft', n_jobs=4)

# votingC = votingC.fit(X_train, y_train)
# vote = votingC.predict(X_test)
# submission_v = pd.DataFrame({
#         "PassengerId": titanic_test["PassengerId"],
#         "Survived": vote
#     })

# submission_v.to_csv('Vote_submission.csv', index=False)
# submission_v.Survived.value_counts()




