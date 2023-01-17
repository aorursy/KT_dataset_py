import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
#test = pd.read_csv('https://raw.githubusercontent.com/yuanic/msba/master/hw1_titanic_test.csv')
#train = pd.read_csv('https://raw.githubusercontent.com/yuanic/msba/master/hw1_titanic_train.csv')

test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
train.head()
train.describe()
train.isnull().sum()
plt.figure(figsize=(8,4))
sns.distplot(train[train['Sex']=='male']['Age'].dropna(),color="b")
sns.distplot(train[train['Sex']=='female']['Age'].dropna(),color="r")
plt.title('Distribution of Age by Gender',fontsize=16,fontweight='bold')
import numpy as np
import matplotlib.pyplot as plt

survival_color=['grey','red']
sns.set_context("paper", font_scale=1.5)
fig = plt.figure(figsize=(20,5))
plt.subplot(1, 3, 1)
total = float(len(train)) 
ax = sns.countplot(x="Sex", hue="Survived", data=train,palette=survival_color) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 5,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title('Survivors by Sex',fontsize=14,fontweight='bold')

plt.subplot(1, 3, 2)
total = float(len(train)) 
ax = sns.countplot(x="Pclass", hue="Survived", data=train,palette=survival_color) # for Seaborn version 0.7 and more                                                  

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 5,
            '{:1.2f}'.format(height),
            ha="center") 
plt.title('Survivors by Pclass',fontsize=14,fontweight='bold')

plt.subplot(1, 3, 3)
total = float(len(train)) 
ax = sns.countplot(x="Embarked", hue="Survived", data=train,palette=survival_color) # for Seaborn version 0.7 and more
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 5,
            '{:1.2f}'.format(height/total),
            ha="center") 
plt.title('Survivors by Embarked',fontsize=14, fontweight='bold')
def survival_rate(column):
    categories = train[column].dropna().unique()
    cat = []
    survival_rate = []
    for i in categories:
        survived = len(train[(train[column] == i) & (train['Survived'] == 1)])
        total = len(train[(train[column] == i)])
        cat.append(i)
        survival_rate.append(survived/total)
    output =  pd.DataFrame(
    {'Category': cat,
     'Survival Rate': survival_rate
    })
    return output
sr_sex = survival_rate('Sex')
sr_pclass = survival_rate('Pclass')
sr_embarked = survival_rate('Embarked')
survival_color=['grey','red']
sns.set_context("paper", font_scale=1.5)
fig = plt.figure(figsize=(20,5))

plt.subplot(1, 3, 1)
ax = sns.barplot(x="Category", y="Survival Rate", data=sr_sex , palette=['red']) # for Seaborn version 0.7 and more
plt.title('Survival Rate by Sex',fontsize=14,fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('')

    
plt.subplot(1, 3, 2)
ax = sns.barplot(x="Category", y="Survival Rate", data=sr_pclass , palette=['red']) # for Seaborn version 0.7 and more
plt.title('Survival Rate by PClass',fontsize=14,fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('')

plt.subplot(1, 3, 3)
ax = sns.barplot(x="Category", y="Survival Rate", data=sr_embarked , palette=['red']) # for Seaborn version 0.7 and more
plt.title('Survival Rate by Embarked',fontsize=14,fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('')

sr_sex
sr_pclass
sr_embarked
plt.figure(figsize=(10,6))
sns.distplot(train[train['Survived']==1]['Age'].dropna(),color='red')
sns.distplot(train[train['Survived']==0]['Age'].dropna(),color='grey')
plt.title('Distribution of Age by Survival',fontsize=14,fontweight='bold')
import math
def custom_round(x, base=5):
    if math.isnan(x):
      return float('nan')
    else:
      return round(x/base)*base
    
train['modified_age'] = train['Age'] .apply(lambda x: custom_round(x, base=5))
sr_age = survival_rate('modified_age')
fig = plt.figure(figsize=(20,6))

plt.subplot(1, 1, 1)
ax = sns.barplot(x="Category", y="Survival Rate", data=sr_age , palette=['red']) # for Seaborn version 0.7 and more
plt.title('Survival Rate by Age Buckets',fontsize=22,fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('')

plt.figure(figsize=(10,6))
sns.distplot(train[train['Survived']==1]['Fare'],color="red")
sns.distplot(train[train['Survived']==0]['Fare'],color="grey").set(xlim=(0,200))
plt.title('Distribution of Survival by Fare',fontsize=16,fontweight='bold')
train['modified_fare'] = train['Fare'] .apply(lambda x: custom_round(x, base=50))
sr_fare = survival_rate('modified_fare')
fig = plt.figure(figsize=(5,6))

plt.subplot(1, 1, 1)
ax = sns.regplot(x="Category", y="Survival Rate", data=sr_fare)
#ax = sns.barplot(x="Category", y="Survival Rate", data=sr_fare , palette=['red']) # for Seaborn version 0.7 and more
plt.title('Survival Rate by Fare Buckets',fontsize=22,fontweight='bold')
plt.ylabel('Survival Rate')
plt.xlabel('')
sr_fare.sort_values("Category")
survived = len(train[(train['Fare'] > 75) & (train['Survived'] == 1)])
total = len(train[(train['Fare'] > 75)])
survived/total
survived = len(train[(train['Fare'] < 75) & (train['Survived'] == 1)])
total = len(train[(train['Fare'] < 75)])
survived/total
dummies_pclass = pd.get_dummies(train['Pclass'], prefix="pclass")
dummies_gender = pd.get_dummies(train['Sex'], prefix="gender")
dummies_embarked = pd.get_dummies(train['Embarked'], prefix="embarked")

train_w_dummies = pd.concat([train['PassengerId'],train['Survived'],train['Age'],train['SibSp'],train['Parch'],train['Fare'], dummies_pclass,dummies_gender,dummies_embarked], axis=1)
train_w_dummies['total_travel_size']=train_w_dummies['SibSp'] + train_w_dummies['Parch'] + 1
train_w_dummies[['total_travel_size','SibSp','Parch','Survived']].corr()
title = train['Name'].str.split(',',expand=True)
train['Title']=title[1].str.split('.',expand=True)[0]
train.groupby('Title').size()

allowed_titles = [' Mr',' Miss',' Mrs', ' Master']
train.loc[~train["Title"].isin(allowed_titles), "Title"] = "Others"
dummies_title = pd.get_dummies(train['Title'], prefix="title")
pd.concat([train['Survived'],dummies_title], axis=1).corr()['Survived'].sort_values()
train_w_dummies_v2 = pd.concat([train_w_dummies, dummies_title], axis=1)
survival_rate('Title')
age_lookup = train.groupby(['Title','Pclass','Sex'])['Age'].median().reset_index(name='median')
train_v2 = pd.merge(train, age_lookup, on=['Title','Pclass','Sex'], how='left', suffixes=('','_'))
train_v2['Age_edit'] = np.where(train_v2['Age'].isnull(), train_v2['median'], train_v2['Age'])
train_v2[train_v2['Age'].isnull()].head()
train_w_dummies_v3 = pd.concat([train_w_dummies_v2,train_v2['Age_edit']], axis=1)
train_dataset = train_w_dummies_v3
train_dataset.columns

X=train_dataset[['SibSp', 'Parch', 'total_travel_size',
       'Fare', 'pclass_1', 'pclass_2', 'pclass_3', 'gender_female',
       'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_ Master', 'title_ Miss', 'title_ Mr', 'title_ Mrs',
       'title_Others', 'Age_edit']]
y=train_dataset['Survived']
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_regression
model_1 = LogisticRegression()
model_1.fit(X, y)
passenger = train_v2['PassengerId']
survived = train_v2['Survived']
prediction = model_1.predict(X)
results=pd.DataFrame(np.column_stack((passenger,survived,prediction)))
results.columns=['PassengerID', 'Survived','Prediction']
results['IsCorrect'] = np.where(results['Prediction'] == results['Survived'], 'Yes', 'No')
len(results[results['IsCorrect']=='Yes'])/len(results)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

t_ntrees = []
t_depth =[]
t_auc=[]
X=train_dataset[['SibSp', 'Parch', 'total_travel_size',
       'Fare', 'pclass_1', 'pclass_2', 'pclass_3', 'gender_female',
       'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_ Master', 'title_ Miss', 'title_ Mr', 'title_ Mrs',
       'title_Others', 'Age_edit']]
y=train_dataset['Survived']

for n_trees in range(1, 50, 1):
    for depth in range(1, 10, 1):
        model = RandomForestClassifier(n_estimators = n_trees, max_depth = depth, random_state=50)
        scores = cross_val_score(model, X, y, scoring='roc_auc',cv=5)
        t_ntrees.append(n_trees)
        t_depth.append(depth)
        t_auc.append(scores.mean())

output1 = pd.DataFrame.from_dict(t_ntrees)
output2 = pd.DataFrame.from_dict(t_depth)
output3 = pd.DataFrame.from_dict(t_auc)
output1.columns=['ntrees']
output2.columns=['depth']
output3.columns=['auc']
output = output1.join(output2).join(output3)
output.sort_values(['auc'],ascending=False).head()
model_2 = RandomForestClassifier(n_estimators = 3, max_depth=5, random_state=50)
model_2.fit(X, y)
scores = cross_val_score(model_2, X, y, scoring='roc_auc',cv=5)
print('CV AUC {}, Average AUC {}'.format(scores, scores.mean()))
features = X.columns
feature_importances = model_2.feature_importances_

features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)

features_df.head()
test['total_travel_size']=test['SibSp'] + test['Parch']+1
test_title = test['Name'].str.split(',',expand=True)
test['Title']=test_title[1].str.split('.',expand=True)[0]
test.loc[~test["Title"].isin(allowed_titles), "Title"] = "Others"
test_dummies_title = pd.get_dummies(test['Title'], prefix="title")
test_dummies_pclass = pd.get_dummies(test['Pclass'], prefix="pclass")
test_dummies_gender = pd.get_dummies(test['Sex'], prefix="gender")
test_dummies_embarked = pd.get_dummies(test['Embarked'], prefix="embarked")
test_v2 = pd.merge(test, age_lookup, on=['Title','Pclass','Sex'], how='left', suffixes=('','_'))
test_v2['Age_edit'] = np.where(test_v2['Age'].isnull(), test_v2['median'], test_v2['Age'])
test_v2[test_v2['Age'].isnull()].head()

test_dataset = pd.concat([test['PassengerId'],test_v2['Age_edit'],test['SibSp'],test['Parch'],test['total_travel_size'],test['Fare'],test_dummies_title, test_dummies_pclass,test_dummies_gender,test_dummies_embarked], axis=1)
test_dataset = test_dataset.fillna(test_dataset.mean())
test_dataset.isnull().sum()
test_passenger = test_v2['PassengerId']
test_X = test_dataset[['SibSp', 'Parch', 'total_travel_size',
       'Fare', 'pclass_1', 'pclass_2', 'pclass_3', 'gender_female',
       'gender_male', 'embarked_C', 'embarked_Q', 'embarked_S',
       'title_ Master', 'title_ Miss', 'title_ Mr', 'title_ Mrs',
       'title_Others', 'Age_edit']]

test_prediction = model_1.predict(test_X)
test_results=np.column_stack((test_passenger,test_prediction))
test_results = pd.DataFrame(test_results)
test_results.columns=['PassengerId', 'Survived']
test_results.to_csv('submission_1.csv',index=False)
#test_results.to_csv('/content/gdrive/My Drive/BAX 452 - Machine Learning/test_result.csv',index=False)
test_prediction = model_2.predict(test_X)
test_results=np.column_stack((test_passenger,test_prediction))
test_results = pd.DataFrame(test_results)
test_results.columns=['PassengerId', 'Survived']
test_results.to_csv('submission_2.csv',index=False)
