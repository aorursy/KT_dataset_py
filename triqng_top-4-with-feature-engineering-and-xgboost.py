import numpy as np

import pandas as pd

import time

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.linear_model import RidgeClassifier

sb.set(font_scale=1) 
# Load data

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train_org = df_train

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_full = pd.concat([df_train,df_test])

df_train.info()

df_test.info()
# Embarked has 2 missing values

df_train[df_train['Embarked'].isnull()]
# Show embarked values

df_train['Embarked'].unique()# Show embarked values

print(df_full['Embarked'].unique())

# Distribution of Embarked vs fares

sb.boxplot(x="Embarked", y="Fare",

            hue="Pclass", palette=["m", "g"],

            data=df_full)
df_train.loc[df_train['Embarked'].isnull(),'Embarked'] = 'C'
df_train.Cabin.fillna('N',inplace=True)

df_test.Cabin.fillna('N',inplace=True)
# Missing fare in test set

df_test[df_test['Fare'].isnull()]
#sb.boxplot(x="Pclass", y="Fare",

 #            palette=["m", "g"],

  #          data=df_full)



#guess_class_3_fare = df_full[(df_full['Pclass']==3) & (df_full['Fare'].notnull())]['Fare'].mean()

#print(guess_class_3_fare)

#df_test.Fare.fillna(guess_class_3_fare,inplace=True)
# See distribution of fare in class 3

df_full = pd.concat([df_train, df_test])

class_3_fare = df_full[df_full.Pclass==3]['Fare']

ax = sb.distplot(class_3_fare)


guess_value = float(class_3_fare.mode())

df_test.Fare.fillna(guess_value, inplace=True)

df_test.loc[df_test['Fare'].isnull(),'Fare'] = guess_value
# Find family size

df_train['FamSize'] = df_train['Parch'] + df_train['SibSp'] + 1

df_test['FamSize'] = df_test['Parch'] + df_test['SibSp'] + 1

# Fare per person

df_train['FarePp'] = df_train['Fare']/df_train['FamSize']

df_test['FarePp'] = df_test['Fare']/df_test['FamSize']

# IsAlone flag

df_train['IsAlone'] = 0

df_train.loc[df_train['FamSize']==1,'IsAlone'] = 1

df_test['IsAlone'] = 0

df_test.loc[df_test['FamSize']==1,'IsAlone'] = 1
# Get family/surname

split_cols = df_train['Name'].str.split(',')

surnames = []

for row in range(0, split_cols.shape[0]):

    name = split_cols[row][0]

    surnames.append(name)

df_train['Surname'] = surnames



split_cols = df_test['Name'].str.split(',')

surnames = []

for row in range(0, split_cols.shape[0]):

    name = split_cols[row][0]

    surnames.append(name)

df_test['Surname'] = surnames

#df_train['Surname'] =df_train['Surname'][0] 
# Check if the fare is per person or for the whole family

fig, axes = plt.subplots(nrows=2,figsize=(10,10))

sb.scatterplot(y='FamSize',x='Fare',data=df_train, ax=axes[0])

axes[0].set_title('Unmodified Fare values')

sb.scatterplot(y='FamSize',x='FarePp',data=df_train,ax=axes[1])

axes[1].set_title('Per person Fare values')
# Check the fare outliner

df_train[df_train['Fare']>500]
# Remove the outliners

#df_train = df_train[df_train['Fare']<500]
# Extract the first letter of the cabins

df_train['CabinClass'] = df_train['Cabin'].astype(str).str[0]

df_test['CabinClass'] = df_test['Cabin'].astype(str).str[0]
# Check fare sorted by cabin class

df_train[df_train['CabinClass']=='B'].sort_values(['Cabin','CabinClass','Name'])

# 
# Create a flag for people where the rest of the family died

df_train['AllDied'] = 1

df_test['AllDied'] = 1

df_train['AllSurvived'] = 0

df_test['AllSurvived'] = 0

df_train =df_train.reset_index()

df_test=df_test.reset_index()

for i in range(0,df_train.shape[0]):

    

    name = df_train.loc[i,'Surname']

    survives = df_train.loc[df_train['Surname']==name,'Survived'].sum() - df_train.loc[i,'Survived']

    if survives > 0:

        df_train.loc[i,'AllDied'] = 0

        df_test.loc[df_test['Surname']==name,'AllDied'] = 0 

    elif df_train.loc[i,'Survived'] == 1: # only the person survived

        df_test.loc[df_test['Surname']==name,'AllDied'] = 0 

    if survives == (df_train.loc[i,'FamSize'] - 1): # rest of family survived

        df_train.loc[i,'AllSurvived'] = 1

        if df_train.loc[i,'Survived'] == 1: # The person survived as well

            df_test.loc[df_test['Surname']==name,'AllSurvived'] = 1
import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\. ', name)

    if title_search:

        return title_search.group(1)

    return ""

all_data = [df_train,df_test]

for data in all_data:

    data['Title'] = data['Name'].apply(get_title)



for data in all_data:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')

    data['Title'] = data['Title'].replace('Mlle','Miss')

    data['Title'] = data['Title'].replace('Ms','Miss')

    data['Title'] = data['Title'].replace('Mme','Mrs')

    

print(pd.crosstab(df_train['Title'], df_train['Sex']))

print("----------------------")

print(df_train[['Title','Survived']].groupby(['Title'], as_index = False).mean())
df_train.Ticket.head()
def get_ticket_type(ticket):

    type_search = re.search('^([A-Za-z]+)', ticket)

    if type_search:

        return type_search[0]

    return ""

for data in all_data:

    data['TicketType'] = data['Ticket'].apply(get_ticket_type)

for data in all_data:

    data['SpecialTicket'] = 0

    data.loc[data['TicketType'] != '','SpecialTicket'] = 1

#df_train[['SpecialTicket','TicketType']]

df_train.to_csv('temp.csv')
# Use title to predict age

df_full = pd.concat([df_train,df_test])

df_full.info()

titles = df_full['Title'].unique().tolist()

for title in titles:

    df_title = df_full[df_full['Title']==title]

    age = df_title[df_title['Age'].notnull()]['Age'].median()

    df_train.loc[(df_train['Title']==title) & (df_train['Age'].isnull()),'Age'] = age

    df_test.loc[(df_test['Title']==title) & (df_test['Age'].isnull()),'Age'] = age

    print(title,': ',age)
sb.barplot(x='TicketType',y='Survived',data=df_train)
sb.barplot(x='Title',y='Age',data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Title',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Pclass',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Sex',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
# Age and sex 

fig, axes = plt.subplots(nrows=2,figsize=(6,8))

sb.kdeplot(df_train[(df_train['Survived']==1) & (df_train['Sex']=='male')]['Age'],label='Male',

    shade=True,ax=axes[0])

sb.kdeplot(df_train[(df_train['Survived']==1) & (df_train['Sex']=='female')]['Age'],label='Female',

    shade=True,ax=axes[0])

axes[0].set_title('Survived')



sb.kdeplot(df_train[(df_train['Survived']==0) & (df_train['Sex']=='male')]['Age'],label='Die, Male',

    shade=True,ax=axes[1])

sb.kdeplot(df_train[(df_train['Survived']==0) & (df_train['Sex']=='female')]['Age'],label='Die, Female',

    shade=True,ax=axes[1])

axes[1].set_title('Die')  
sb.barplot(x='SibSp',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Parch',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='TicketType',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Embarked',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='IsAlone',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='AllDied',y='Survived', hue='Sex',data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='AllSurvived',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='SpecialTicket',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='SibSp',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='Parch',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
sb.barplot(x='CabinClass',y='Survived', data=df_train,

            linewidth=3,

            capsize = .05,

            errcolor='blue',

            errwidth = 2)
# Code the sex column

df_train.loc[df_train['Sex']=='male','Sex'] = 1

df_train.loc[df_train['Sex']=='female','Sex'] = 0

df_test.loc[df_test['Sex']=='male','Sex'] = 1

df_test.loc[df_test['Sex']=='female','Sex'] = 0
df_train['Sex'] = df_train['Sex'].astype(int)

df_test['Sex'] = df_test['Sex'].astype(int)
_ , ax = plt.subplots(figsize =(14, 12))

mask = np.zeros_like(df_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sb.set_style('whitegrid')

sb.heatmap(df_train.corr(),

annot=True,

linewidths=.2, 

            ax = ax,

            linecolor='white',

            cmap = 'RdBu',

            mask = mask,

            fmt='.2g',

            center = 0,

            #square=True

            )
cat_cols = ['Embarked','Title','Pclass','TicketType']

df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=False)

df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=False)
df_train.info()
dropped_cols = ['index','PassengerId','Name','Fare','Ticket','Cabin','Surname','SpecialTicket','FamSize','AllSurvived','CabinClass']

df_train.drop(dropped_cols,axis=1,inplace=True)

test_id = df_test['PassengerId']

df_test.drop(dropped_cols,axis=1,inplace=True)
X = df_train.drop(['Survived'],axis=1)

y = df_train['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .3, random_state=3

)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#X_train_scale = scaler.fit_transform(X_train)

#X_test_org = X_test

#X_test = scaler.transform(X_test)
model = GaussianNB()

model = XGBClassifier(learning_rate=0.02, gamma=0.3, n_estimators=140, objective='binary:logistic',

                    silent=True, nthread=1,min_child_weight=1,max_depth=2,

                    colsample_bytree= 0.8, subsample= 0.7)#learning_rate=0.05,max_depth=4, n_classifier)



eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train,y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)



import matplotlib.pyplot as plt

results = model.evals_result()

epochs = len(results['validation_0']['error'])

x_axis = range(0, epochs)

# plot log loss

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['logloss'], label='Train')

ax.plot(x_axis, results['validation_1']['logloss'], label='Validation')

ax.legend()

plt.ylabel('Log Loss')

plt.title('XGBoost Log Loss')

plt.show()

# plot classification error

fig, ax = plt.subplots()

ax.plot(x_axis, results['validation_0']['error'], label='Train')

ax.plot(x_axis, results['validation_1']['error'], label='Validation')

ax.legend()

plt.ylabel('Classification Error')

plt.title('XGBoost Classification Error')

plt.show()
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

#y_pred = y_pred[:,0].tolist()

predictions = [round(value) for value in y_pred]

#predictions= predictions or X_test_org['Title_Master'].tolist()

#predictions = [ x|y for (x,y) in zip(predictions, X_test['SpecialTicket'].tolist() )]

#predictions= predictions or X_test_org['Title_Mrs'].tolist()



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
df_check = X_test.copy()

df_check['Real'] = y_test 

df_check['Pred'] = predictions
mismatch = df_check[df_check['Real'] != df_check['Pred']]

mismatch = df_train_org.loc[mismatch.index]

mismatch.to_csv('wrong.csv')
df_test.fillna(0,inplace=True)

df_test.describe()
# Make prediction on test set

df_test['CabinClass_T'] = 0

df_test = df_test.reindex(columns=X.columns)

df_test_np = df_test #.to_numpy()

#X_final_test = scaler.fit_transform(df_test_np)

X_final_test = df_test_np

final_pred = model.predict(X_final_test)

#final_pred = final_pred[:,0].tolist()

final_predictions = [round(value) for value in final_pred]

#predictions= predictions or X_test_org['Title_Master'].tolist()

submission_headers = ['PassengerId','Survived']

submissions = pd.DataFrame(list(zip(test_id,final_predictions)),columns=submission_headers)

submissions.to_csv('submission_final.csv',index=False)