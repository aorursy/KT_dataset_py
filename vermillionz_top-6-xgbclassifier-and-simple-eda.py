#Base

import pandas as pd

import numpy as np



#Plot

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#Modelos

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier



#Metrics

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

#Model Select

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import RFECV



from hyperopt import hp, tpe, fmin



import warnings

warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
print('TrainShape is:',df_train.shape,'\TestShape is:',df_test.shape)
df_append = df_train.append(df_test)
print('AppendShape is',df_append.shape)
df_append.head(2)
df_append.isna().sum()
sns.heatmap(df_append.isnull(), cbar=False)
df_append['Age'].describe()
df_append.groupby('Sex').Age.plot(kind='kde')

plt.legend()

plt.show()
df_append.Age.plot(kind='hist')

plt.legend()

plt.show()
facetgrid = sns.FacetGrid(df_append, col="Sex", row="Survived", margin_titles=True)

facetgrid.map(plt.hist, "Age",color="Blue");
df_append['Age'].fillna(df_append['Age'].median(),inplace=True)

df_append['Fare'].fillna(df_append['Fare'].median(),inplace=True)
df_append['Embarked'].describe()
df_append['Embarked'].fillna('S',inplace=True)
df_append['Embarked'].value_counts().plot(kind='bar', alpha=0.75)
sns.set(font_scale=1)

factorplot = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=df_append, saturation=.8,

                    kind="bar", ci=None, aspect=.6)

(factorplot.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

factorplot.fig.suptitle('Sobreviventes do sexos masc. e femin. por classe');
df_append.Age[df_append.Pclass == 1].plot(kind='kde')    

df_append.Age[df_append.Pclass == 2].plot(kind='kde')

df_append.Age[df_append.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution class")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best');
plt.figure(figsize=(10, 10))

corr = df_append.corr()

sns.heatmap(corr,square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation features');
df_append.dtypes
df_append['Sex'] = df_append['Sex'].replace('male',0)

df_append['Sex'] = df_append['Sex'].replace('female',1)

df_append['Embarked'] = df_append['Embarked'].replace('S',0)

df_append['Embarked'] = df_append['Embarked'].replace('Q',1)

df_append['Embarked'] = df_append['Embarked'].replace('C',2)
df_append.head(2)
df_append['Title'] = df_append['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

df_append['Title'].value_counts(normalize=True)*100
mapping = {'Mlle': 'Miss', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs',

           'Jonkheer': 'Rare', 'Lady': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Mrs', 'Rev':'Rare', 'Dr':'Rare'}



df_append.replace({'Title': mapping}, inplace=True)



df_append['Title'].value_counts(normalize=True)*100
df_append['Title'] = df_append['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Rare':4})
bins = [-1, 0, 18, 25, 35, 60, np.inf]

labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

df_append['AgeGroup'] = pd.cut(df_append["Age"], bins, labels = labels)

age_mapping = {'Unknown': None,'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}

df_append['AgeGroup'] = df_append['AgeGroup'].map(age_mapping)
df_append['Family_size'] = df_append['SibSp'] + df_append['Parch'] + 1
df_append['Alone'] = 1

df_append['Alone'].loc[df_append['Family_size'] > 1] = 0
# Cabin

df_append.Cabin.fillna('0', inplace=True)

df_append.loc[df_append.Cabin.str[0] == 'A', 'Cabin'] = 1

df_append.loc[df_append.Cabin.str[0] == 'B', 'Cabin'] = 1

df_append.loc[df_append.Cabin.str[0] == 'C', 'Cabin'] = 1

df_append.loc[df_append.Cabin.str[0] == 'D', 'Cabin'] = 2

df_append.loc[df_append.Cabin.str[0] == 'E', 'Cabin'] = 2

df_append.loc[df_append.Cabin.str[0] == 'F', 'Cabin'] = 3

df_append.loc[df_append.Cabin.str[0] == 'G', 'Cabin'] = 3

df_append.loc[df_append.Cabin.str[0] == 'T', 'Cabin'] = 3
df_append['Ticket_Frequency'] = df_append.groupby('Ticket')['Ticket'].transform('count')
df_append['Fare_per_person'] = df_append.Fare / np.mean(df_append.SibSp + df_append.Parch + 1)
df_append.head(2)
df_append.isna().sum()
features = ['Embarked','Fare','Pclass','Sex','Title','AgeGroup','Family_size','Alone','Ticket_Frequency','Fare_per_person']
df_train = df_append[0:891]

df_test = df_append[891:]
X = df_train[features]

y = df_train['Survived'].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=78941)
y_train.value_counts()
space = {'eta':hp.uniform('eta', 0.01, 1),

         'max_depth':hp.uniform('max_depth', 10,600),

         'n_estimators':hp.uniform('n_estimators', 10, 3000),

         'learning_rate':hp.uniform('learning_rate', 0.0001,0.95),

         'colsample_bytree':hp.uniform('colsample_bytree', 0.1, 0.95),

          'subsample':hp.uniform('subsample',0.1, 1),

          'min_child_weight':hp.uniform('min_child_weight',1,10)

        }



def objective(params):

    params = {'eta': params['eta'],

             'max_depth': int(params['max_depth']),

             'n_estimators': int(params['n_estimators']),

             'learning_rate': params['learning_rate'],

             'colsample_bytree': params['colsample_bytree'],

              'objective':'binary:logistic',

              'subsample':params['subsample'],

              'min_child_weight':params['min_child_weight']

             }

    cv = StratifiedKFold(n_splits=7, random_state=974411, shuffle=False)

    xb_a= XGBClassifier(**params)

    score = cross_val_score(xb_a, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1).mean()

    return -score
best = fmin(fn= objective, space= space, max_evals=20, rstate=np.random.RandomState(1), algo=tpe.suggest)
xb_b = XGBClassifier(random_state=0,

                        eta=best['eta'], 

                        max_depth= int(best['max_depth']),

                        n_estimators= int(best['n_estimators']),

                        learning_rate= best['learning_rate'],

                        colsample_bytree= best['colsample_bytree'],

                         objective='binary:logistic',

                          subsample=best['subsample'],

                          min_child_weight=best['min_child_weight']

                       )



xb_b.fit(X_train, y_train);
preds = xb_b.predict(X_test)

target_names = ['Morreu', 'Sobreviveu']

print(classification_report(y_test, preds,target_names=target_names))

print(confusion_matrix(y_test, preds))

print(accuracy_score(y_test, preds))
from yellowbrick.model_selection import RFECV
# Instantiate RFECV visualizer with a linear SVM classifier

cv = StratifiedKFold(n_splits=7, random_state=974411, shuffle=False)

visualizer = RFECV(RandomForestClassifier(),scoring='accuracy',cv=cv)



visualizer.fit(X_train, y_train)        # Fit the data to the visualizer

visualizer.show(); 
df_test.isna().sum()
preds = visualizer.predict(df_test[features])

Id = df_test.PassengerId

output = pd.DataFrame({'PassengerId': Id, 'Survived':preds})

output.to_csv('../input/output/submission.csv', index=False)