import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

%pip install feature_engine
df = pd.read_csv('../input/titanic/train.csv')
df_copy = df.copy()
test = pd.read_csv('../input/titanic/test.csv')
test_copy = test.copy()
submission3 = pd.read_csv('../input/titanic/gender_submission.csv')
df.head()
df.drop(columns=['Name'],inplace=True)
df.drop(columns=['Ticket'],inplace=True)
df.drop(columns=['PassengerId'],inplace=True)


test.drop(columns=['PassengerId','Ticket','Name'],inplace=True)
df.info()
df['Pclass'] = df['Pclass'].astype('object')

test['Pclass'] = test['Pclass'].astype('object')
categoricals = [var for var in df.columns if df[var].dtype == 'object' and var != 'Survived']
discreate = [var for var in df.columns if df[var].dtype != 'object' and var != 'Survived' and len(df[var].unique()) < 10]
continuous = [var for var in df.columns if df[var].dtype != 'object' and var != 'Survived'and var not in discreate]
categoricals
discreate
continuous
pd.DataFrame(df[categoricals].nunique(),columns=['unique']).T
missing = {var:np.round(df[var].isnull().mean(),2) for var in df.columns if df[var].isnull().mean() > 0.05}
missing
{var:np.round(df[var].isnull().mean(),2) for var in df.columns if df[var].isnull().mean() < 0.05}

df.isnull().mean().plot.bar(figsize = (10,5))

sns.despine(bottom=True,left=True);
from scipy.stats import norm

i = 1
for col in continuous:
    

    plt.figure(figsize = (10,5))
    plt.figure(i)
    
    sns.distplot(df[col].dropna(),fit=norm)
    
    sns.despine(bottom=True,left=True);
    
    i += 1
plt.figure(figsize = (12,6))
sns.distplot(np.log(1+df['Fare'].dropna()),fit=norm)
sns.despine(bottom=True,left=True);
for var in continuous:
    df[var].fillna(df[var].mean(),inplace = True)
    test[var].fillna(test[var].mean(),inplace = True)
df.isnull().mean()
df['Cabin'] = df['Cabin'].str[0]
test['Cabin'] = test['Cabin'].str[0]
for var in categoricals:
    df[var].fillna('Missing',inplace = True)
    test[var].fillna('Missing',inplace = True)
df['family_size'] = df['SibSp'] + df['Parch']

test['family_size'] = test['SibSp'] + test['Parch']
import missingno as msno
msno.bar(df);
msno.bar(test);
from sklearn.model_selection import train_test_split
X = df.drop('Survived',axis = 1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape, X_test.shape
def rare_laebl(df):
    i = 1
    for col in categoricals:
        plt.figure(i)
        (df[col].value_counts() / len(df)).plot.bar(figsize=(8,4));
        plt.axhline(0.05,c = 'red',ls = '--')
        plt.title(f'Labels in {col}')
        sns.despine(bottom=True,left=True);
        i += 1
        
        
rare_laebl(X_train)
pd.DataFrame(X_train[categoricals].nunique(),columns=['unique']).T
from sklearn.preprocessing import KBinsDiscretizer
kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
kb.fit(X_train[['Age','Fare']])
X_train[['Age','Fare']] = kb.transform(X_train[['Age','Fare']])
X_test[['Age','Fare']] = kb.transform(X_test[['Age','Fare']])

test[['Age','Fare']] = kb.transform(test[['Age','Fare']])
X_train['Pclass'] = X_train['Pclass'].astype('object')
X_test['Pclass'] = X_test['Pclass'].astype('object')

test['Pclass'] = test['Pclass'].astype('object')
display(X_train.head(),test.head())
from feature_engine.categorical_encoders import MeanCategoricalEncoder
fre_encoder = MeanCategoricalEncoder(variables=['Embarked','Sex','Pclass','Cabin'])
fre_encoder.fit(X_train,y_train)
X_train = fre_encoder.transform(X_train)
X_test = fre_encoder.transform(X_test)

test = fre_encoder.transform(test)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier(n_jobs=4)
rf.fit(X_train,y_train)
prediction = rf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
roc_auc_score(y_test,prediction)
print('Roc_acu on the train: ',roc_auc_score(y_train,rf.predict(X_train)))
print('Roc_acu on the test: ',roc_auc_score(y_test,prediction))
# submission = rf.predict(test)
# submission
# submission = pd.DataFrame({'Survived':submission,'PassengerId':test_copy['PassengerId'].values})
# submission.set_index('PassengerId',inplace=True)
# submission.to_csv('Tiniancc_kaggle1.csv')
