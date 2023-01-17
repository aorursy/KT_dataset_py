# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import xgboost as xgb


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.drop(['PassengerId'], axis = 1, inplace = True)
train.head()
train.isnull().sum()
sns.boxplot(y = train['Fare'], x = train['Survived'])
def isalone(column):
    if column == 0:
        return 0
    else:
        return 1

def richfemale(columns):
    if columns[0] > 1:
        if columns[1] == 1:
            if columns[2] == 1:
                return 1
            else:
                return 2
        else:
            return 3
    else:
        return 0
    
def parch_corr(columns):
    if pd.isnull(columns[0]) == True:
        if columns[1] == 0:
            return 0
        else:
            return 0.5
    else:
        return columns[0]

train['IsAlone'] = train['Parch'].apply(isalone)

def embarkcorr(column):
    if column == 'C':
        return 1
    elif column == 'S':
        return 2
    elif column == 'Q':
        return 3
    else:
        return 0
train['Embarked'] = train['Embarked'].apply(embarkcorr)

def agedet(columns):
    Age = columns[0]
    Pclas = columns[1]
    Embark = columns[2]
    if np.isnan(Age) == True:
        if Embark == 3:
            return 50
        elif Pclas == 1:
            return 38
        elif Pclas == 2:
            return 30
        else:
            return 25
    else:
        return Age
train['Age'] = train[['Age', 'Pclass', 'Embarked']].apply(agedet, axis = 1)
    
def agecat(column):
    if column <= 15:
        return 0
    elif 15<column<40:
        return 1
    else:
        return 2

train['Age'] = train['Age'].apply(agecat)

train['Cabin'] = train['Cabin'].fillna(0)

def cabin_crr(columns):
        if columns == 0:
            return columns
        else:
            column = columns[0]
            if column == 'B':
                return 2
            elif column == 'C':
                return 1
            elif column == 'D':
                return 3
            elif column == 'F':
                return 6
            elif column == 'E':
                return 4
            elif column == 'A':
                return 5
            elif column == 'G':
                return 7
            else:
                return 8 
        
train['Cabin'] = train['Cabin'].apply(cabin_crr)

def Pclasscorr(column):
    if column == 1:
        return '1'
    elif column == 2:
        return '2'
    else:
        return '3'

train['Pclass'] = train['Pclass'].apply(Pclasscorr)
pclass = pd.get_dummies(train['Pclass'])
train = pd.concat([train, pclass], axis = 1)


def SibSpcorr(columns):
    if pd.isnull(columns[0]) == True:
        if columns[1] == 1:
            return 0.5
        else:
            return 0
    else:
        return columns[0]

def Farecorr(columns):
    if pd.isnull(columns[0]) == True:
        if columns[1] == 1:
            return 55
        elif columns[1] == 2:
            return 20
        else:
            return 12
    else:
        return columns[0]
    
def Farecat(column):
    if column<=30:
        return 0
    elif 30<column<=70:
        return 1
    else:
        return 2
train['Fare'] = train['Fare'].apply(Farecat)

def namesplit(columns):
    a = columns[0].split()
    if a[1] == 'Mr.':
        return 0
    elif a[1] == 'Mrs.':
        return 2
    elif a[1] == 'Miss.':
        return 1
    elif a[1] == 'Master.':
        return 3
    else:
        return 4

Sex = pd.get_dummies(train['Sex'])
train = pd.concat([train, Sex], axis = 1)
train.drop(['Sex', 'Ticket'], axis = 1, inplace = True)

train['Name'] = train[['Name', 'female']].apply(namesplit, axis = 1)

    
#Q1 = train.quantile(0.15)
#Q3 = train.quantile(0.85)
#IQR = Q3 - Q1

#train = train[(train >= Q1 - 1.5 * IQR) & (train <= Q3 + 1.5 *IQR)] 

#train['SibSp'] = train[['SibSp', 'IsAlone']].apply(SibSpcorr, axis = 1)
#train['Fare'] = train[['Fare', 'Pclass']].apply(Farecorr, axis = 1)
#train['Parch'] = train[['Parch', 'female']].apply(parch_corr, axis = 1)
#train['Cabin'] = train['Cabin'].apply(cabin_crr)
#train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mean())
#train.drop(['Cabin'], axis = 1, inplace = True)
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mean())
train['richfemale'] = train[['Fare', 'Pclass', 'female']].apply(richfemale, axis = 1)
train.drop('Pclass', axis = 1, inplace = True)

#train.drop(['Pclass', 'male', 'Parch', 'Age'], axis = 1, inplace = True)

#scaler= StandardScaler()
#survived = train['Survived']
#train.drop(['Survived'], axis = 1, inplace = True)
#scaler.fit(train)
#scaleddata = scaler.transform(train)
#train = pd.DataFrame(scaleddata, columns = train.columns)
#train = pd.concat([train, survived], axis = 1)
train
plt.figure(figsize=(20,10))
sns.heatmap(train.corr(), annot = True)
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                         train['Survived'], test_size=0.3, random_state = 42)
n_estimators = [50, 250, 500, 750, 1000, 1500, 3000, 5000]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [10, 25, 40, 50]
max_depth.append(None)
min_samples_split = [2, 5, 15, 20]
min_samples_leaf = [1, 2, 5, 10]

grid_param = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 
              'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

RF = RandomForestClassifier(random_state = 42)

RF_random = RandomizedSearchCV(estimator = RF, param_distributions = grid_param, n_iter = 500,
                              cv = 5, verbose = 2, random_state = 42, n_jobs = -1)
RF_random.fit(X_train, y_train)
print(RF_random.best_params_)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).columns
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
model
#sns.heatmap(train.corr(), annot = True)
train['Cabin'].unique()
sns.boxplot(y=train['Cabin'], x=train['Survived'])
k = 1
deger = 0
degerson = 0 
a = 0
b = 0
while degerson < 0.9:
    X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['Survived'], axis = 1), 
                                                         train['Survived'], test_size=0.2)

#model = RandomForestClassifier(n_estimators = 500, min_samples_split = 15, min_samples_leaf = 1, max_features = 'log2', 
     #                              max_depth = 40)
    #model = KNeighborsClassifier(n_neighbors = 35, weights = 'distance')
    #model = GradientBoostingClassifier(n_estimators = 100, loss = 'exponential')
    #model = MLPClassifier(hidden_layer_sizes=(200, ), learning_rate = 'adaptive')
    model = xgb.XGBClassifier(
        gamma=1,
        objective = 'binary:logistic',
        learning_rate=0.2,
        max_depth=3,
        n_estimators=1000,                                                                    
        subsample=0.8)
    model.fit(X_train, y_train) 
    #predictions = model.predict(X_test)
    predict = model.predict(X_valid)
    a = accuracy_score(y_valid, model.predict(X_valid))

    #deger = f1_score(y_test, predictions)
    
    X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 
                                                         train['Survived'], test_size=0.4)
    deger = cross_validate(model, X_test, y_test, cv = 20)
    degerson = deger['test_score'].mean()
    k += 1
    if k>10000:
        break
degerson    
#accuracy_score(y_test, model.predict(X_test))
cfm = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(cfm, annot = True, fmt = 'd')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test['Embarked'] = test['Embarked'].apply(embarkcorr)
test['Age'] = test[['Age', 'Pclass', 'Embarked']].apply(agedet, axis = 1)
test['Age'] = test['Age'].apply(agecat)
test['IsAlone'] = test['Parch'].apply(isalone)
test['Pclass'] = test['Pclass'].apply(Pclasscorr)
pclasst = pd.get_dummies(test['Pclass'])
test = pd.concat([test, pclasst], axis = 1)
test['Cabin'] = test['Cabin'].fillna(0)
test['Cabin'] = test['Cabin'].apply(cabin_crr)
test['Fare'] = test['Fare'].apply(Farecat)
Sex = pd.get_dummies(test['Sex'])
test = pd.concat([test, Sex], axis = 1)
df = test['PassengerId']
test.drop(['PassengerId', 'Sex', 'Ticket'], axis = 1, inplace= True)

test['Name'] = test[['Name','female']].apply(namesplit, axis = 1)
test['Embarked'] = test['Embarked'].fillna('backfill')

#Q1 = test.quantile(0.15)
#Q3 = test.quantile(0.85)
#IQR = Q3 - Q1

#test = test[(test >= Q1 - 1.5 * IQR) & (test <= Q3 + 1.5 *IQR)] 

test['richfemale'] = test[['Fare', 'Pclass', 'female']].apply(richfemale, axis = 1)
test.drop('Pclass', axis = 1, inplace = True)
#scalert= StandardScaler()
#scalert.fit(test)
#scaleddatat = scalert.transform(test)
#test = pd.DataFrame(scaleddatat, columns = test.columns)
#test.isnull().sum()
test
predictt = model.predict(test)
df = pd.DataFrame(df, columns=['PassengerId'])
dff = pd.DataFrame(predictt, columns=['Survived'])
sonuc = pd.concat([df, dff], axis = 1)
sonuc.to_csv('/kaggle/working/predict.csv', index = False)
