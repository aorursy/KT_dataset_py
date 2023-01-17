!pip install autofeat
import pandas as pd

import numpy as np 



from sklearn.preprocessing import LabelEncoder

from autofeat import AutoFeatRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import explained_variance_score



# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



import warnings

warnings.filterwarnings("ignore")



pd.set_option('max_columns', 100)
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, testdf], axis=0, sort=False)
df.head(5)
#Thanks to:

# https://www.kaggle.com/mauricef/titanic

# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code

#

df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - \

                                    df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)

df['Alone'] = (df.WomanOrBoyCount == 0)



#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster

#"Title" improvement

df['Title'] = df['Title'].replace('Ms','Miss')

df['Title'] = df['Title'].replace('Mlle','Miss')

df['Title'] = df['Title'].replace('Mme','Mrs')

# Embarked

df['Embarked'] = df['Embarked'].fillna('S')

# Cabin, Deck

df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'



# Thanks to https://www.kaggle.com/erinsweet/simpledetect

# Fare

med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]

df['Fare'] = df['Fare'].fillna(med_fare)

#Age

df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

# Family_Size

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1



# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis

cols_to_drop = ['Name','Ticket','Cabin']

df = df.drop(cols_to_drop, axis=1)



df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)

df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)

df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)

df.Alone = df.Alone.fillna(0)

df.Alone = df.Alone*1
df.head(5)
target = df.Survived.loc[traindf.index]

df = df.drop(['Survived'], axis=1)

train, test = df.loc[traindf.index], df.loc[testdf.index]
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

features = train.columns.values.tolist()

for col in features:

    if train[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
# Encoding categorical features

for col in categorical_columns:

    if col in train.columns:

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))
train.head()
test.head()
model = AutoFeatRegression()

model
X_train_feature_creation = model.fit_transform(train.to_numpy(), target.to_numpy().flatten())

X_test_feature_creation = model.transform(test.to_numpy())

X_train_feature_creation.head()
# Number of new features

print('Number of new features -',X_train_feature_creation.shape[1] - train.shape[1])
df2 = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1)

train2, test2 = df2.loc[traindf.index], df2.loc[testdf.index]
X_train_feature_creation2 = model.fit_transform(train2.to_numpy(), target.to_numpy().flatten())

X_test_feature_creation2 = model.transform(test2.to_numpy())

X_train_feature_creation2.head()
# Number of new features

print('Number of new features -',X_train_feature_creation2.shape[1] - train2.shape[1])
test_x = df2.loc[testdf.index]



# The one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 

test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \

          ((test_x.WomanOrBoySurvived > 0.238) & \

           ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))



# Saving the result

pd.DataFrame({'Survived': test_x['Survived'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived.csv', index=False)
LB_simple_rule = 0.83253
# Linear Regression without Autofeat

model_LR = LinearRegression().fit(train,target.to_numpy().flatten())



# Linear Regression with Autofeat

model_Autofeat = LinearRegression().fit(X_train_feature_creation, target.to_numpy().flatten())
test['Survived_LR'] = np.clip(model_LR.predict(test),0,1)

test['Survived_AF'] = np.clip(model_Autofeat.predict(X_test_feature_creation),0,1)
def hyperopt_gb_score(params):

    clf = GradientBoostingClassifier(**params)

    current_score = cross_val_score(clf, X_train_feature_creation, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params = space_eval(space_gb, best)

params
# Gradient Boosting Classifier



gradient_boosting = GradientBoostingClassifier(**params)

gradient_boosting.fit(X_train_feature_creation, target)

Y_pred = gradient_boosting.predict(X_test_feature_creation).astype(int)

gradient_boosting.score(X_train_feature_creation, target)

acc_gradient_boosting = round(gradient_boosting.score(X_train_feature_creation, target) * 100, 2)

acc_gradient_boosting
# Saving the results

pd.DataFrame({'Survived': test['Survived_LR'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived_LR16.csv', index=False)

pd.DataFrame({'Survived': test['Survived_AF'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived_Autofeat16.csv', index=False)

pd.DataFrame({'Survived': Y_pred}, \

             index=testdf.index).reset_index().to_csv('survived_GBC16.csv', index=False)
# After download solutions in Kaggle competition:

LB_LR16 = 0.69377

LB_Autofeat16 = 0.67942

LB_GBC16 = 0.82296
# Linear Regression without Autofeat

model_LR2 = LinearRegression().fit(train2,target.to_numpy().flatten())



# Linear Regression with Autofeat

model_Autofeat2 = LinearRegression().fit(X_train_feature_creation2, target.to_numpy().flatten())
test2['Survived_LR'] = np.clip(model_LR2.predict(test2),0,1)

test2['Survived_AF'] = np.clip(model_Autofeat2.predict(X_test_feature_creation2),0,1)
def hyperopt_gb_score(params):

    clf = GradientBoostingClassifier(**params)

    current_score = cross_val_score(clf, X_train_feature_creation2, target, cv=10).mean()

    print(current_score, params)

    return current_score 

 

space_gb = {

            'n_estimators': hp.choice('n_estimators', range(100, 1000)),

            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            

        }

 

best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)

print('best:')

print(best)
params2 = space_eval(space_gb, best)

params2
# Gradient Boosting Classifier



gradient_boosting2 = GradientBoostingClassifier(**params2)

gradient_boosting2.fit(X_train_feature_creation2, target)

Y_pred2 = gradient_boosting2.predict(X_test_feature_creation2).astype(int)

gradient_boosting2.score(X_train_feature_creation2, target)

acc_gradient_boosting2 = round(gradient_boosting2.score(X_train_feature_creation2, target) * 100, 2)

acc_gradient_boosting2
# Saving the results

pd.DataFrame({'Survived': test2['Survived_LR'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived_LR3.csv', index=False)

pd.DataFrame({'Survived': test2['Survived_AF'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived_Autofeat3.csv', index=False)

pd.DataFrame({'Survived': Y_pred2}, \

             index=testdf.index).reset_index().to_csv('survived_GBC3.csv', index=False)
# After download solutions in Kaggle competition:

LB_LR3 = 0.67942

LB_Autofeat3 = 0.69377

LB_GBC3 = 0.83253
models = pd.DataFrame({

    'Model': ['Simple rule','Linear Regression without Autofeat', 'Linear Regression with Autofeat',

              'GradientBoostingClassifier with Autofeat'],

    

    'LB_for_16_features': [LB_simple_rule, LB_LR16, LB_Autofeat16, LB_GBC16],



    'LB_for_3opt_features': [LB_simple_rule, LB_LR3, LB_Autofeat3, LB_GBC3]})
models.sort_values(by=['LB_for_3opt_features', 'LB_for_16_features'], ascending=False)
models.sort_values(by=['LB_for_16_features', 'LB_for_3opt_features'], ascending=False)