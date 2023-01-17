# Download data and preparing to prediction (including FE) 

# Thanks to: https://www.kaggle.com/mauricef/titanic

import pandas as pd

import numpy as np 

traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, testdf], axis=0, sort=False)

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df['Alone'] = (df.WomanOrBoyCount == 0)

train_y = df.Survived.loc[traindf.index]

df = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1)



test_x = df.loc[testdf.index]
# My upgrade - the one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 

test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \

          ((test_x.WomanOrBoySurvived > 0.238) & \

           ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))
# Saving the result

pd.DataFrame({'Survived': test_x['Survived'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived.csv', index=False)
import pandas as pd

import numpy as np 

import graphviz

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier, export_graphviz



import warnings

warnings.filterwarnings("ignore")
# Download data and preparing to prediction (including FE) 

traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, testdf], axis=0, sort=False)
# FE

df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))

df['LastName'] = df.Name.str.split(',').str[0]

family = df.groupby(df.LastName).Survived

df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())

df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)

df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())

df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)

df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)

df['Alone'] = (df.WomanOrBoyCount == 0)
df
train_y = df.Survived.loc[traindf.index]

data = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, \

                  df.Sex.replace({'male': 0, 'female': 1})], axis=1)

train_x, test_x = data.loc[traindf.index], data.loc[testdf.index]

train_x.head(5)
# Tuning the DecisionTreeClassifier by the GridSearchCV

parameters = {'max_depth' : np.arange(2, 5, dtype=int),

              'min_samples_leaf' :  np.arange(2, 5, dtype=int)}

classifier = DecisionTreeClassifier(random_state=1000)

model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=2, n_jobs=-1)

model.fit(train_x, train_y)

best_parameters = model.best_params_

print(best_parameters)
model=DecisionTreeClassifier(max_depth = best_parameters['max_depth'], 

                             random_state = 1000)

model.fit(train_x, train_y)
# plot tree

dot_data = export_graphviz(model, out_file=None, feature_names=train_x.columns, class_names=['0', '1'], 

                           filled=True, rounded=False,special_characters=True, precision=3) 

graph = graphviz.Source(dot_data)

graph 
# # Prediction by the DecisionTreeClassifier

y_pred = model.predict(test_x).astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# The one line of the code for prediction : LB = 0.80382 (Titanic Top 6%) 

test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \

                      ((test_x.WomanOrBoySurvived > 0.238) & \

                       ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))

y_pred = test_x['Survived'].astype(int)

print('Mean =', y_pred.mean(), ' Std =', y_pred.std())
# Saving the result

pd.DataFrame({'Survived': y_pred}, index=testdf.index).reset_index().to_csv('submission.csv', index=False)