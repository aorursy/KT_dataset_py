import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline



import featuretools as ft

from featuretools.primitives import *

from featuretools.variable_types import Numeric



from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import explained_variance_score

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, chi2



# model tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval



import warnings

warnings.filterwarnings("ignore")
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
df_optimum = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1)
target = df.Survived.loc[traindf.index]

df = df.drop(['SibSp','Parch','IsWomanOrBoy','WomanOrBoyCount','FamilySurvivedCount','WomanOrBoySurvived','Alone'], axis=1)

df['PassengerId'] = df.index

df.head()
es = ft.EntitySet(id = 'titanic_data')

es = es.entity_from_dataframe(entity_id = 'df', dataframe = df.drop(['Survived'], axis=1), 

                              variable_types = 

                              {

                                  'Embarked': ft.variable_types.Categorical,

                                  'Sex': ft.variable_types.Boolean,

                                  'Title': ft.variable_types.Categorical,

                                  'Family_Size': ft.variable_types.Numeric,

                                  'LastName': ft.variable_types.Categorical

                              },

                              index = 'PassengerId')
es = es.normalize_entity(base_entity_id='df', new_entity_id='Pclass', index='Pclass')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Sex', index='Sex')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Age', index='Age')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Fare', index='Fare')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Embarked', index='Embarked')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Title', index='Title')

es = es.normalize_entity(base_entity_id='df', new_entity_id='LastName', index='LastName')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Deck', index='Deck')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Family_Size', index='Family_Size')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Title_Sex', index='Sex')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Sex_LastName', index='LastName')

es = es.normalize_entity(base_entity_id='df', new_entity_id='Title_LastName', index='LastName')

es
primitives = ft.list_primitives()

pd.options.display.max_colwidth = 500

primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] == 'aggregation'].shape[0])
pd.set_option('max_columns',500)

pd.set_option('max_rows',500)
features, feature_names = ft.dfs(entityset = es, 

                                 target_entity = 'df', 

                                 max_depth = 2)

len(feature_names)
feature_names
features
# Determination categorical features

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

categorical_columns = []

cols = features.columns.values.tolist()

for col in cols:

    if features[col].dtype in numerics: continue

    categorical_columns.append(col)

categorical_columns
# Encoding categorical features

for col in categorical_columns:

    if col in features.columns:

        le = LabelEncoder()

        le.fit(list(features[col].astype(str).values))

        features[col] = le.transform(list(features[col].astype(str).values))
features.head(3)
train, test = features.loc[traindf.index], features.loc[testdf.index]

X_norm = MinMaxScaler().fit_transform(train)
# Threshold for removing correlated variables

threshold = 0.9



def highlight(value):

    if value > threshold:

        style = 'background-color: pink'

    else:

        style = 'background-color: palegreen'

    return style



# Absolute value correlation matrix

corr_matrix = features.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.style.applymap(highlight)
# Select columns with correlations above threshold

collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

features_filtered = features.drop(columns = collinear_features)

#features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]

print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])
FE_option0 = features.columns

FE_option1 = features_filtered.columns

print(len(FE_option0), len(FE_option1))
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train, target)

model = SelectFromModel(lsvc, prefit=True)

X_new = model.transform(train)

X_selected_df = pd.DataFrame(X_new, columns=[train.columns[i] for i in range(len(train.columns)) if model.get_support()[i]])

X_selected_df.shape
FE_option2 = X_selected_df.columns

FE_option2
lasso = LassoCV(cv=5).fit(train, target)

model = SelectFromModel(lasso, prefit=True)

X_new = model.transform(train)

X_selected_df = pd.DataFrame(X_new, columns=[train.columns[i] for i in range(len(train.columns)) if model.get_support()[i]])
FE_option3 = X_selected_df.columns

FE_option3
# Visualization from https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

# but to k='all'

bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(train, target)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(train.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']  #naming the dataframe columns

print(featureScores.nlargest(len(dfcolumns),'Score')) 
FE_option4 = featureScores[featureScores['Score'] > 1000]['Feature']

len(FE_option4)
FE_option5 = featureScores[featureScores['Score'] > 100]['Feature']

len(FE_option5)
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=50, step=10, verbose=5)

rfe_selector.fit(X_norm, target)
rfe_support = rfe_selector.get_support()

rfe_feature = train.loc[:,rfe_support].columns.tolist()

print(str(len(rfe_feature)), 'selected features')
FE_option6 = rfe_feature
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=200), threshold='1.25*median')

embeded_rf_selector.fit(train, target)
embeded_rf_support = embeded_rf_selector.get_support()

embeded_rf_feature = train.loc[:,embeded_rf_support].columns.tolist()

print(str(len(embeded_rf_feature)), 'selected features')
FE_option7 = embeded_rf_feature
test_rule = df_optimum.loc[testdf.index]

# The one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 

test_rule['Survived'] = (((test_rule.WomanOrBoySurvived <= 0.238) & (test_rule.Sex > 0.5) & (test_rule.Alone > 0.5)) | \

          ((test_rule.WomanOrBoySurvived > 0.238) & \

           ~((test_rule.WomanOrBoySurvived > 0.55) & (test_rule.WomanOrBoySurvived <= 0.633))))



# Saving the result

pd.DataFrame({'Survived': test_rule['Survived'].astype(int)}, \

             index=testdf.index).reset_index().to_csv('survived.csv', index=False)
acc_simple_rule = 92.7

LB_simple_rule = 0.83253
def RF (features_set,file):

    # Tuning Random Forest model for features "features_set", makes prediction and save it into file  

    train_fe = train[features_set]

    test_fe = test[features_set]

    random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 500]}, cv=5).fit(train_fe, target)

    random_forest.fit(train_fe, target)

    Y_pred = random_forest.predict(test_fe).astype(int)

    random_forest.score(train_fe, target)

    acc_random_forest = round(random_forest.score(train_fe, target) * 100, 2)

    pd.DataFrame({'Survived': Y_pred}, index=testdf.index).reset_index().to_csv(file, index=False)

    return acc_random_forest
acc0 = RF(FE_option0, 'survived_FT.csv')

acc1 = RF(FE_option1, 'survived_FE1_Pearson.csv')

acc2 = RF(FE_option2, 'survived_FE2_LinSVC.csv')

acc3 = RF(FE_option3, 'survived_FE3_Lasso.csv')

acc4 = RF(FE_option4, 'survived_FE4_Chi2_1000.csv')

acc5 = RF(FE_option5, 'survived_FE5_Chi2_100.csv')

acc6 = RF(FE_option6, 'survived_FE6_RFE_LogR.csv')

acc7 = RF(FE_option7, 'survived_FE7_RFE_RF.csv')
# After download solutions in Kaggle competition:

# 2019:

# LB0 = 0.74641

# LB1 = 0.73684

# LB2 = 0.75119

# LB3 = 0.75598

# LB4 = 0.76076

# LB5 = 0.74641

# LB6 = 0.74641

# LB7 = 0.74162

# 2020:

LB0 = 0.74162

LB1 = 0.73444

LB2 = 0.74401

LB3 = 0.74401

LB4 = 0.75358

LB5 = 0.73923

LB6 = 0.75358

LB7 = 0.73444
models = pd.DataFrame({

    'Model': ['Simple rule',

              'FT',

              'FT + Pearson correlation', 

              'FT + SelectFromModel with LinearSVC',

              'FT + SelectFromModel with Lasso', 

              'FT + SelectKBest with Chi-2 with Score > 1000',

              'FT + SelectKBest with Chi-2 with Score > 100',

              'FT + RFE with Logistic Regression',

              'FT + RFE with Random Forest'],

    

    'acc':  [acc_simple_rule, acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7],



    'LB':   [LB_simple_rule, LB0, LB1, LB2, LB3, LB4, LB5, LB6, LB7]})
models.sort_values(by=['acc', 'LB'], ascending=False)
models.sort_values(by=['LB', 'acc'], ascending=False)