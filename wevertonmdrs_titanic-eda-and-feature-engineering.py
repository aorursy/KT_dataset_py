import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn import metrics

import category_encoders as ce

import plotly.express as px

import itertools

X_train = pd.read_csv('../input/titanic/train.csv')

X_test = pd.read_csv('../input/titanic/test.csv')



X_train.head()
X_train.describe()
X_survivors = X_train[X_train['Survived'] == 1]

X_n_survivors = X_train[X_train['Survived'] == 0]
X_survivors_sex = X_survivors.Sex.value_counts()

X_survivors_sex = pd.DataFrame({'Sex' : X_survivors_sex.index, 

                                'count' : X_survivors_sex.values})



X_n_survivors_sex = X_n_survivors.Sex.value_counts()

X_n_survivors_sex = pd.DataFrame({'Sex' : X_n_survivors_sex.index, 

                                'count' : X_n_survivors_sex.values})



###################



X_survivors_pclass = X_survivors.Pclass.value_counts()

X_survivors_pclass = pd.DataFrame({'Pclass' : X_survivors_pclass.index, 

                                'count' : X_survivors_pclass.values})



X_n_survivors_pclass = X_n_survivors.Pclass.value_counts()

X_n_survivors_pclass = pd.DataFrame({'Pclass' : X_n_survivors_pclass.index, 

                                'count' : X_n_survivors_pclass.values})



##################



plt.figure(1, figsize=(20,10))

the_grid = GridSpec(2, 2)



cmap = plt.get_cmap('Spectral')

colors = [cmap(i) for i in np.linspace(0, 2, 8)]



plt.title('Sex percentage of survivors and non-survivors')

plt.subplot(the_grid[0, 1], aspect = 1, title='Survived')

plt.pie(X_survivors_sex['count'], explode = (0.1, 0.0), labels = X_survivors_sex['Sex'], 

                     autopct = '%1.1f%%', shadow = False, colors = colors)



plt.subplot(the_grid[0, 0], aspect = 1, title='N_Survived')

plt.pie(X_n_survivors_sex['count'], explode = (0.1, 0.0), labels = X_n_survivors_sex['Sex'], 

                     autopct = '%1.1f%%', shadow = False, colors = colors)





plt.subplot(the_grid[1, 1], aspect = 1, title = 'Survived')

plt.pie(X_survivors_pclass['count'], explode = (0.1, 0.1, 0.1), 

        labels = X_survivors_pclass['Pclass'], autopct = '%1.1f%%', 

        shadow = False, colors = colors)



plt.subplot(the_grid[1, 0], aspect = 1, title='N_Survived')

plt.pie(X_n_survivors_pclass['count'], explode = (0.1, 0.1, 0.1), 

        labels = X_n_survivors_pclass['Pclass'], 

        autopct = '%1.1f%%', shadow = False, colors = colors)
def family_size(row):

    if (row.SibSp + row.Parch) == 0:

        return 'alone'

    elif (row.SibSp + row.Parch) == 1 or (row.SibSp + row.Parch) == 2:

        return 'small'

    elif (row.SibSp + row.Parch) == 3 or (row.SibSp + row.Parch) == 4:

        return 'medium'

    elif (row.SibSp + row.Parch) >= 5:

        return 'large'

    



def is_alone(row):

    if (row.SibSp + row.Parch) == 0:

        return 'yes'

    elif (row.SibSp + row.Parch) > 0:

        return 'no'

    

    

X_train['FamilySize'] = X_train.apply(lambda row: family_size(row), axis = 1)

X_train['IsAlone'] = X_train.apply(lambda row: is_alone(row), axis = 1)
X_train.head()
fig, (ax1,ax2) = plt.subplots(figsize = [12,12], nrows = 1, ncols = 2)

plt.subplots_adjust(left = 0, bottom = None, right = 1, top = 0.5, 

                    wspace = 0.2, hspace = None)



plt.figure

sns.countplot(x = 'Survived', hue = 'FamilySize', data = X_train, ax = ax1)



ax1.set_title('Relation between Family Size and its survivor')

ax1.set_xlabel('Survived', fontsize = 10)

ax1.set_ylabel('Quantity', fontsize = 10)



sns.kdeplot(data=X_train['Age'], shade = True, ax = ax2)



ax2.set_title('Distribution of the age passengers', fontsize = 10)

ax2.set_ylabel('Distribution function',fontsize = 10)

ax2.set_xlabel('Age',fontsize = 10)

def sex(row):

    if (row.Sex) == 'female':

        return 1

    elif (row.Sex) == 'male':

        return 0

    else:

        return 0.5

    

X_train_corr = X_train.copy()

# X_train_corr.drop(['Sex'], axis = 1, inplace = True)

X_train_corr['Sex_new'] = X_train_corr.apply(lambda row: sex(row), axis = 1)
X_train_corr.head()
X_correlation = X_train_corr.corr()



plt.figure(figsize=(8,8))

sns.heatmap(X_correlation, annot = True, cmap = 'Blues')

plt.title('Correlation among features')

plt.show()
# Get names of columns with missing values

cols_with_missing = X_train.isnull().sum()



print('Features with missing data:')

print(cols_with_missing[cols_with_missing > 0])
X_train.loc[X_train.Age.isnull(), 'Age'] = X_train.Age.median()

X_train.loc[X_train.Embarked.isnull(), 'Embarked'] = X_train.Embarked.describe().top



X_train.drop(['Cabin', 'PassengerId'], axis = 1, inplace = True)
# # # Remove missing columns

# # cols_with_missing = [col for col in X_train_ref.columns 

# #                      if X_train_ref[col].isnull().sum()]



# # X_train_ref.drop(cols_with_missing, axis = 1, inplace = True)

# # # X_valid_ref.drop(cols_with_missing, axis = 1, inplace = True)



# # Columns that will be encoded

# low_cardinality_cols = [cname for cname in X_train_ref.columns 

#                         if X_train_ref[cname].nunique() < 10 

#                         and X_train_ref[cname].dtype == 'object']



# # Columns of numerical data

# nume_cols = [cname for cname in X_train_ref.columns if X_train_ref[cname].dtype

#                                                     in ['int64', 'float64']]



# my_cols = low_cardinality_cols + nume_cols

# X_train_ref = X_train_ref[my_cols].copy()

# X_train = X_train[my_cols].copy()

# # X_valid = X_valid_ref[my_cols].copy()

# # X_test = test[my_cols].copy()
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
X_train.describe(include=[np.object])
X_train.drop(['Name', 'Ticket'], axis = 1, inplace = True)
X_train_ref = X_train.copy()

X_train_ref.drop(['FamilySize', 'IsAlone'], axis = 1, inplace = True)

X_train_ref.head()
#Aplicando o One-Hot Encoding



# object_cols = ['Sex', 'Embarked', 'FamilySize', 'IsAlone']

object_cols = ['Sex', 'Embarked']



OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_ref[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train_ref.index



# Remove categorical columns (will replace with one-hot encoding)

X_train_ref.drop(object_cols, axis=1, inplace = True)



# Add one-hot encoded columns to numerical features

X_train_ref = pd.concat([X_train_ref, OH_cols_train], axis=1)
def evaluate_model(my_pipeline, X, y):

    

    # Predicting the data    

    accuracy = 1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='accuracy')

    

    auc = 1 * cross_val_score(my_pipeline, X, y,

                          cv=5,

                          scoring='roc_auc')

    

    return accuracy.mean(), auc.mean()
y = X_train_ref['Survived']

X = X_train_ref.drop(['Survived'], axis = 1)





RFC = RandomForestClassifier(n_estimators = 900, criterion = 'gini', max_depth = 10, 

                             n_jobs = 4, random_state = 0)



accuracy_RF, auc_RF = evaluate_model(RFC, X, y)



print("Accuracy:\t {}".format(accuracy_RF))

print("AUC:\t\t {}".format(auc_RF))
# Definição do modelo

param = {'num_leaves': 64, 'objective': 'binary', 

             'seed': 7}



model_lgb = lgb.LGBMClassifier(num_leaves = 64, n_estimators = 40, objective = 'binary', 

                               seed = 7,

                              n_jobs = 4)





# Métricas de avaliação

accuracy_lgb, auc_lgb = evaluate_model(model_lgb, X, y)



print("Accuracy:\t {}".format(accuracy_lgb))

print("AUC:\t\t {}".format(auc_lgb))

parcial_results = pd.DataFrame({'Model': ('RandomForest Classifier', 'LGBM Classifier'), 

                             'Accuracy': (accuracy_RF, accuracy_lgb), 

                             'AUC': [auc_RF, auc_lgb], 

                               'Encoder' : ('One-Hot Encoder', 'One-Hot Encoder')})



parcial_results
X_train.head()
cat_features = ['Pclass', 'Sex', 'FamilySize', 'Age']

interactions = pd.DataFrame(index = X_train.index)



# Iterate through each pair of features, combine them into interaction features

for col1, col2 in itertools.combinations(cat_features, 2):

    new_col_name = '_'.join([col1, col2])



        # Convert to strings and combine

    new_values = X_train[col1].map(str) + "_" + X_train[col2].map(str)

    interactions[new_col_name] = new_values



X_train = X_train.join(interactions)



X_train.head()
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



X_train_CBE = X_train.copy()



target_enc = ce.CatBoostEncoder(cols = object_cols)



target_enc.fit(X_train_CBE[object_cols], X_train_CBE['Survived'])



# Transform the features, rename columns with _cb suffix, and join to dataframe

X_train_CBE = X_train_CBE.join(target_enc.transform(X_train_CBE[object_cols]).add_suffix('_cb'))

X_train_CBE.drop(object_cols, axis = 1, inplace = True)
X_train_CBE.head()
X_train_OHE = X_train.copy()



OH_encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_OHE[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train_OHE.index



# Remove categorical columns (will replace with one-hot encoding)

X_train_OHE.drop(object_cols, axis=1, inplace = True)



# Add one-hot encoded columns to numerical features

X_train_OHE = pd.concat([X_train_OHE, OH_cols_train], axis=1)
X_train_OHE.head()
y_tuned_CBE = X_train_CBE['Survived']

X_tuned_CBE = X_train_CBE.drop(['Survived'], axis = 1)



y_tuned_OHE = X_train_OHE['Survived']

X_tuned_OHE = X_train_OHE.drop(['Survived'], axis = 1)





RFC = RandomForestClassifier(n_estimators = 900, criterion = 'gini', max_depth = 10, 

                             n_jobs = 4, random_state = 0)



accuracy_RF_tuned_CBE, auc_RF_tuned_CBE = evaluate_model(RFC, X_tuned_CBE, 

                                                                     y_tuned_CBE)

accuracy_RF_tuned_OHE, auc_RF_tuned_OHE = evaluate_model(RFC, X_tuned_OHE, 

                                                                     y_tuned_OHE)



print("Accuracy tuned with CBE:\t {}".format(accuracy_RF_tuned_CBE))

print("AUC tuned with CBE:\t\t {}".format(auc_RF_tuned_CBE))



print("Accuracy tuned with OHE:\t {}".format(accuracy_RF_tuned_OHE))

print("AUC tuned with OHE:\t\t {}".format(auc_RF_tuned_OHE))
# Definição do modelo

param = {'num_leaves': 64, 'objective': 'binary', 

             'seed': 7}



model_lgb = lgb.LGBMClassifier(num_leaves = 64, n_estimators = 40, objective = 'binary', 

                               seed = 7,

                              n_jobs = 4)





# Métricas de avaliação

accuracy_lgb_tuned_CBE, auc_lgb_tuned_CBE = evaluate_model(model_lgb, X_tuned_CBE, 

                                                               y_tuned_CBE)

accuracy_lgb_tuned_OHE, auc_lgb_tuned_OHE = evaluate_model(model_lgb, X_tuned_OHE, 

                                                               y_tuned_OHE)



print("Accuracy tuned with CBE:\t {}".format(accuracy_lgb_tuned_CBE))

print("AUC tuned with CBE:\t\t {}".format(auc_lgb_tuned_CBE))



print("Accuracy tuned with OHE:\t {}".format(accuracy_lgb_tuned_OHE))

print("AUC tuned with OHE:\t\t {}".format(auc_lgb_tuned_OHE))
final_results = pd.DataFrame({'Model': ('RandomForest Classifier', 

                                        'RandomForest Classifier tuned with OHE',

                                        'RandomForest Classifier tuned with CBE',

                                        'LGBM Classifier', 

                                        'LGBM Classifier tuned with OHE',

                                        'LGBM Classifier tuned with CBE'), 

                             'Accuracy': (accuracy_RF, accuracy_RF_tuned_OHE,

                                          accuracy_RF_tuned_CBE,

                                          accuracy_lgb, accuracy_lgb_tuned_OHE,

                                          accuracy_lgb_tuned_CBE), 

                             'AUC': (auc_RF, auc_RF_tuned_OHE, auc_RF_tuned_CBE,

                                     auc_lgb, auc_lgb_tuned_OHE, auc_lgb_tuned_CBE),

                             'Encoder' : ('One-Hot Encoder without feat. engineering',

                                         'One-Hot Encoder with feat. engineering',

                                          'CatBoost Encoder with feat. engineering',

                                         'One-Hot Encoder without feat. engineering',

                                          'One-Hot Encoder with feat. engineering',

                                         'CatBoost Encoder with feat. engineering',

                                          )})



final_results