import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# ReadIn the training data

titanic_train = pd.read_csv("../input/titanic/train.csv")

print (titanic_train.info())
titanic_train.head()
titanic_train.isnull().sum()
print(f'All column names: {titanic_train.columns}')

X_train = titanic_train.copy()

y_train = X_train.pop('Survived')

print(f'Training data column names: {X_train.columns}')

print(f'Training label: {y_train.name}')
def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if str.find(big_string, substring) != -1:

            return substring

    # print (big_string)

    return np.nan



def replace_titles(x):

    title=x['salut']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady', 'Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
# Split Name and extract the salutation



X_train['salut'] = X_train['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0].str.strip()

print("Unique values from salut - training dataset:\n", X_train['salut'].unique(), "\n")



print ("salut Before:")

print (X_train['salut'].value_counts(), "\n")



# X_train.drop(['firstname', 'last_name', 'lastname', 'lastname1'], axis=1, inplace=True)

print (X_train.columns, "\n")



X_train['salut']=X_train.apply(replace_titles, axis=1)

print ("salut After:")

print (X_train['salut'].value_counts())



Age_salut = pd.crosstab(X_train.Age, X_train.salut)

Age_salut.tail(10)
# Imputing Age - We are using the 'salut' feature to group the respondent to impute the age

print ("Null values for Age before imputation: ", X_train['Age'].isnull().sum())

X_train['Age'] = X_train.groupby('salut').Age.transform(lambda x: x.fillna(x.mean()))

print ("Null values for Age after imputation: ", X_train['Age'].isnull().sum())
# Imputing Cabin - This cannot be imputed as there is no logic and hence we fill the NAs with 'Null' string

print("Null values for Cabin before imputation: ", X_train['Cabin'].isnull().sum())



print("Value Counts of Cabin - Before")

print (X_train['Cabin'].value_counts(dropna = False))



X_train['Cabin'] = X_train['Cabin'].fillna('Null')



print("Value Counts of Cabin - After")

print (X_train['Cabin'].value_counts(dropna = False))
# Imputing the whole dataset just in case there are any furhter missing values

X_train = X_train.fillna(method='ffill').fillna(method='bfill')

print("Null values after imputation: ")

print(X_train.isnull().sum())
# Deck

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Null']

X_train['Deck']=X_train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))



X_train['Deck'].value_counts()
# Family Size and Fare per Passenger

X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1

print(X_train['FamilySize'].value_counts())
# Drop features which are unique across respondents as they are not useful

X_train.drop(['Name', 'PassengerId'], axis=1, inplace=True)



# One Hot Encoding - To convert categorical to binary data

X_train_dummies = pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'Cabin', 'Embarked', 'salut', 'Ticket', 'Deck'])



print ("Shape of training dataset after One Hot Encoding: ", X_train_dummies.shape)

X_train_dummies.head()
# ReadIn the test data



titanic_test = pd.read_csv("../input/titanic/test.csv")
titanic_test.info()
titanic_test.head()
titanic_test.isnull().sum()
X_test = titanic_test.copy()



# Split Name and extract the salutation

X_test['salut'] = X_test['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0].str.strip()

print("Unique values from salut - test dataset:\n", X_test['salut'].unique())



print ("salut Before:")

print (X_test['salut'].value_counts())



print (X_test.columns)



X_test['salut']=X_test.apply(replace_titles, axis=1)



print ("salut After:")

print (X_test['salut'].value_counts())



Age_salut_test = pd.crosstab(X_test.Age, X_test.salut)

print(Age_salut_test.head(6))

print(Age_salut_test.tail(6))
# Imputing missing values - Test Data

print ("Null values for Age before imputation: ", X_test['Age'].isnull().sum())

X_test['Age'] = X_test.groupby('salut').Age.transform(lambda x: x.fillna(x.mean()))

print ("Null values for Age after imputation: ", X_test['Age'].isnull().sum(), "\n")



print ("Null values for Fare before imputation: ", X_test['Fare'].isnull().sum())

X_test['Fare'] = X_test.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.median()))

print ("Null values for Fare after imputation: ", X_test['Fare'].isnull().sum(), "\n")



print("Null values for Cabin before imputation: ", X_train['Cabin'].isnull().sum())

X_test['Cabin'] = X_test['Cabin'].fillna('Null')

print("Null values for Cabin after imputation: ", X_train['Cabin'].isnull().sum(), "\n")



# Imputing the whole dataset just in case there are any furhter missing values

X_test = X_test.fillna(method='ffill').fillna(method='bfill')



print("Null values after imputation: ")

print(X_test.isnull().sum())
## Feature Engineering

# Deck

X_test['Deck']=X_test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

print(X_test['Deck'].value_counts(), "\n")



# Family Size and Fare per Passenger

X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1

# X_test['FarePerPassenger'] = X_test['Fare']/(X_test['FamilySize'] + 1)



print(X_test['FamilySize'].value_counts(), "\n")

# print()

# print(X_test['FarePerPassenger'].value_counts(), "\n")
## Converting Categorical and String features into Numeric



# Drop features which are unique across respondents as they are not useful

X_test.drop(['PassengerId'], axis=1, inplace=True)



# One Hot Encoding - To convert categorical to binary data

X_test_dummies = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'Cabin', 'Embarked', 'salut', 'Ticket', 'Deck'])

print ("Shape of test dataset after One Hot Encoding: ", X_test_dummies.shape)

print (X_test_dummies.head())
print ("Shape of training dataset after One Hot Encoding: ", X_train_dummies.shape)

print ("Shape of test dataset after One Hot Encoding: ", X_test_dummies.shape)
# Align the Train and Test datset for One Hot Encoding 

X_train_final, X_test_final = X_train_dummies.align(X_test_dummies, join='left', axis=1)

print (X_train_final.shape)

print (X_test_final.shape)



for col in (col for col in X_test_final.columns if X_test_final[col].isnull().any()):

    X_test_final[col] = 0



print(X_test_final.isnull().sum())
X_train_final[['Age', 'SibSp', 'Fare', 'FamilySize', 'Pclass_3', 'Sex_female', 'Sex_male']].describe().T
fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('Original Distributions')



column_list = ['Age', 'SibSp', 'Fare', 'FamilySize', 'Pclass_3', 'Sex_female', 'Sex_male']



X_train_final[column_list].apply(sns.kdeplot, ax = ax1);
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



scaler = MinMaxScaler()

# scaler = RobustScaler()

# scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train_final)

# print(X_train_scaled.mean(axis=0))

X_test_scaled = scaler.transform(X_test_final)



df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_final.columns)
df_X_train_scaled[['Age', 'SibSp', 'Fare', 'FamilySize', 'Pclass_3', 'Sex_female', 'Sex_male']].describe().T
# plot original distribution plot

fig, (ax2) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('Scaled Distributions')



column_list = ['Age', 'SibSp', 'Fare', 'FamilySize', 'Pclass_3', 'Sex_female', 'Sex_male']



df_X_train_scaled[column_list].apply(sns.kdeplot, ax = ax2);
# split the data into train and evaluation data

from sklearn.model_selection import train_test_split



# X, val_X, y, val_y = train_test_split(X_train_final, y_train, train_size=0.7, test_size=0.3, random_state=123, stratify=y_train)



# Applying scaled data

X, val_X, y, val_y = train_test_split(df_X_train_scaled, y_train, train_size=0.7, test_size=0.3, random_state=123, stratify=y_train)



print (X.shape)

print (val_X.shape)

print('All:', np.bincount(y_train) / float(len(y_train)) * 100.0)

print('Training:', np.bincount(y) / float(len(y)) * 100.0)

print('Test:', np.bincount(val_y) / float(len(val_y)) * 100.0)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import StackingClassifier

from sklearn.model_selection import GridSearchCV
param_grid_rf = {'n_estimators': [10, 20, 30], 

                 'min_samples_split': [2, 3, 4, 5],

                 'max_leaf_nodes':[90, 900, 9000, None]}



model_random_forest_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid_rf, verbose=3, n_jobs=-1)

model_random_forest_grid.fit(X, y)



results_grid_rf = pd.DataFrame(model_random_forest_grid.cv_results_)



print ("Parameters: ", model_random_forest_grid.get_params)



print("\nGridSearchCV best score - Random Forest: ", model_random_forest_grid.best_score_)

print("\nGridSearchCV best params - Random Forest: ", model_random_forest_grid.best_params_)

print("\nGridSearchCV best estimator - Random Forest: ", model_random_forest_grid.best_estimator_)



print ("\nGridSearchCV Score for validation data - Random Forest: ", model_random_forest_grid.score(val_X, val_y))
print(results_grid_rf.shape)

results_grid_rf.head(3)
current_palette = sns.color_palette("Set2", 4)



plt.figure(figsize=(12,8))



sns.lineplot(data=results_grid_rf,

             x='param_n_estimators',

             y='mean_test_score',

             hue='param_min_samples_split',

             palette=current_palette,

             marker='o')



plt.show()
param_grid_kNN = {'n_neighbors': [1, 5, 10, 15], 

                  'weights': ['uniform', 'distance'], 

                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}



model_knearest_neighbors_grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_kNN, verbose=3, n_jobs=-1)

model_knearest_neighbors_grid.fit(X, y)



results_grid_kNN = pd.DataFrame(model_knearest_neighbors_grid.cv_results_)



# print ("Parameters: ", model_knearest_neighbors_grid.get_params)



print("\nGridSearchCV best score - k-Nearest Neighbor: ", model_knearest_neighbors_grid.best_score_)

print("\nGridSearchCV best params - k-Nearest Neighbor: ", model_knearest_neighbors_grid.best_params_)

print("\nGridSearchCV best estimator - k-Nearest Neighbor: ", model_knearest_neighbors_grid.best_estimator_)



print ("\nGridSearchCV Score for validation data - k-Nearest Neighbor: ", model_knearest_neighbors_grid.score(val_X, val_y))
print(results_grid_kNN.shape)

results_grid_kNN.head(3)
current_palette = sns.color_palette("Set2", 2)



plt.figure(figsize=(12,8))



sns.lineplot(data=results_grid_kNN,

             x='param_n_neighbors',

             y='mean_test_score',

             hue='param_weights',

             palette=current_palette,

             marker='o')



plt.show()
estimators = [

    ('rf', model_random_forest_grid.best_estimator_),

    ('kNN', model_knearest_neighbors_grid.best_estimator_)

]

model_stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)

# model_stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
model_stacking_clf.fit(X, y)

print("Parameters for Stacking Classifier:")

model_stacking_clf.get_params()
predictions = model_stacking_clf.predict(val_X)

predict_proba = model_stacking_clf.predict_proba(val_X)



print ("Count for validation data - actual: ", np.bincount(val_y))

print ("Count for validation data - prediction: ", np.bincount(predictions), "\n")
score_val_dataset = model_stacking_clf.score(val_X, val_y)

print ("Score for validation data - Stacked Classifier: ", score_val_dataset, "\n")
print("Predictions: ", predictions[0:6], "\n")

print("Prediction Probabilities:\n", predict_proba[0:6])
from sklearn.metrics import classification_report



print("Classification Report - Validation Data")

print(classification_report(val_y, model_stacking_clf.predict(val_X)))