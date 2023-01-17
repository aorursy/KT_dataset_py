import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score



from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read the data

X_full = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

X_full.head()



X_test_full = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
X_full.describe(include='all')
# Pair plot, we can look at the interaction of features with each other

pair_grid_plot = sns.PairGrid(data=X_full)

pair_grid_plot.map(plt.scatter)

plt.show()
# Estimate how many passengers survived in total

sns.countplot(x='Survived', data=X_full)

plt.show()



print(f"Totally survived: {(X_full.Survived.sum() / X_full.Survived.count()):.2f}")
# Assess whether gender affects survival

sns.catplot(x='Sex', col='Survived', kind='count', data=X_full)

plt.show()
# More clearly

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

X_full.Survived[X_full.Sex == 'male'].value_counts().plot.pie(explode=[0, 0.2], autopct='%1.1f%%', shadow=True)

plt.subplot(1, 2, 2)

X_full.Survived[X_full.Sex == 'female'].value_counts().plot.pie(explode=[0, 0.2], autopct='%1.1f%%', shadow=True)

plt.show()
# Crosstab of survived people depending on class

pd.crosstab(index=X_full.Pclass, columns=X_full.Survived, margins=True)
# Let's look at the correlation between features and make the first

# conclusion about which features are important in analysis

X_full_corr = X_full.copy()



X_full_corr = pd.get_dummies(X_full_corr, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

X_full_corr.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_full_corr.dropna(inplace=True)



corr = X_full_corr.corr()



plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, fmt='.2f')

plt.show()
# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)
# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,

                                                                train_size=0.7,

                                                                test_size=0.3,

                                                                random_state=0)
# Select numerical columns

numerical_cols = [c_name for c_name in X_train_full.columns if 

                  X_train_full[c_name].dtype in ['int64', 'float64']]



# Select categorical columns

categorical_cols = [c_name for c_name in X_train_full.columns if 

                    X_train_full[c_name].nunique() < 10 and

                    X_train_full[c_name].dtype == 'object']
# Keep selected cols only

cols = numerical_cols + categorical_cols



X_train = X_train_full[cols].copy()

X_valid = X_valid_full[cols].copy()

X_test = X_test_full[cols].copy()
# Processing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



# Processing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])





# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
model = XGBClassifier(n_estimators=1000, learning_rate=0.05)



clf = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])



clf.fit(X_train, y_train)



preds = clf.predict(X_valid)

print(f'Accuracy: = {accuracy_score(y_valid, preds):.2f}')
clf.get_params().keys()
parameters_grid = {

    'model__learning_rate': [0.01, 0.05, 0.1],

    'model__n_estimators': [n for n in range(200, 1001, 200)],

    'preprocessor__num__strategy': ['mean', 'median', 'most_frequent', 'constant'],

    'preprocessor__cat__imputer__strategy': ['most_frequent', 'constant']

}
#grid_cv = GridSearchCV(clf, parameters_grid, scoring='accuracy', cv=5)

#grid_cv.fit(X_train, y_train)



#print(grid_cv.best_score_)

#print(grid_cv.best_params_)
# Processing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



# Processing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])





# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
model = XGBClassifier(n_estimators=200, learning_rate=0.01)



clf = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', model)

])



clf.fit(X_train, y_train)



preds = clf.predict(X_valid)

print(f'Accuracy: = {accuracy_score(y_valid, preds):.2f}')
preds_test = clf.predict(X_test)



y_test = pd.read_csv('../input/titanic/gender_submission.csv')



print(f"Accuracy: {(accuracy_score(y_test.Survived.values, preds_test) * 100):.2f} %")