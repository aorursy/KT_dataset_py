## imports and datasets



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import cross_val_score, GridSearchCV



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier



data_folder = '../input/titanic/'

df = pd.concat([pd.read_csv(data_folder + 'train.csv'), pd.read_csv(data_folder + 'test.csv')], sort=False)

df.set_index(np.arange(df.shape[0]), inplace=True)

print(df.info())
## some useful functions



def estimator_results(est):

    print(est.best_score_)

    print(est.best_estimator_)

    

def feature_importances(est):

    fig, ax = plt.subplots(figsize=(12,4))

    f_i = pd.Series(data=est.best_estimator_.feature_importances_,

                    index=df.columns).sort_values(ascending=False)

    sns.barplot(x=f_i.values, y=f_i.index, orient='h', ax=ax)

    

def scaled_coefs(est):

    fig, ax = plt.subplots(figsize=(12,4))

    s = MinMaxScaler((-1, 1)).fit_transform(est.best_estimator_.coef_.T)

    coefs = pd.Series(data=s.reshape(-1), index=df.columns).sort_values(ascending=False)

    sns.barplot(x=coefs.values, y=coefs.index, orient='h', ax=ax)

    

def main_output(ests):

    scores = {} 

    for name, est in ests.items():

        out_df = pd.DataFrame()

        out_df['PassengerId'] = np.arange(892, 1310)

        out_df['Survived'] = est.predict(test)

        out_df['Survived'] = out_df.Survived.astype('uint8')

        filename = name + '.csv'

        out_df.to_csv(filename, index=False)

        scores[name] = str(np.round(est.best_score_, 3))

    score_frame = pd.DataFrame(data = scores.items(), columns = ('Name', 'Score'))

    score_frame.to_csv('titanic_scores.csv', index=False)

    score_frame.sort_values(by='Score', ascending=False, inplace=True)

    return score_frame
## feature processing



## extracting titles from names

df['Title'] = df.Name.str.extract(r'([A-Za-z]+)\.')



## some of the titles are present in only one instance, so I'll merge them into 6 groups

df.Title = df.Title.replace(['Ms', 'Mme'], 'Mrs')

df.Title = df.Title.replace(['Mlle'], 'Miss')

df.Title = df.Title.replace(['Rev', 'Dr', 'Capt', 'Col', 'Major'], 'Crew')

df.Title = df.Title.replace(['Countess', 'Don', 'Lady', 'Sir', 'Jonkheer', 'Dona'], 'Royalty')



## the cabin feature has too many lost data to be useful, so I just mark if it is known or not

df.Cabin = [int(pd.notna(x)) for x in df.Cabin]



## missing age values are filled using group medians

df.Age = df.Age.groupby([df.Title, df.Pclass]).apply(lambda x: x.fillna(x.median()))

df.Age.describe()



## these two features are almost full, only a couple values are missing, 

## so I fill them with general mode and median

df.Embarked.fillna(df.Embarked.mode()[0], inplace=True)

df.Fare.fillna(df.Fare.median(), inplace=True)



## transform sex to integers

df.Sex = df.Sex.transform(lambda x: 1 if x == 'male' else 0)



## combine family size related features

df['Family'] = df.SibSp + df.Parch
## relation of cabin, class, title, sex, embarkment port and family size to survival rate



count_cols = ['SibSp', 'Parch', 'Family', 'Title', 'Sex', 'Embarked', 'Pclass', 'Cabin']



fig, axs = plt.subplots(2, 3, figsize = (15, 8))

for r in range(2):

    for c in range(3):

        i = r*3+c

        ax = axs[r][c]

        sns.countplot(x=count_cols[i+2], hue='Survived', data=df, ax=ax)
## compare family size feature to its two origin features



fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):

    ax = axs[i]

    sns.countplot(x=count_cols[i], hue='Survived', data=df, ax=ax)
## visualize age and fare grouped by class and sex



age_grid = sns.FacetGrid(df, row='Sex', col='Pclass', hue='Survived', height=3.6)

age_grid.map(sns.scatterplot, 'Age', 'Fare')

age_grid.add_legend()
## correlation matrix



corr_matrix = df.corr()

cmap = sns.diverging_palette(20, 150, as_cmap=True)

mask = np.ones_like(corr_matrix, dtype='bool')

mask[np.tril_indices_from(mask)] = False

sns.heatmap(corr_matrix, cmap=cmap, mask=mask, square=True)
## one-hot encoding the non-ordinal features

encoded_feats = pd.get_dummies(df[['Embarked', 'Title']])



## getting rid of excessive data and adding the encoded features

df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked', 'Title', 'SibSp', 'Parch'], axis=1, inplace=True)

df = df.join(encoded_feats)

df = df.astype({'Pclass': 'uint8', 'Family': 'uint8', 'Cabin': 'uint8'})



print(df.info(), '\n')

print(df.describe())
## making final train and test sets



target = df.pop('Survived')

target.dropna(inplace = True)



scaled_set = StandardScaler().fit_transform(df)



test = scaled_set[891:]

train = scaled_set[:891]
## decision tree



param_grid = {

    'max_depth': [5, 10, 15, 20], 

    'min_samples_split': [2, 5, 10, 15, 20, 30]

}

tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(tree_clf)

feature_importances(tree_clf)
## random forest



param_grid = {

    'n_estimators': [5, 10, 30, 70, 120, 200], 

    'max_depth': [7, 9, 10, 12],

    'min_samples_split': [2, 3, 5], 

    'max_features':  [5, 7, 9, None]

}

forest_clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(forest_clf)

feature_importances(forest_clf)
## extra trees



param_grid = {

    'n_estimators': [5, 10, 12, 15, 20, 30, 70, 120], 

    'max_depth': [1, 3, 5, 7],

    'min_samples_split': [2, 3, 5], 

    'max_features': [5, 7, 10, None]

}

extra_clf = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(extra_clf)

feature_importances(extra_clf)
## adaboost



ada_estimators = [DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=2), None]

param_grid = {

    'base_estimator': ada_estimators, 

    'n_estimators': [10, 30, 50, 100, 200, 500],

    'learning_rate': [1, 0.9, 0.8, 0.5], 

    'algorithm': ['SAMME.R']

}

ada_clf = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(ada_clf)

feature_importances(ada_clf)
## logistic regression



param_grid = {

    'intercept_scaling': [0.5, 0.7, 0.8, 1], 

    'C': [0.1, 0.5, 1, 10, 100],

    'solver': ['liblinear', 'newton-cg', 'lbfgs']

}

log_clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(log_clf)

scaled_coefs(log_clf)
## stochastic gradient descent



param_grid = {

    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],

    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 

    'max_iter': [500, 1000, 5000]

}

sgd_clf = GridSearchCV(SGDClassifier(), param_grid, cv=5, verbose=1, n_jobs=-1).fit(train, target)

estimator_results(sgd_clf)

scaled_coefs(sgd_clf)
## preparing output data and making csv files with submissions for every estimator and scores



estimators = {

    'decision tree': tree_clf,

    'random forest': forest_clf,

    'extra trees': extra_clf,

    'adaboost': ada_clf,

    'logistic regression': log_clf,

    'stochastic gradient descent': sgd_clf

}



main_output(estimators)