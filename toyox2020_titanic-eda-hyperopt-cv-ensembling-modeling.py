import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK, space_eval
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# check shape of the dataset
print('train shape:', train.shape)
print('test shape:', test.shape)
# total null values
print('---- train null ----\n', train.isnull().sum())
print('\n---- test null ----\n', test.isnull().sum())
# set seaborn style
sns.set_palette('rainbow')
sns.set_style('darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

train['Survived'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Dead', 'Survived'], ax=ax1)
ax1.set_title('Survived ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

sns.heatmap(train.drop(['PassengerId'], axis=1).corr(), annot=True, fmt='.2f', cmap='rainbow', ax=ax2)
ax2.set_title('Feature correlation matrix', fontsize=16, y=1.05)

plt.show()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))

train['Sex'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
ax1.set_title('Sex ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

sns.barplot(x='Sex', y='Survived', hue='Sex', data=train, ax=ax2)
ax2.set_title('Survival rate by Sex', fontsize=16, y=1.05)
plt.show()
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
fig.suptitle('Age distribution by sex', fontsize=16, y=1.02)

male_age = train[train['Sex'] == 'male']['Age'].dropna()
female_age = train[train['Sex'] == 'female']['Age'].dropna()

ax.hist([male_age, female_age], label=['male', 'female'], bins=16, stacked=True)
ax.set_xlabel('age')
ax.set_ylabel('count')
ax.legend()

plt.show()
# create bins every 5
bins = list(range(0, 66, 5))

# category of age band for label
age_cat = [str(age) + '-' + str(age+5) for age in bins[:-1]]

# all sex
cut, bins = pd.cut(train['Age'], bins, retbins=True)
groupby_all = train.groupby(cut)
surv_rate_a = groupby_all.sum()['Survived'].dropna() / groupby_all.count()['Survived'].dropna()

# male
male = train.query('Sex == "male"')
cut, bins = pd.cut(male['Age'], bins, retbins=True)
groupby_m = male.groupby(cut)
surv_rate_m = groupby_m.sum()['Survived'].dropna() / groupby_m.count()['Survived'].dropna()

# female
female = train.query('Sex == "female"')
cut, bins = pd.cut(female['Age'], bins, retbins=True)
groupby_f = female.groupby(cut)
surv_rate_f = groupby_f.sum()['Survived'].dropna() / groupby_f.count()['Survived'].dropna()

# convert train to DataFrame for survival rates by age-band
surv_rate_a = pd.DataFrame(index=age_cat, columns=['all'], data=surv_rate_a.values)
surv_rate_m = pd.DataFrame(index=age_cat, columns=['male'], data=surv_rate_m.values)
surv_rate_f = pd.DataFrame(index=age_cat, columns=['female'], data=surv_rate_f.values)

# plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Survival rates by sex and age', fontsize=16, y=1.02)

sns.barplot(x=surv_rate_a.index, y='all', data=surv_rate_a, ax=ax1)
ax1.set_xticklabels(surv_rate_m.index, rotation=90)
ax1.set_ylabel('survival rate')
ax1.set_title('all')

sns.barplot(x=surv_rate_m.index, y='male', data=surv_rate_m, ax=ax2)
ax2.set_xticklabels(surv_rate_m.index, rotation=90)
ax2.set_xlabel('age')
ax2.set_ylabel('')
ax2.set_title('male')

sns.barplot(x=surv_rate_f.index, y='female', data=surv_rate_f, ax=ax3)
ax3.set_xticklabels(surv_rate_f.index, rotation=90)
ax3.set_ylabel('')
ax3.set_title('female')

plt.show()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.distplot(train['Fare'], kde=False, bins=18, ax=ax1)
ax1.set_title('Fare distribution', fontsize=16)
ax1.set_ylabel('count')

fare_s0 = train[train['Survived'] == 0]['Fare'].dropna()
fare_s1 = train[train['Survived'] == 1]['Fare'].dropna()
ax2.hist([fare_s0, fare_s1], label=['dead', 'survived'], bins=18)
ax2.set_title('Survived by Fare', fontsize=16)
ax2.set_xlabel('fare')
ax2.set_ylabel('count')
ax2.legend()

plt.show()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))

train['Pclass'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
ax1.set_title('Pclass ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

sns.barplot(x='Pclass', y='Survived', data=train, ax=ax2)
ax2.set_title('Survival rate by Pclass', fontsize=16, y=1.05)

plt.show()
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(12, 10))

train['SibSp'].value_counts().plot.pie(ax=ax1)
ax1.set_title('SibSp ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

sns.barplot(x='SibSp', y='Survived', hue='Sex', data=train, ax=ax2)
ax2.set_title('Survival rate by SibSp and Sex', fontsize=16, y=1.05)

train['Parch'].value_counts().plot.pie(ax=ax3)
ax3.set_title('Parch ratio', fontsize=16, y=1.05)
ax3.set_ylabel('')

sns.barplot(x='Parch', y='Survived', hue='Sex', data=train, ax=ax4)
ax4.set_title('Survival rate by Parch and Sex', fontsize=16, y=1.05)

plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()
f,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(18,4))

train['Embarked'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
ax1.set_title('Embarked ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

sns.barplot(x='Embarked', y='Survived', data=train, ax=ax2)
ax2.set_title('Survival rates by Embarked', fontsize=16, y=1.05)

sns.countplot('Embarked',hue='Pclass',data=train, ax=ax3)
ax3.set_title('Embarked rates by Pclass', fontsize=16, y=1.05)

plt.show()
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12, 4))

train['Cabin'].notnull().value_counts().plot.pie(autopct='%1.1f%%', labels=['null', 'not null'], ax=ax1)
ax1.set_title('Cabin null ratio', fontsize=16, y=1.05)
ax1.set_ylabel('')

tmp = pd.DataFrame()
tmp['Survived'] = train['Survived']
tmp['CabinLevel'] = train['Cabin'].str[0]
tmp['CabinLevel'].replace(np.nan, 'U', inplace=True)
sns.barplot(x='CabinLevel', y='Survived', data=tmp, order=sorted(tmp['CabinLevel'].unique()), ax=ax2)
ax2.set_title('Survival rates by Cabin level', fontsize=16, y=1.05)

plt.show()
combine = pd.concat([train, test], sort=False)
# convert to numeric
combine['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
# fill NaN Ages with median
combine['Age'].fillna(combine['Age'].median(), inplace=True)
# standardize
standard = StandardScaler()
combine[['Age']] = standard.fit_transform(combine[['Age']])
# Fill in the missing values with median
combine['Fare'].fillna(combine['Fare'].median(), inplace=True)
# robust scaling
rscaler = RobustScaler(quantile_range=(25., 75.))
combine[['Fare']] = rscaler.fit_transform(combine[['Fare']])
# logarithmic conversion
print('before:', combine['Fare'].skew())
combine['Fare'] = np.log1p(combine['Fare'])
print('after:', combine['Fare'].skew())
# normalize 
combine['Pclass'] = (combine['Pclass'] - 1) / 2
# add "FamilySize" column
combine['FamilySize'] = combine['SibSp'] + combine['Parch']
# Visualization
fig = sns.barplot(x='FamilySize', y='Survived', hue='Sex', data=combine)
fig.set_title('Survival rates by FamilySize and Sex', fontsize=16, y=1.05)
plt.show()
# standardize
standard = StandardScaler()
combine[['FamilySize']] = standard.fit_transform(combine[['FamilySize']])
# Fill in the missing values with mode
combine['Embarked'].fillna(combine['Embarked'].mode()[0], inplace=True)
# one-hot encoding
combine = pd.get_dummies(combine, columns=['Embarked'], drop_first=True)
# extract titles from Names
combine['Title'] = combine.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
combine['Title'].unique()
# Count titles
pd.crosstab(combine['Title'], combine['Sex'])
# Aggregate 10 or less titles as "other"
combine.loc[(combine['Title'] != 'Master')\
          & (combine['Title'] != 'Miss')\
          & (combine['Title'] != 'Mr')\
          & (combine['Title'] != 'Mrs'), 'Title'] = 'other'
pd.crosstab(combine['Title'], combine['Sex'])
# Visualization
fig = sns.barplot(x='Title', y='Survived', data=combine)
fig.set_title('Survival rates by title', fontsize=16, y=1.05)
plt.show()
# convert to numeric
combine = pd.get_dummies(combine, columns=['Title'], drop_first=True)
# replace notnull to 1
tmp = pd.DataFrame()
tmp['Cabin'] = combine['Cabin']
tmp['Cabin'].loc[tmp['Cabin'].notnull()] = 1
# replace null to 0
combine['Cabin'] = tmp['Cabin']
combine['Cabin'].replace(np.nan, 0, inplace=True)
combine.head()
# remove unnecessary columns
combine_cleaned = combine.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket'], axis=1)
combine_cleaned.head()
# split combine_cleaned into train and test
train = combine_cleaned[:train.shape[0]]
test = combine_cleaned[train.shape[0]:]

# split feature and label
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop('Survived', axis=1)

# check the shapes
print('X_train shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_test shape:', X_test.shape)
def hyperopt_and_pred(models, max_evals=100):
    preds = []
    
    for model in models:
        clf = model['classifier']
        space = model['space']
        
        # define objective function
        def objective(space):
            # create model object
            classifier = clf(**space)
            # train the model
            classifier.fit(X_train, y_train)
            # cross validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            acc = cross_validate(estimator=classifier, X=X_train, y=y_train, cv=skf)
            mean_acc = np.mean(acc['test_score'])
            return{'loss': 1 - mean_acc, 'status': STATUS_OK}

        print('='*10, str(clf).split('.')[-1].replace('\'>', ''), '='*10)
        
        # create Trials object
        trials = Trials()
        # minimize the objective over the space
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=1)
        # fit and predict
        best_params = space_eval(space, best)
        predict = clf(**best_params).fit(X_train, y_train).predict(X_test)
        acc = 1 - trials.best_trial['result']['loss']
        preds.append(predict)
        
        print('\n', 'best parameters:', best_params)
        print('accuracy:', f'{acc:.04f}', '\n\n')
        
    return preds
# Set models and hyper-parameters to be optimized
models = [
    # Logistic Regression
    {
        'classifier': LogisticRegression,
        'space': {
            'C': hp.uniform('C', 0, 100),
            'max_iter': hp.choice('max_iter', [2000]),
            'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'sag', 'saga'])
        }
    },
    # Random Forest
    {
        'classifier': RandomForestClassifier,
        'space': {
            'n_estimators': hp.choice('n_estimators', np.arange(10, 401, 10)),
            'max_depth': hp.uniform('max_depth', 1, 5),
            'criterion': hp.choice('criterion', ['gini', 'entropy'])
        }
    },
    # KNeighbors
    {
        'classifier': KNeighborsClassifier,
        'space': {
            'n_neighbors': hp.choice('n_neighbors', np.arange(1, 15))
        }
    },
    # AdaBoost
    {
        'classifier': AdaBoostClassifier,
        'space': {
            'n_estimators': hp.choice('n_estimators', [30,50,100,200,300]),
            'learning_rate': hp.uniform('learning_rate', 0.8, 1.4)
        }
    },
    # SVC
    {
        'classifier': SVC,
        'space': {
            'C': hp.uniform('C', 0, 2),
            'gamma': hp.loguniform('gamma', -8, 2),
            'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
        }
    },
    # GBDT
    {
        'classifier': LGBMClassifier,
        'space': {
            'objective': hp.choice('objective', ['binary']),
            'max_bin': hp.choice ('max_bin', np.arange(64, 513, 1)),
            'num_leaves': hp.choice('num_leaves', np.arange(30, 201, 10)),
            'max_depth': hp.choice('max_depth', np.arange(3, 10, 1)),
            'learning_rate': hp.uniform('learning_rate', 0.03, 0.2)
        }
    },
    # Multilayer perceptron
    {
        'classifier': MLPClassifier,
        'space': {
            'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [8, 16, 32, (8,8), (16,16)]),
            'activation': hp.choice('activation', ['relu', 'tanh']),
            'max_iter': hp.choice('max_iter', [3000])
        }
    }
]
# fit and predict
predictions = hyperopt_and_pred(models)
# check correlation of each predictions
clf_names = [str(clf['classifier']).split('.')[-1].replace('\'>', '') for clf in models]
corr_preds = pd.DataFrame(predictions).T.corr()
corr_preds.columns = clf_names
corr_preds.index = clf_names
fig = sns.heatmap(corr_preds, annot=True, fmt='.2f', cmap='rainbow')
fig.set_title('Predictions correlation matrix', fontsize=16, y=1.05)
plt.show()
# take the average of the predictions
ensembled_pred = np.round(sum(predictions) / len(predictions)).astype('int')
results = pd.Series(ensembled_pred, name='Survived')
submission = pd.concat([submission['PassengerId'], results], axis=1)
submission.to_csv('submission.csv', index=False)