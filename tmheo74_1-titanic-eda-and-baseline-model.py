# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import random
from datetime import datetime

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display
# Set a few plotting defaults
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 14
plt.rcParams['patch.edgecolor'] = 'k'
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')
train_raw.shape, test_raw.shape
train_raw.head()
test_raw.head()
train_raw.info()
train_raw.describe()
train_raw['data'] = 'train'
test_raw['data'] = 'test'
data_all = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)
g = sns.countplot(data_all.Survived)
plt.title("Survived Count")
plt.show()
g = sns.countplot(x='data',hue='Sex',data=data_all)
plt.title("Sex Count")
plt.show()
g = sns.countplot(x='data',hue='Pclass',data=data_all)
plt.title("Pclass Count")
plt.show()
g = sns.countplot(x='data',hue='Embarked',data=data_all)
plt.title("Embarked Count")
plt.show()
def plot_distribution_by_target(df, field):
    
    df = df[df[field].notnull()]

    fig = plt.figure(figsize = (14, 12))
    ax1 = plt.subplot(221)
    
    sns.kdeplot(df[field], label='Total', alpha=0.7, ax=ax1)
    sns.kdeplot(df[df.data == 'train'][field], label='Train', alpha=0.7, ax=ax1)
    sns.kdeplot(df[df.data == 'test'][field], label='Test', alpha=0.7, ax=ax1)

    plt.xlabel(field.upper())
    plt.ylabel('Density')
    
    ax2 = plt.subplot(222)

    sns.boxplot(x='data', y=field, data=df, ax=ax2)
    
    df = df[df.data == 'train']
    
    ax3 = plt.subplot(223)

    sns.kdeplot(df[df.Survived == 1][field], label='Survived', alpha=0.7, ax=ax3)
    sns.kdeplot(df[df.Survived == 0][field], label='Not Survived', alpha=0.7, ax=ax3)

    plt.xlabel(field.upper())
    plt.ylabel('Density')

    ax4 = plt.subplot(224)

    sns.boxplot(x='Survived', y=field, data=df, ax=ax4)
    plt.xticks((0,1), ('Not Survived','Survived'))
    
    fig.suptitle(f'{field.upper()} Distribution', fontsize=20)
    
    plt.show()
train_raw['data'] = 'train'
test_raw['data'] = 'test'
data_all = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)
plot_distribution_by_target(data_all, 'Age')
plot_distribution_by_target(data_all, 'Fare')
plot_distribution_by_target(data_all, 'SibSp')
plot_distribution_by_target(data_all, 'Parch')
g = sns.catplot("Survived", col="Sex", data=train_raw, kind="count")
plt.show()
g = sns.catplot("Survived", col="Pclass", data=train_raw, kind="count")
plt.show()
g = sns.catplot("Survived", col="Embarked", data=train_raw, kind="count")
plt.show()
features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_raw[features]
X_test = test_raw[features]
y_train = train_raw.Survived
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
X_train.info()
X_test.info()
X_train.Embarked = X_train.Embarked.fillna(X_train.Embarked.mode()[0])
X_test.Embarked = X_test.Embarked.fillna(X_train.Embarked.mode()[0])
le = LabelEncoder()
X_train.Sex = le.fit_transform(X_train[['Sex']])
X_test.Sex = le.transform(X_test[['Sex']])
X_train.Embarked = le.fit_transform(X_train[['Embarked']])
X_test.Embarked = le.transform(X_test[['Embarked']])
median_imputer = SimpleImputer(strategy='median')
X_train.Age = median_imputer.fit_transform(X_train[['Age']])
X_test.Age = median_imputer.transform(X_test[['Age']])

X_train.Fare = median_imputer.fit_transform(X_train[['Fare']])
X_test.Fare = median_imputer.transform(X_test[['Fare']])
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
def cv_model(train, train_labels, model, name, model_results=None, cv=10, scoring='accuracy'):
    """Perform k fold cross validation of a model"""
    
    cv_scores = cross_val_score(model, train, train_labels, cv=cv, scoring=scoring, n_jobs=-1)
    print(f'{cv} Fold CV {scoring} for {name}: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
    if model_results is None:
        model_results = pd.DataFrame({
            'model': name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }, index = [0])
    else:
        model_results = model_results.append(pd.DataFrame({
            'model': name,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }, index = [0]), ignore_index = True)

    return model_results, model
def show_model_results(model_results):
    df = model_results.copy().set_index('model')
    df.sort_values(by='cv_mean', inplace=True)
    display(df)
    df['cv_mean'].plot.bar(color = 'orange', figsize = (10, 8),
                                      yerr = list(df['cv_std']),
                                      edgecolor = 'k', linewidth = 2)
    plt.title('Model Train CV Accuracy Score Results');
    plt.ylabel('Mean CV Accuracy Score (with error bar)');
    plt.show()
# Dataframe to hold results
model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])
for n in [1, 5, 10, 20]:
    print(f'\nKNN with {n} neighbors\n')
    model_results, _ = cv_model(X_train, y_train,
                             KNeighborsClassifier(n_neighbors = n, n_jobs=-1), f'KNN-{n}', model_results)
model_results, _ = cv_model(X_train, y_train, LinearSVC(random_state=RANDOM_SEED), 'LSVC',
                            model_results)
model_results, _ = cv_model(X_train, y_train, SVC(random_state=RANDOM_SEED), 'SVC',
                            model_results)
model_results, _ = cv_model(X_train, y_train, LogisticRegression(n_jobs=-1, random_state=RANDOM_SEED),
                            'LR', model_results)
model_results, _ = cv_model(X_train, y_train, DecisionTreeClassifier(random_state=RANDOM_SEED),
                            'DTree', model_results)
model_results, _ = cv_model(X_train, y_train,
                            RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED),
                            'RF', model_results)
model_results, _ = cv_model(X_train, y_train, xgb.XGBClassifier(random_state=RANDOM_SEED),
                            'XGB', model_results)
show_model_results(model_results)
base_model = xgb.XGBClassifier(random_state=RANDOM_SEED)
base_model.fit(X_train, y_train)
pred = base_model.predict(X_test)
def plot_feature_importances(estimator, x_cols, n=20, threshold = 0.95):
    df = pd.DataFrame({'feature': x_cols, 'importance': estimator.feature_importances_})
    
    # Sort features with most important at the head
    df = df.sort_values('importance', ascending = False).reset_index(drop = True)
    
    # Normalize the feature importances to add up to one and calculate cumulative importance
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])
    
    plt.rcParams['font.size'] = 12
    
    # Bar plot of n most important features
    df.loc[:n, :].plot.barh(y = 'importance_normalized', 
                            x = 'feature', color = 'darkgreen', 
                            edgecolor = 'k', figsize = (12, 8),
                            legend = False, linewidth = 2)

    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 
    plt.title(f'{min(n, len(df))} Most Important Features', size = 18)
    plt.gca().invert_yaxis()
    
    if threshold:
        # Cumulative importance plot
        plt.figure(figsize = (8, 6))
        plt.plot(list(range(1, len(df)+1)), df['cumulative_importance'], 'b-')
        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 
        plt.title('Cumulative Feature Importance', size = 18);
        
        # Number of features needed for threshold cumulative importance
        # This is the index (will need to add 1 for the actual number)
        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
        
        # Add vertical line to plot
        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')
        plt.show();
        
        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 
                                                                                  100 * threshold))
    
    print(f'zero importance feature count : {len(df[df.importance == 0])}')
    
    return df
plot_feature_importances(base_model, features)
submit_df = pd.DataFrame()
submit_df['PassengerId'] = test_raw.PassengerId
submit_df['Survived'] = pred
submit_df[['PassengerId','Survived']].to_csv(f'baseline_submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv', index=False)
