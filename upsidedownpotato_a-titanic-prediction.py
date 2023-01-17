# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# survival	Survival	0 = No, 1 = Yes

# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd

# sex	Sex	

# Age	Age in years	

# sibsp	# of siblings / spouses aboard the Titanic	

# parch	# of parents / children aboard the Titanic	

# ticket	Ticket number	

# fare	Passenger fare	

# cabin	Cabin number	

# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# Import the data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
df_train.describe()
df_train.info()
df_train.groupby('Pclass').Survived.mean()
print('Missing data:', df_train.Pclass.isna().sum())

sns.barplot(df_train.Pclass, df_train.Survived)
print('Missing data:', df_train.Name.isna().sum())

titles = df_train.Name.map(lambda name: name.split(',')[1].split('.')[0])

dummy = df_train.copy()

dummy['Name_Title'] = titles

dummy.groupby('Name_Title').Survived.mean()

f, axes = plt.subplots(2,1, figsize=(20,10))

sns.countplot(titles, hue=df_train.Survived, ax=axes[0])

name_lengths = df_train.Name.map(lambda name: len(name))

dummy['Name_Length'] = name_lengths

name_lengths_plot = sns.countplot(pd.qcut(name_lengths,5), hue=df_train.Survived, ax=axes[1])

name_lengths_plot.set(xlabel='Names Lengths', ylabel='Survived')
print('Missing data:', df_train.Sex.isna().sum())

print ('Mean survival\n', df_train.groupby('Sex').Survived.mean())

sns.countplot(df_train.Sex, hue=df_train.Survived)
print('Missing data:', df_train.Age.isna().sum())

f, axes = plt.subplots(2,1, figsize=(20,10))

sns.countplot(pd.qcut(df_train.Age, 8), hue=df_train.Survived, ax=axes[0])

sns.distplot(df_train.Age.fillna(df_train.Age.median()), ax=axes[1])
print('Missing data:', df_train.SibSp.isna().sum())

print ('SibSp survival mean', df_train.groupby('SibSp').Survived.mean())

sns.countplot(df_train.SibSp, hue=df_train.Survived)
print('Missing data', df_train.Parch.isna().sum())

print('Parch survival means', df_train.groupby('Parch').Survived.mean())

sns.countplot(df_train.Parch, hue=df_train.Survived)
sns.countplot(df_train.Parch + df_train.SibSp, hue=df_train.Survived)
print ('Fare missing data:', df_train.Fare.isna().sum())

sns.distplot(df_train.Fare)

# Looks heavily skewed to the right

sns.countplot(pd.qcut(df_train.Fare, 4), hue=df_train.Survived)
print ('Embarked missing data:', df_train.Embarked.isna().sum())

print('Embarked mean survial', df_train.groupby('Embarked').Survived.mean())

sns.countplot(df_train.Embarked.fillna(df_train.Embarked.mode()), hue=df_train.Survived)
# Function to replace missing data

def replace_missing(df):

    for key in df:

        column = df[key]

        if column.isna().sum() > 0:

            if column.dtypes == object:

                column.fillna(column.mode()[0], inplace=True)

            else:

                column.fillna(column.median(), inplace=True)

    return df
# Function to add name titles and lengths features

def format_names(df):

    titles = df.Name.map(lambda name: name.split(',')[1].split('.')[0].strip())

    titles = titles.map(lambda title: title if title == 'Mr' or title == 'Mrs' or title == 'Miss' or title == 'Master' else 'Other' )

    df['Name_Title'] = titles

    name_lengths = df.Name.map(lambda name: len(name))

    df['Name_Length'] = name_lengths

    return df
# Function to combine Parch and SibSp variables

def get_family_size(df):

    combined = df.Parch + df.SibSp

#     combined = combined.map(lambda x: 3 if x > 8 else (2 if x > 4 else 1))

    df['FamilySize'] = combined

    df = df.drop(['Parch', 'SibSp'], axis=1)

    return df
# Function to preprocess dataframes

def preprocess(df):

    # Let's create the name titles column as discussed above

    df = format_names(df)

    # We don't need passengerIds, ticket numbers or names as they are unique cateogrical values

    passenger_ids = df.PassengerId

    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    # Let's combine the Parch and SibSp into a new variable FamilySize as discussed above

    df = get_family_size(df)

    # Let's take care of missing data

    print ('Missing data\n', df.isna().sum())

    # Cabin number as over 50% missing data so we will drop the column

    df = df.drop(['Cabin'], axis=1)

    df_filled = replace_missing(df)

    # Encoding categorical data

    df_dummies = pd.get_dummies(df_filled, drop_first=True)

    # Taking the log of the fare variable as it is heavily skewed. This is to minimize the impact of outliers.

    df_dummies['Fare'] = np.log1p(df_dummies.Fare)

    # Let's split the data into inputs and targets

    inputs = df_dummies

    targets = []

    if 'Survived' in df_dummies.columns.values:

        inputs = df_dummies.drop(['Survived'], axis=1)

        targets = df_dummies['Survived']

    scaler = StandardScaler()

    scaler.fit(inputs)

    scaled_inputs = scaler.transform(inputs)

    # Let's standardize the inputs

    df_inputs_scaled = pd.DataFrame(columns=inputs.columns.values, data=scaled_inputs)

    return df_inputs_scaled, targets
X_tr, targets = preprocess(df_train)
# Let's first do a simple logistic fit with StatsModels

import statsmodels.api as sm

x = sm.add_constant(X_tr)

log_reg = sm.Logit(targets, x)

log_res = log_reg.fit()

log_res.summary()
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()

reg.fit(X_tr, targets)
# Logistic regression accuracy

reg.score(X_tr, targets)
# Since the test dataset has no dependent variable, let's split our training dataset for cross validation

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tr, targets, test_size=0.2, random_state=0)
# Cross validation function

from sklearn.model_selection import cross_val_score

from sklearn import metrics

def modelfit(alg, X_train, y_train, X_test, y_test):

    alg.fit(X_train, y_train)

    # Predict training set:

    dtrain_predictions = alg.predict(X_train)

    cv_score = cross_val_score(alg, X_train, y_train, cv=5)

    # Print model report:

    print("\nModel Report")

    print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((np.exp(y_train)).values, dtrain_predictions)))

    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    # Predict on testing data:

    predictions = alg.predict(X_test)

    accuracy = ((predictions == y_test).sum())/len(predictions)

    print('Test dataset accuracy:', accuracy)

    return alg

    

modelfit(LogisticRegression(), X_train, y_train, X_test, y_test)
# Support Vector Machine (SVM)

from sklearn.svm import SVC

svm = SVC(kernel='rbf')

modelfit(svm, X_train, y_train, X_test, y_test)
# Hyper parameter tuning

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)



param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10], "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 400, 700, 1000]}



gs = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)



gs = gs.fit(X_tr, targets)
print(gs.best_score_)

print(gs.best_params_)

print(gs.cv_results_)
# Random forest

rf = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=16, n_estimators=50)

modelfit(rf, X_train, y_train, X_test, y_test)
# Submission

submit = pd.read_csv('../input/gender_submission.csv')

submit.set_index('PassengerId',inplace=True)

X_processed_test, _ = preprocess(df_test)

rf.fit(X_tr, targets)

predictions = rf.predict(X_processed_test)

submit['Survived'] = predictions

submit['Survived'] = submit['Survived'].apply(int)

submit

submit.to_csv('submit.csv')