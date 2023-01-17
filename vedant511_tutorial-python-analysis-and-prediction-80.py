import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
train_set = pd.read_csv('../input/train.csv')

test_set = pd.read_csv('../input/test.csv')
train_set.head()
train_set.describe()
train_set.info()

cat_cols = ['Survived', 'Sex', 'Pclass', 'Embarked', 'Parch', 'SibSp']

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)







for i in range(2):

    for j in range(3):

        c = i * 3 + j

        ax = axs[i][j]

    

        sns.countplot(train_set[cat_cols[c]], hue=train_set['Survived'], ax = ax)

        ax.set_title(cat_cols[c], fontsize=14, fontweight='bold')

        ax.grid()

bins = np.arange(0, 80, 5)

g = sns.FacetGrid(train_set, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()

plt.show()  

bins = np.arange(0, 80, 5)

g = sns.FacetGrid(train_set, row='Sex', col='Embarked', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Age', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()

plt.show()  
bins = np.arange(0, 550, 50)

g = sns.FacetGrid(train_set, row='Sex', col='Pclass', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()

plt.show()  
bins = np.arange(0, 550, 50)

g = sns.FacetGrid(train_set, row='Sex', col='Embarked', hue='Survived', margin_titles=True, size=3, aspect=1.1)

g.map(sns.distplot, 'Fare', kde=False, bins=bins, hist_kws=dict(alpha=0.6))

g.add_legend()

plt.show()  
def missing_zero_values_table(df):

    zero_val = (df == 0.00).astype(int).sum(axis=0)

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

    mz_table = mz_table.rename(

    columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

    mz_table['Data Type'] = df.dtypes

    mz_table = mz_table[

        mz_table.iloc[:,1] != 0].sort_values(

    '% of Total Values', ascending=False).round(1)

    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      

        "There are " + str(mz_table.shape[0]) +

          " columns that have missing values.")

    return mz_table
missing_zero_values_table(train_set)
missing_zero_values_table(test_set)
train_set[train_set['Embarked'].isna()]

train_set[(train_set['Pclass']==1) & (train_set['Embarked']=='Q')]['Sex'].value_counts()
train_set['Embarked'][61] = 'S'

train_set['Embarked'][829] = 'S'
train_set[train_set['Age'].isna()]['Sex'].value_counts()
test_set[test_set['Fare'].isna()]
test_set.at[152, 'Fare'] = np.nanmedian(test_set[(test_set['Pclass']==3) & (test_set['Embarked']=='S')]['Fare'])
test_set['Fare'][152]
train_set['FamilySize'] = train_set['SibSp'] + train_set['Parch'] + 1

test_set['FamilySize'] = test_set['SibSp'] + test_set['Parch'] + 1
train_set['Name']
def extract_title(name):

    return name.split(',')[1].split()[0].strip()

train_set['Title'] = train_set['Name'].apply(extract_title)

train_set['Title'].value_counts()
def refine_title(title):

    if title in ['Mr.', 'Sir.', 'Major.', 'Dr.', 'Capt.']:

        return 'mr'

    elif title == 'Master.':

        return 'master'

    elif title in ['Miss.', 'Ms.']:

        return 'miss'

    elif title in ['Mrs.', 'Lady.']:

        return 'mrs'

    else:

        return 'other'
train_set['Title'] = train_set['Title'].apply(refine_title)

train_set['Title'].value_counts()
test_set['Title'] = test_set['Name'].apply(extract_title)

test_set['Title'] = test_set['Title'].apply(refine_title)

test_set['Title'].value_counts()

train_set.head()

sns.distplot(train_set['Fare'])
sns.distplot(test_set['Fare'])

fare_bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, np.inf]

fare_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

train_set['FareBin'] = pd.cut(train_set['Fare'], bins=fare_bins, labels=fare_labels)

test_set['FareBin'] = pd.cut(test_set['Fare'], bins=fare_bins, labels=fare_labels)

train_set.head()

test_set.head()

def fill_age(df):

    for idx, row in df.iterrows():

        if pd.isnull(row['Age']):

            value = df[ 

                (df['Pclass']==row['Pclass']) & 

                (df['Sex']==row['Sex']) & 

                (df['Embarked']==row['Embarked']) & 

                (df['Title']==row['Title']) & 

                (df['FareBin']==row['FareBin'])

            ]['Age'].median()

            if pd.isnull(value):

                value = df[ 

                (df['Sex']==row['Sex']) & 

                (df['Title']==row['Title']) & 

                (df['FareBin']==row['FareBin'])

            ]['Age'].median()

            if pd.isnull(value):

                value = df[df['Title']==row['Title']]['Age'].median()

            df.at[idx, 'Age'] = value 

fill_age(train_set)

fill_age(test_set)

missing_zero_values_table(train_set)

missing_zero_values_table(test_set)
sns.distplot(train_set['Age'])
sns.distplot(test_set['Age'])
train_set['Age'].describe()
age_bins = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, np.inf]

age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

train_set['AgeBin'] = pd.cut(train_set['Age'], bins=age_bins, labels=age_labels)

test_set['AgeBin'] = pd.cut(test_set['Age'], bins=age_bins, labels=age_labels)

train_set.head()
features = ['Pclass', 'Sex', 'Embarked', 'FamilySize', 'Title', 'FareBin', 'AgeBin', 'Fare', 'Age']
y_train = train_set['Survived']
x_train = train_set[features]

x_test = test_set[features]
x_train.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(x_train[['Age', 'Fare', 'FamilySize']])

x_test[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(x_test[['Age', 'Fare', 'FamilySize']])
x_train.head()
x_test.head()
x_train['Pclass'] = pd.Categorical(x_train['Pclass'])

x_train['Sex'] = pd.Categorical(x_train['Sex'])

x_train['Embarked'] = pd.Categorical(x_train['Embarked'])

x_train['Title'] = pd.Categorical(x_train['Title'])



x_test['Pclass'] = pd.Categorical(x_test['Pclass'])

x_test['Sex'] = pd.Categorical(x_test['Sex'])

x_test['Embarked'] = pd.Categorical(x_test['Embarked'])

x_test['Title'] = pd.Categorical(x_test['Title'])
x_train.info()
x_test.info()
x_train = pd.get_dummies(x_train)

x_test = pd.get_dummies(x_test)
x_train.head()
x_test.head()
X = x_train.copy()

y = y_train.copy()

test_data = x_test.copy()



X.to_csv('X.csv', index=False, header=True)

test_data.to_csv('test_data.csv', index=False, header=True)

y.to_csv('y.csv', index=False, header=True)

from sklearn.externals import joblib



joblib.dump(scaler, 'scaler.pkl')
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV

from xgboost import XGBClassifier
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=25)
X_train.head()
y_train.value_counts()
y_valid.value_counts()
gnb_clf = GaussianNB()

gnb_clf.fit(X_train, y_train)

gnb_prediction = gnb_clf.predict(X_valid)

print(classification_report(y_valid, gnb_prediction))

svc_clf = SVC(kernel='linear')

svc_clf.fit(X_train, y_train)

svc_prediction = svc_clf.predict(X_valid)

print(classification_report(y_valid, svc_prediction))

tree_clf = DecisionTreeClassifier(max_depth=5)

tree_clf.fit(X_train, y_train)

tree_prediction = tree_clf.predict(X_valid)

print(classification_report(y_valid, tree_prediction))

rf_clf = RandomForestClassifier(n_estimators=200, max_depth=10)

rf_clf.fit(X_train, y_train)

rf_prediction = rf_clf.predict(X_valid)

print(classification_report(y_valid, rf_prediction))

ada_clf = AdaBoostClassifier()

ada_clf.fit(X_train, y_train)

ada_prediction = ada_clf.predict(X_valid)

print(classification_report(y_valid, ada_prediction))

knn_clf = KNeighborsClassifier(6)

knn_clf.fit(X_train, y_train)

knn_prediction = knn_clf.predict(X_valid)

print(classification_report(y_valid, knn_prediction))

mlpc_clf = MLPClassifier(alpha=1, max_iter=5000)

mlpc_clf.fit(X_train, y_train)

mlpc_prediction = mlpc_clf.predict(X_valid)

print(classification_report(y_valid, mlpc_prediction))

gp_clf = GaussianProcessClassifier(1.0 * RBF(1.0))

gp_clf.fit(X_train, y_train)

gp_prediction = gp_clf.predict(X_valid)

print(classification_report(y_valid, gp_prediction))

log_clf = LogisticRegressionCV(cv=5, max_iter=5000)

log_clf.fit(X_train, y_train)

log_prediction = log_clf.predict(X_valid)

print(classification_report(y_valid, log_prediction))

log_prediction_soft = log_clf.predict_proba(X_valid)

xg_clf = XGBClassifier()

xg_clf.fit(X_train, y_train)

xg_prediction = xg_clf.predict(X_valid)

print(classification_report(y_valid, xg_prediction))

models = [gnb_clf, knn_clf, log_clf, svc_clf, tree_clf, ada_clf, xg_clf, rf_clf, mlpc_clf, gp_clf]

model_names = ['Gaussian NB', 'KNN', 'Logistic Reg', 'SVC', 'Decision Tree', 'Adaboost', 'XGBoost', 

              'Random Forest', 'MLPC', 'Gaussian Process']

accuracies = [np.round(m.score(X_valid, y_valid), 2) for m in models]

result_df = pd.DataFrame({'Model': model_names, 

                         'Accuracy': accuracies}).set_index('Model').sort_values('Accuracy', ascending=False)

result_df
final_model = RandomForestClassifier(max_depth=10, n_estimators=200).fit(X, y)
print(final_model.score(X_valid, y_valid))
print(final_model.score(X_train, y_train))
test_ids = test_set['PassengerId']

prediction = final_model.predict(test_data)

submission = pd.DataFrame({'PassengerId': test_ids,

                          'Survived': prediction})

submission.head()
submission.to_csv('titanic_challenge_submission_vedant511.csv', index=False, header=True)

from sklearn.externals import joblib



joblib.dump(final_model, 'model.pkl')
