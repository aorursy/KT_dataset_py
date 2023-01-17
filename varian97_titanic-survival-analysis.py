%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
warnings.filterwarnings('ignore')
plt.style.use("ggplot")

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
data = pd.read_csv('../input/train.csv')
data.head()
data.describe()
data.describe(include=['O'])
data.shape
data.isnull().sum()
data.groupby(['Survived'])['Embarked'].value_counts()
data['Embarked'].fillna('S', inplace=True)
data['Embarked'].isnull().sum()
def extract_title(x):
    res = re.findall(r'[A-Za-z]+[.]', x)
    if res:
        return res[0]
    else:
        return None

data['NameTitle'] = data['Name'].apply(extract_title)
data.head(3)
pd.crosstab(data['NameTitle'], data['Sex'], margins=True)
def fix_title(data):
    data['NameTitle'].replace(['Capt.', 'Col.', 'Countess.', 'Don.', 'Dr.', 'Lady.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.'], 
                              'Other.', inplace=True)
    data['NameTitle'].replace(['Mme.', 'Mlle.', 'Ms.'], 'Miss.', inplace=True)

fix_title(data)
data.groupby('NameTitle')['Age'].mean()
data.loc[data['Age'].isnull() & data['NameTitle'].str.contains('Master.'), 'Age'] = 5
data.loc[data['Age'].isnull() & data['NameTitle'].str.contains('Miss.'), 'Age'] = 22
data.loc[data['Age'].isnull() & data['NameTitle'].str.contains('Mr.'), 'Age'] = 32
data.loc[data['Age'].isnull() & data['NameTitle'].str.contains('Mrs.'), 'Age'] = 36
data.loc[data['Age'].isnull() & data['NameTitle'].str.contains('Other.'), 'Age'] = 46
data['Age'].isnull().any()
plt.pie(data['Survived'].value_counts(), labels=['Not Survived', 'Survived'], autopct='%1.1f%%');
pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='vlag')
f, ax = plt.subplots(1,2, figsize=(15,5), sharey=True)
data['Pclass'].value_counts().plot(kind='bar', ax=ax[0]);
ax[0].set_title('Pclass Distribution')
ax[0].set_ylabel('Count')
ax[0].set_xlabel('Pclass')
sns.countplot('Pclass', hue='Survived', data=data, ax=ax[1], palette='deep');
ax[1].set_title('Pclass vs Survived')
pd.crosstab(data['Survived'], data['Sex'], margins=True).style.background_gradient(cmap='vlag')
plt.figure(figsize=(20,12))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

ax1.set_title('Sex Distribution')
ax1.set_ylabel('Count')
data['Sex'].value_counts().plot(kind='bar', ax=ax1, rot=0)

ax2.set_title('Sex vs Survived')
sns.countplot('Sex', hue='Survived', data=data, ax=ax2, palette="deep")

ax3.set_title('Sex and Pclass vs Survived')
sns.pointplot('Pclass', 'Survived', hue='Sex', data=data, ax=ax3, palette="deep")
sns.distplot(data['Age'], bins=20, kde=False, color='b')
f, (ax1, ax2) = plt.subplots(2,2,figsize=(20,15))
ax1[0].set_title('Age Distribution for Survived')
sns.distplot(data['Age'][data['Survived'] == 1], bins=20, kde=False, ax=ax1[0], color='b')

ax1[1].set_title('Age Distribution for Not Survived')
sns.distplot(data['Age'][data['Survived'] == 0], bins=20, kde=False, ax=ax1[1], color='r')

ax2[0].set_title("Sex and Age vs Survived")
ax2[0].set_yticks(range(0,110,10))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=data, split=True, palette='deep', ax=ax2[0])

ax2[1].set_title("Pclass and Age vs Survived")
ax2[1].set_yticks(range(0,110,10))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data, split=True, palette='deep', ax=ax2[1])
data.groupby('SibSp')['Survived'].mean().plot(kind='bar')
data.groupby('Parch')['Survived'].mean().plot(kind='bar')
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('Fare Distribution, Survived=1')
sns.distplot(data['Fare'][data['Survived'] == 1], ax=ax[0], bins=20)
ax[1].set_title('Fare Distribution, Survived=0')
sns.distplot(data['Fare'][data['Survived'] == 0], ax=ax[1], bins=20)
f, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].set_title('Pclass 1 Fare Distribution')
sns.distplot(data['Fare'][data['Pclass'] == 1], bins=20, kde=False, ax=ax[0], color='g')
ax[1].set_title('Pclass 2 Fare Distribution')
sns.distplot(data['Fare'][data['Pclass'] == 2], bins=20, kde=False, ax=ax[1], color='b')
ax[2].set_title('Pclass 3 Fare Distribution')
sns.distplot(data['Fare'][data['Pclass'] == 3], bins=20, kde=False, ax=ax[2], color='r')
f, ax = plt.subplots(1, 2, figsize=(12,5))
ax[0].set_title('Embarked vs Survived')
data.groupby('Embarked', sort=False)['Survived'].mean().plot(kind='bar', ax=ax[0], rot=0)
ax[1].set_title('Embarked Distribution')
data['Embarked'].value_counts().plot(kind='bar', ax=ax[1], rot=0)
plt.figure(figsize=(20,15))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0))
ax3 = plt.subplot2grid((3, 3), (1, 1))
ax4 = plt.subplot2grid((3, 3), (1, 2))

ax1.set_title('Embarked based on Pclass')
sns.countplot('Embarked', hue='Pclass', data=data, palette='deep', ax=ax1)
ax2.set_title('Embarked=S')
sns.pointplot('Pclass', 'Survived', hue='Sex', data=data[data['Embarked'] == 'S'], palette='deep', ax=ax2)
ax3.set_title('Embarked=C')
sns.pointplot('Pclass', 'Survived', hue='Sex', hue_order=['male', 'female'], data=data[data['Embarked'] == 'C'], palette='deep', ax=ax3)
ax4.set_title('Embarked=Q')
sns.pointplot('Pclass', 'Survived', hue='Sex', data=data[data['Embarked'] == 'Q'], palette='deep', ax=ax4)
data.groupby(['Embarked', 'Sex'])['Survived'].mean().plot(kind='bar')
data['NameTitle'].value_counts()
f, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('Title Distribution')
data['NameTitle'].value_counts(sort=True).plot(kind='bar', ax=ax[0], rot=0, colormap='vlag')
ax[1].set_title('Title vs Survived')
data.groupby('NameTitle')['Survived'].mean().sort_values(ascending=False).plot(kind='bar', ax=ax[1], rot=0, colormap='vlag')
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.head()
data['AgeBand'] = pd.cut(data['Age'], 5)
data.groupby('AgeBand')['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
def age_group(x):
    if x <= 16:
        return 4
    elif x > 16 and x <= 32:
        return 1
    elif x > 32 and x <= 48:
        return 2
    elif x > 48 and x <= 64:
        return 3
    else:
        return 0
data['AgeGroup'] = data['Age'].apply(age_group)
data.head()
data.drop('AgeBand', axis=1, inplace=True)
data.drop('Age', axis=1, inplace=True)
data.head()
data['FareBand'] = pd.qcut(data['Fare'], 5)
data.groupby('FareBand')['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
def fare_group(x):
    if x <= 7.854:
        return 0
    elif x > 7.854 and x <= 10.5:
        return 1
    elif x > 10.5 and x <= 21.679:
        return 2
    elif x > 21.679 and x <= 39.688:
        return 3
    else:
        return 4
data['FareGroup'] = data['Fare'].apply(fare_group)
data.head()
data.drop('Fare', axis=1, inplace=True)
data.drop('FareBand', axis=1, inplace=True)
data.head()
data['Embarked'] = data['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
data['NameTitle'] = data['NameTitle'].map({'Mr.': 0, 'Other.': 1, 'Master.': 2, 'Miss.': 3, 'Mrs.': 4}).astype(int)
data.head()
plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True)
data.drop('Sex', axis=1, inplace=True)
data.head()
data['FamilySize'] = data['SibSp'] + data['Parch']
data.head()
data.groupby('FamilySize')['Survived'].mean().plot.bar()
data.drop('SibSp', axis=1, inplace=True)
data.drop('Parch', axis=1, inplace=True)
plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True)
X = data[data.columns[1:]]
y = data[data.columns[0]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train Size: ({}, {}), Validation Size: ({}, {})".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
label = ['Linear SVM', 'Radial SVM', 'Decision Tree', 'Random Forest', 'Logistic Regression']
models = [svm.LinearSVC(), svm.SVC(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=100), LogisticRegression()]
accuracy = []
fscore = []

for model in models:
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    accuracy.append(metrics.accuracy_score(prediction, y_val))
    fscore.append(metrics.f1_score(y_val, prediction))

model_df = pd.DataFrame({
    'accuracy': accuracy,
    'fscore': fscore
}, index=label)

model_df
cv_accuracy = []
cv_std = []
kfold = KFold(n_splits=10, random_state=23)
for model in models:
    cv_result = cross_val_score(model, X, y, cv=kfold)
    cv_accuracy.append(cv_result.mean())
    cv_std.append(cv_result.std())

model_df['cv_accuracy'] = cv_accuracy
model_df['cv_std'] = cv_std
model_df
estimators = [10, 100, 500, 1000]
criterions = ['gini', 'entropy']
bootstraps = [True, False]
parameters = {'n_estimators': estimators, 'criterion': criterions, 'bootstrap': bootstraps}
gd = GridSearchCV(RandomForestClassifier(random_state=23), parameters)
gd.fit(X, y)
print("Best Score: ", gd.best_score_)
print("Best Model: ", gd.best_estimator_)
n_estimators = list(range(100,1100,100))
learn_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(random_state=23),param_grid=hyper,verbose=True)
gd.fit(X, y)
print("Best Score: ", gd.best_score_)
print("Best Model: ", gd.best_estimator_)
n_estimators = list(range(100,1100,100))
learn_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
depth = [1,2,3]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate,'max_depth':depth}
gd=GridSearchCV(estimator=GradientBoostingClassifier(random_state=23),param_grid=hyper,verbose=True)
gd.fit(X, y)
print("Best Score: ", gd.best_score_)
print("Best Model: ", gd.best_estimator_)
n_estimators = list(range(100,1100,100))
learn_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
gd=GridSearchCV(estimator=XGBClassifier(random_state=23),param_grid=hyper,verbose=True)
gd.fit(X, y)
print("Best Score: ", gd.best_score_)
print("Best Model: ", gd.best_estimator_)
f, (ax1, ax2) = plt.subplots(2,2,figsize=(15,7))

# Random Forest
model1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=23, verbose=0, warm_start=False)
model1.fit(X,y)
pd.Series(model1.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax1[0])
ax1[0].set_title('Feature Importance in Random Forests')

# Adaboost
model2 = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=100, random_state=23)
model2.fit(X,y)
pd.Series(model2.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax1[1])
ax1[1].set_title('Feature Importance in AdaBoost')

# Gradient Boosting
model3 = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.5, loss='deviance', max_depth=2,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=300,
              presort='auto', random_state=23, subsample=1.0, verbose=0,
              warm_start=False)
model3.fit(X,y)
pd.Series(model3.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax2[0])
ax2[0].set_title('Feature Importance in Gradient Boosting')

# XGBoost
model4 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.3, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=200,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=23,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
model4.fit(X,y)
pd.Series(model4.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax2[1])
ax2[1].set_title('Feature Importance in XGBoost')
test = pd.read_csv('../input/test.csv')
test.head()
test.isnull().sum()
# extract the title
test['NameTitle'] = test['Name'].apply(extract_title)
fix_title(test)

# filling missing values for age
test.loc[test['Age'].isnull() & test['NameTitle'].str.contains('Master.'), 'Age'] = 5
test.loc[test['Age'].isnull() & test['NameTitle'].str.contains('Miss.'), 'Age'] = 22
test.loc[test['Age'].isnull() & test['NameTitle'].str.contains('Mr.'), 'Age'] = 32
test.loc[test['Age'].isnull() & test['NameTitle'].str.contains('Mrs.'), 'Age'] = 36
test.loc[test['Age'].isnull() & test['NameTitle'].str.contains('Other.'), 'Age'] = 46
test['Age'].isnull().any()

# binning the age
test['AgeGroup'] = test['Age'].apply(age_group)

# filling missing value for fare
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

# binning the fare
test['FareGroup'] = test['Fare'].apply(fare_group)

# convert categorical into numeric
test['Embarked'] = test['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
test['NameTitle'] = test['NameTitle'].map({'Mr.': 0, 'Other.': 1, 'Master.': 2, 'Miss.': 3, 'Mrs.': 4, 'Dona.': 1}).astype(int)
test.head()
# FamilySize
test['FamilySize'] = test['SibSp'] + test['Parch']
test.head()
# dropping some unused columns
test_id = test[test.columns[0]]
test.drop('PassengerId', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
test.drop('Sex', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)
test.drop('Fare', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)
test.drop('SibSp', axis=1, inplace=True)
test.drop('Parch', axis=1, inplace=True)
test.head()
prediction = model3.predict(test)
submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": prediction
})
submission.to_csv('submission.csv', index=False)
