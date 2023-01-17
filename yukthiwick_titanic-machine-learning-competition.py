import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy import stats


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd. read_csv('/kaggle/input/titanic/test.csv')
train.info()
train['Survived'].value_counts()
train.head(25)
# Dropping Columns
train = train.drop(columns=['Ticket'])
test = test.drop(columns=['Ticket'])
train.isna().sum()
test.isna().sum()
age_na = train[train['Age'].isna()]
age_na['Pclass'].value_counts()
age_means = {}
age_means[1] = round(train[train['Pclass'] == 1]['Age'].mean())
age_means[2] = round(train[train['Pclass'] == 2]['Age'].mean())
age_means[3] = round(train[train['Pclass'] == 3]['Age'].mean())
age_means
train['Age'] = train.apply(
    lambda row: age_means[row['Pclass']] if np.isnan(row['Age']) else row['Age'],
    axis=1
)

test['Age'] = test.apply(
    lambda row: age_means[row['Pclass']] if np.isnan(row['Age']) else row['Age'],
    axis=1
)
test[test['Fare'].isna()]
train[train['Fare'] == 0]
test[test['Fare'] == 0]
# Filling in the null fare value with the median fare for 3rd class
median_third_class = test[test['Pclass']==3]['Fare'].median()
test['Fare'] = test['Fare'].fillna(median_third_class)
# Defining a function that will find the median fare for the desired PClass as we itterate through both the train and test sets
def train_fare(row):
    fares = {1: train[train['Pclass'] == 1].median(),
             2: train[train['Pclass'] == 2].median(),
             3: train[train['Pclass'] == 3].median()}
    return(fares[row['Pclass']])

def test_fare(row):
    fares = {1: test[test['Pclass'] == 1].median(),
             2: test[test['Pclass'] == 2].median(),
             3: test[test['Pclass'] == 3].median()}
    return(fares[row['Pclass']])
medians = train[train['Fare'] == 0].apply(train_fare, axis=1)['Fare']
medians_test = test[test['Fare'] == 0].apply(test_fare, axis=1)['Fare']
train.loc[train['Fare'] == 0,'Fare'] = medians
test.loc[test['Fare'] == 0,'Fare'] = medians_test
# Changing Cabin to a binary column based on if value is null or not
train.loc[train['Cabin'].notnull(),'Cabin'] = 1
train.loc[train['Cabin'].isnull(),'Cabin'] = 0
train['Cabin'] = train['Cabin'].astype('int')

test.loc[test['Cabin'].notnull(),'Cabin'] = 1
test.loc[test['Cabin'].isnull(),'Cabin'] = 0
test['Cabin'] = test['Cabin'].astype('int')
train[train['Embarked'].isna()]
train.groupby(['Pclass','Embarked']).count()
# Most first class passengers embarked from Southampton so we will fill the missing embarked values in the Train set with 'S'

train['Embarked'] = train['Embarked'].fillna('S')
# Changing Pclass and Embarked to categorical using One-hot encoding
train[['Pclass', 'Embarked']]= train[['Pclass', 'Embarked']].astype('category')
test[['Pclass', 'Embarked']] = test[['Pclass', 'Embarked']].astype('category')

dummy_class_train = pd.get_dummies(train['Pclass'])
dummy_class_test = pd.get_dummies(test['Pclass'])
train = pd.concat([train, dummy_class_train], axis=1)
test = pd.concat([test, dummy_class_test], axis=1)


dummy_emb_train = pd.get_dummies(train['Embarked'])
dummy_emb_test = pd.get_dummies(test['Embarked'])
train = pd.concat([train, dummy_emb_train], axis=1)
test = pd.concat([test, dummy_emb_test], axis=1)

# Changing Sex to a binary column
train['Sex'] = train['Sex'].apply(lambda x: 1 if x=='male' else 0)
test['Sex'] = test['Sex'].apply(lambda x: 1 if x=='male' else 0)

# Extracting titles from names
train['Title'] = train['Name'].str.extract(r',\s(.+?)\.')
test['Title'] = test['Name'].str.extract(r',\s(.+?)\.')
train['Title'].value_counts()
# Renaming rare titles as "Rare", making Title column categorical
common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
train.loc[~train["Title"].isin(common_titles), "Title"] = "Rare"
test.loc[~test["Title"].isin(common_titles), "Title"] = "Rare"

train['Title'] = train['Title'].astype('category')
test['Title'] = test['Title'].astype('category')

dummy_title_train = pd.get_dummies(train['Title'])
dummy_title_test = pd.get_dummies(test['Title'])
train = pd.concat([train, dummy_title_train], axis=1)
test = pd.concat([test, dummy_title_test], axis=1)

numerical_train = train.select_dtypes(exclude = ['object'])
corr_mat = numerical_train.corr()

sns.heatmap(corr_mat, center = 0)
g = sns.distplot(train['Fare'])
train['Fare'].skew()
g = sns.distplot(test['Fare'])
test['Fare'].skew()
# Comparing log transform and box-cox transform to fix skew

box_cox = stats.boxcox(train['Fare'])[0]
g = sns.distplot(box_cox)
pd.Series(box_cox).skew()
log_fare = train['Fare'].apply(lambda x:np.log(x))
g = sns.distplot(log_fare)
log_fare.skew()
# With the box-cox transform clearly fixing the skew best we will use it to fix our fare column
train['box_fare'] = box_cox
test['box_fare'] = stats.boxcox(test['Fare'])[0]
g = sns.distplot(test['box_fare'])
test['box_fare'].skew()
g = sns.catplot(x="Embarked", y="Survived", data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
g = sns.catplot(x="Title", y="Survived",data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
g = sns.catplot(x="SibSp", y="Survived",data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
    
g = sns.catplot(x="Sex", y="Survived",data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")
g = sns.catplot(x="Parch", y="Survived",data=train,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("survival probability")   
features = train.columns
features = features.drop(['PassengerId','Survived', 'Pclass', 'Title', 'Fare', 'Name', 'Embarked'])
train[features].head()
# Defining pipeline
def random_search(model, parameter_grid, cvn, train_data, feature_list):
    clf = RandomizedSearchCV(estimator = model, param_distributions = parameter_grid, cv=cvn, return_train_score = False, n_jobs = -1)
    clf.fit(train_data[feature_list], train_data['Survived'])
    return clf

def grid_search(model, parameter_grid, cvn, train_data, feature_list):
    clf = GridSearchCV(estimator = model, param_grid= parameter_grid, cv=cvn, return_train_score = False, n_jobs = -1)
    clf.fit(train_data[feature_list], train_data['Survived'])
    return clf

def create_predictions(model, train_data,test_data, features, csv_name):
    model.fit(train_data[features], train_data['Survived'])
    predictions = model.predict(test_data[features])
    submission = pd.concat([test_data['PassengerId'], pd.Series(predictions)], axis=1)
    submission = submission.rename(columns = {0:'Survived'})
    submission.to_csv("./" + csv_name, index=False)
    
param_grid_rf = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

param_grid_logit = {'penalty': ['l1', 'l2'],
                    'solver' : ['lbfgs', 'liblinear']    
}

param_grid_xgb = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


rf = RandomForestClassifier(random_state = 123)
lr = LogisticRegression(random_state=123)
clf = xgb.XGBClassifier(random_state=123)
rf_opt = random_search(rf, param_grid_rf, 4, train, features)
best_rf_params = rf_params.best_params_
create_predictions(rf_opt, train, test, features, "rf_opt.csv")
rf_opt.fit(train[features], train['Survived'])
predictions_rf = rf_opt.predict(train[features])

rf_accuracy = accuracy_score(train['Survived'], predictions_rf)
lr_opt = grid_search(lr, param_grid_logit, 4, train, features)
lr_opt.fit(train[features], train['Survived'])
lr_opt.best_params_
lr_predictions = lr_opt.predict(train[features])
lr_accuracy = accuracy_score(train['Survived'], lr_predictions)
lr_accuracy
create_predictions(lr_opt, train, test, features, "lr_opt.csv")
xgb_opt = random_search(clf, param_grid_xgb, 4, train, features)
xgb_opt.fit(train[features], train['Survived'])
xgb_params = xgb_opt.best_params_
xgb_predictions = xgb_opt.predict(train[features])
xgb_accuracy= accuracy_score(train['Survived'], xgb_predictions)

xgb_opt = xgb.XGBClassifier(**xgb_params, random_state = 123)
create_predictions(xgb_opt, train, test, features, "xgb_opt.csv")