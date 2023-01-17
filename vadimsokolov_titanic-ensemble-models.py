import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix

from sklearn.calibration import CalibratedClassifierCV





from vecstack import stacking



from imblearn.over_sampling import SMOTE



from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')
# uploading data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.info()
train.describe()
test.head()
test.info()
test.describe()
def merge_data(train, test):

    return pd.concat([train, test], sort = True).reset_index(drop=True)

def divide_data(data):

    return data.iloc[:891], data.iloc[891:].drop(['Survived'], axis = 1)



data = merge_data(train, test)
def countplot(y, hue, title):

    plt.figure(figsize = (12, 6))

    plt.title(title)

    sns.set()

    sns.countplot(y = y, hue = hue)

    plt.grid()

countplot(train['Pclass'], train['Survived'], 

          'The part of passengers who survived, depending on the Pclass')

countplot(train['Embarked'], train['Survived'], 

          'The part of passengers who survived, depending on the Embarked')
countplot(train['Sex'], train['Survived'], 'The part of passengers who survived, depending on the Sex')
countplot(train['Parch'], train['Survived'], 'The part of passengers who survived, depending on the Parch')
countplot(train['SibSp'], train['Survived'], 'The part of passengers who survived, depending on the SibSp')
def kdeplot(feature, xlabel, title):

    plt.figure(figsize = (12, 8))

    ax = sns.kdeplot(train[feature][(train['Survived'] == 0) & 

                             (train[feature].notnull())], color = 'lightcoral', shade = True)

    ax = sns.kdeplot(train[feature][(train['Survived'] == 1) & 

                             (train[feature].notnull())], color = 'darkturquoise', shade= True)

    plt.xlabel(xlabel)

    plt.ylabel('frequency')

    plt.title(title)

    ax.legend(['not survived','survived'])

    

kdeplot('Age', 'age', 'The distribution of the surviving passengers depending on the Age')
def boxplot(x, y, title):

    plt.figure(figsize = (12, 8))

    sns.boxplot(x = x, y = y)

    plt.title(title)

boxplot(train['Survived'], train['Age'], 'The boxplot for Age')
kdeplot('Fare', 'fare', 'The distribution of the surviving passengers depending on the Fare')
boxplot(train['Survived'], train['Fare'], 'The boxplot for Fare')
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

data['Title'].unique()
# Exchange many titles with a more common name or classify them as Rare.

data.groupby('Title')['Sex'].count()



data['Title'] = data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 

                                      'Jonkheer', 'Major', 'Sir', 'Rev', 'Dona'], 'Rare')

data['Title'] = data['Title'].replace(['Lady', 'Mlle', 'Mme', 'Ms'], 

                                      ['Mrs', 'Miss', 'Miss', 'Mrs'])
# create a new features - family survival!

def family_survival():



    data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])



    default_survival_rate = 0.5

    data['Family_survival'] = default_survival_rate



    for grp, grp_df in data[['Survived', 'Name', 'Last_Name', 

                                 'Fare', 'Ticket', 'PassengerId',

                                 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):



        if (len(grp_df) != 1):

            for ind, row in grp_df.iterrows():

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                ID = row['PassengerId']

                if (smax == 1.0):

                    data.loc[data['PassengerId'] == ID, 

                                 'Family_survival'] = 1

                elif (smin == 0.0):

                    data.loc[data['PassengerId'] == ID, 

                                 'Family_survival'] = 0



    for _, grp_df in data.groupby('Ticket'):

        if (len(grp_df) != 1):

            for ind, row in grp_df.iterrows():

                if (row['Family_survival'] == 0) | (

                        row['Family_survival'] == 0.5):

                    smax = grp_df.drop(ind)['Survived'].max()

                    smin = grp_df.drop(ind)['Survived'].min()

                    ID = row['PassengerId']

                    if (smax == 1.0):

                        data.loc[data['PassengerId'] == ID, 

                                     'Family_survival'] = 1

                    elif (smin == 0.0):

                        data.loc[data['PassengerId'] == ID, 

                                     'Family_survival'] = 0



    return data
data = family_survival()
# Creating a new colomn - Name_length

#data['Name_length'] = data['Name'].apply(len)





#def name_length_category(length):

#    if length <= 20:

#        return 0

#    if 20 < length <= 35:

#        return 1

#    if 35 < length <= 45:

#        return 2

#    else:

#        return 3



#data['Name_length'] = data['Name_length'].apply(name_length_category)
data['Age'] = data.groupby(['Title', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# create age groups



def age_category(age):

    if age <=2:

        return 0

    if 2 < age <= 18:

        return 1

    if 18 < age <= 35:

        return 2

    if 35 < age <= 65:

        return 3

    else:

        return 4

data['Age'] = data['Age'].apply(age_category)
data['Age*Pclass'] = data['Age']*data['Pclass']
data[data['Fare'].isnull()]
# replace the fare value

data.loc[data['Fare'].isnull(), 'Fare'] = data.loc[(data['Embarked'] == 'S') 

                                                   & (data['Pclass'] == 3) & (data['SibSp'] == 0)]['Fare'].median()
data['Fare'].value_counts()
# create fare groups

def fare_category(fare):

    if fare <= 7.91:

        return 0

    if 7.91 < fare <= 14.454:

        return 1

    if 14.454 < fare <= 31:

        return 2

    if 31 < fare <= 99:

        return 3

    if 99 < fare <= 250:

        return 4

    else:

        return 5

data['Fare'] = data['Fare'].apply(fare_category)
data[data['Embarked'].isnull()]
data.loc[(data['Fare'] < 80) & (data['Pclass'] == 1)]['Embarked'].value_counts()

# => both embarked in Southhampton

data.loc[data['Embarked'].isnull(), 'Embarked'] = 'S'
#print(data['Cabin'].unique())

#print('Count unique values:', data['Cabin'].nunique())

# keep all first letters of cabin and use 'N' for each missing values

#data['Cabin'] = data['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'N')
#data.groupby('Cabin')['Survived'].mean().sort_values(ascending = False)#

# group together the values

#data.loc[data['Cabin'] == 'T', 'Cabin'] = 'A'

#data['Cabin'] = data['Cabin'].replace(['A', 'B', 'C'], 'ABC')

#data['Cabin'] = data['Cabin'].replace(['D', 'E'], 'DE')

#data['Cabin'] = data['Cabin'].replace(['F', 'G'], 'FG')

#data['Cabin'].value_counts()
#data['Ticket'].value_counts()

# let's try creating a string length as feature

#data['Ticket_length'] = data['Ticket'].apply(len)

# create ticket category

def ticket_category(Ticket_length):

    if Ticket_length <= 6:

        return 0

    if 6 < Ticket_length <= 10:

        return 1

    else:

        return 2

#data['Ticket_length'] = data['Ticket_length'].apply(ticket_category)
# creating a feature Family_size

data['Family_size'] = data['Parch'] + data['SibSp'] + 1
data[['Family_size', 'Survived']].groupby('Family_size').mean()
# create a family_size category

data['Single'] = data['Family_size'].map(lambda x: 1 if x == 1 else 0)

#data['Small_family'] = data['Family_size'].map(lambda x: 1 if 2 <= x <= 4 else 0)

#data['Medium_family'] = data['Family_size'].map(lambda x: 1 if 5 <= x <= 7 else 0)

#data['Large_family'] = data['Family_size'].map(lambda x: 1 if x > 7 else 0)
data['Is_Married'] = 0

data['Is_Married'] = data['Title'].map(lambda x: 1 if x == 'Mrs' else 0)
data = data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Last_Name'], axis = 1)
# label encoder

def object_to_int(df):

    if df.dtype=='object':

        df = LabelEncoder().fit_transform(df)

    return df

data = data.apply(lambda x: object_to_int(x))
# correlation matrix

plt.figure(figsize = (12, 8))

sns.heatmap(data.corr(), annot = True)

plt.title('Correlation matrix')
# one-hot encoder

#ohe_columns = data.drop(['Age'], axis = 1)

data= pd.get_dummies(data, columns = ['Age', 'Title', 'Embarked'], drop_first = True)
# add polynomial features

def add_polynomial_features(frame, poly_degree=2, interaction=False):

    poly = PolynomialFeatures(degree = poly_degree, interaction_only = interaction, include_bias = False)

    poly_features = poly.fit_transform(frame[['Age', 'Name_length', 'Fare']])

    df_poly = pd.DataFrame(poly_features, columns = poly.get_feature_names())

    return pd.concat([frame, df_poly.drop(['x0'], axis=1)], axis=1)

#data = add_polynomial_features(data, 3, False)
train, test = divide_data(data)
X = train.drop(['Survived'], axis = 1)

y = train['Survived']



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 12345)
# OverSampling

#sm = SMOTE()

#X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
#numeric = ['Age', 'Name_length', 'Fare', 'Age*Pclass']

#for i in X_train.columns:

#    if X_train[i].dtype =='float64' or X_train[i].dtype =='float32':

#        numeric += [i]

#scaler = StandardScaler()

#scaler.fit(X_train[numeric])

#X_train[numeric] = scaler.transform(X_train[numeric])

#X_valid[numeric] = scaler.transform(X_valid[numeric])

#test[numeric] = scaler.transform(test[numeric])
# create functions for confusion matrix and feature importance

def confusion_m(model, title):

    cm = confusion_matrix(y_valid, model.predict(X_valid))

    f, ax = plt.subplots(figsize = (8, 6))

    sns.heatmap(cm, annot = True, linewidths = 0.5, color = 'red', fmt = '.0f', ax = ax)

    plt.xlabel('y_predicted')

    plt.ylabel('y_true')

    plt.title(title)

    plt.show()



def feature_importance(model, title):

    dataframe = pd.DataFrame(model, X_train.columns).reset_index()

    dataframe = dataframe.rename(columns = {'index':'features', 0:'coefficients'})

    dataframe = dataframe.sort_values(by = 'coefficients', ascending = False)

    plt.figure(figsize=(13,10), dpi= 60)

    ax = sns.barplot(x = 'coefficients', y = 'features', data = dataframe ,palette = 'husl')

    plt.title(title, fontsize = 20)
lr = LogisticRegression(random_state = 12345)

parameters_lr = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 3, 5, 7, 10, 15, 20, 25, 30, 50], 

                 'penalty':['l1', 'l2', 'elasticnet', 'none'],

                 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

                 'class_weight': [1, 3, 10],

                 'max_iter': [200, 500, 800, 1000, 2000]}

search_lr = RandomizedSearchCV(lr, parameters_lr, cv=5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_lr.fit(X_train, y_train)

best_lr = search_lr.best_estimator_

predict_lr = best_lr.predict(X_valid)

auc_lr = cross_val_score(best_lr, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_lr = cross_val_score(best_lr, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for Logistic Regression on validation dataset:', sum(auc_lr)/len(auc_lr))

print('Accuracy for Logistic Regression on validation dataset:', sum(acc_lr)/len(acc_lr))
confusion_m(best_lr, 'Confusion matrix for Logistic Regression')

feature_importance(best_lr.coef_[0], 'Feature importance for Logistic Regression')
dt = DecisionTreeClassifier(random_state = 12345)

parameters_dt = {'criterion': ['gini', 'entropy'], 

                 'max_depth':range(1, 100, 1), 

                 'min_samples_leaf': range(1, 20), 

                 'max_features':range(1, X_train.shape[1]+1)}

search_dt = RandomizedSearchCV(dt, parameters_dt, cv=5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_dt.fit(X_train, y_train)

best_dt = search_dt.best_estimator_

acc_dt = cross_val_score(best_dt, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1)   

auc_dt = cross_val_score(best_dt, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

print('AUC-ROC for Decision Tree on validation dataset:', sum(auc_dt)/len(auc_dt))

print('Accuracy for Decision Tree on validation dataset:', sum(acc_dt)/len(acc_dt))
confusion_m(best_dt, 'Confusion matrix for Logistic Regression')

feature_importance(best_dt.feature_importances_, 'Feature importance for Decision Tree')
rf = RandomForestClassifier(random_state = 12345)

parameters_rf = {'n_estimators': range(1, 1800, 25), 

                 'criterion': ['gini', 'entropy'], 

                 'max_depth':range(1, 100), 

                 'min_samples_split': range(1, 12), 

                 'min_samples_leaf': range(1, 12), 

                 'max_features':['auto', 'log2', 'sqrt', 'None']}

search_rf = RandomizedSearchCV(rf, parameters_rf, cv=5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)



search_rf.fit(X_train, y_train)

best_rf = search_rf.best_estimator_

predict_rf = best_rf.predict(X_valid)

auc_rf = cross_val_score(best_rf, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_rf = cross_val_score(best_rf, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1) 

print('AUC-ROC for Random Forest on validation dataset:', sum(auc_rf)/len(auc_rf))

print('Accuracy for Random Forest on validation dataset:', sum(acc_rf)/len(acc_rf))
confusion_m(best_rf, 'Confusion matrix for Random Forest')

feature_importance(best_rf.feature_importances_, 'Feature importance for Random Forest')
xgb = XGBClassifier(random_state = 12345, eval_metric='auc')

parameters_xgb = {'eta': [0.01, 0.05, 0.1, 0.001, 0.005, 0.04, 0.2, 0.0001],  

                  'min_child_weight':range(1, 5), 

                  'max_depth':range(1, 6), 

                  'learning_rate': [0.01, 0.05, 0.1, 0.001, 0.005, 0.04, 0.2], 

                  'n_estimators':range(0, 2001, 50)}

search_xgb = RandomizedSearchCV(xgb, parameters_xgb, cv = 5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_xgb.fit(X_train, y_train)

best_xgb = search_xgb.best_estimator_

predict_xgb = best_xgb.predict(X_valid)

auc_xgb = cross_val_score(best_xgb, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_xgb = cross_val_score(best_xgb, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for XGBoost on validation dataset:', sum(auc_xgb)/len(auc_xgb))

print('Accuracy for XGBoost on validation dataset:', sum(acc_xgb)/len(acc_xgb))
confusion_m(best_xgb, 'Confusion matrix for XGBoost')

feature_importance(best_xgb.feature_importances_, 'Feature importance for XGBoost')
cb = CatBoostClassifier(random_state = 12345, iterations = 300, eval_metric='Accuracy', verbose = 100)

parameters_cb = {'depth': range(6, 11),

                 'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.0001]}

search_cb = RandomizedSearchCV(cb, parameters_cb, cv = 5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_cb.fit(X_train, y_train, verbose = 100)

best_cb = search_cb.best_estimator_

predict_cb = best_cb.predict(X_valid)

auc_cb = cross_val_score(best_cb, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_cb = cross_val_score(best_cb, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for CatBoost on validation dataset:', sum(auc_cb)/len(auc_cb))

print('Accuracy for CatBoost on validation dataset:', sum(acc_cb)/len(acc_cb))
confusion_m(best_cb, 'Confusion matrix for CatBoost')

feature_importance(best_cb.feature_importances_, 'Feature importance for CatBoost')
et = ExtraTreesClassifier(random_state = 12345)

parameters_et = {'n_estimators': range(50, 501, 25), 

                 'criterion': ['gini', 'entropy'], 

                 'max_depth':range(1, 100), 

                 'min_samples_split': range(1, 12), 

                 'min_samples_leaf': range(1, 12), 

                 'max_features':['auto', 'log2', 'sqrt', 'None']}

search_et = RandomizedSearchCV(et, parameters_et, cv=5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_et.fit(X_train, y_train)

best_et = search_et.best_estimator_

predict_et = best_et.predict(X_valid)

auc_et = cross_val_score(best_et, X_valid, y_valid, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_et = cross_val_score(best_et, X_valid, y_valid, scoring = 'accuracy', cv = 10, n_jobs = -1) 

print('AUC-ROC for Extra Trees on validation dataset:', sum(auc_et)/len(auc_et))

print('Accuracy for Extra Trees on validation dataset:', sum(acc_et)/len(acc_et))
confusion_m(best_et, 'Confusion matrix for ExtraTreesClassifier')

feature_importance(best_et.feature_importances_, 'Feature importance for ExtraTreesClassifier')
vc = VotingClassifier(estimators=[('lr', best_lr), ('xgb', best_xgb), ('rf', best_rf), ('et', best_et), ('cb', best_cb)], voting='soft')

vc.fit(X_train, y_train)

predict_vc = vc.predict(X_valid)

auc_vc = cross_val_score(vc, X_valid, y_valid, scoring = 'roc_auc', cv = 5, n_jobs = -1)

acc_vc = cross_val_score(vc, X_valid, y_valid, scoring = 'accuracy', cv = 5, n_jobs = -1)

print('AUC-ROC for ensemble models on validation dataset:', sum(auc_vc)/len(auc_vc))

print('Accuracy for ensemble models on validation dataset:', sum(acc_vc)/len(acc_vc))
confusion_m(vc, 'Confusion matrix for VotingClassifier')
models = ['logistic_regression', 'decision_tree', 'random_forest',

          'xgboost', 'catboost', 

          'extra_trees', 'voting']

dict_values = {'accuracy': [acc_lr.mean(), acc_dt.mean(), acc_rf.mean(),

                            acc_xgb.mean(), acc_cb.mean(), 

                            acc_et.mean(), acc_vc.mean()],

               'auc_roc': [auc_lr.mean(), auc_dt.mean(), auc_rf.mean(),

                           auc_xgb.mean(), auc_cb.mean(), 

                           auc_et.mean(), auc_vc.mean()]}

df_score = pd.DataFrame(dict_values, index = models, columns = ['accuracy', 'auc_roc'])

df_score
# stacking

#models = [best_rf, best_cb, best_et]

#

#S_train, S_test = stacking(models, X_train, y_train, test, regression=False, mode='oof_pred_bag', 

#                           needs_proba=False, save_dir=None, metric=accuracy_score, n_folds=4, 

#                           stratified=True, shuffle=True, random_state=12345, verbose=2)
#final_model = best_rf

#final_model.fit(S_train, y_train)
# Out-of-fold predictions

X_train = X_train.values

test = test.values



ntrain = X_train.shape[0]

ntest = test.shape[0]

n_folds = 5 



kf = KFold(n_splits = n_folds, random_state = 12345)



def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((n_folds, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train.iloc[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
et_oof_train, et_oof_test = get_oof(best_et, X_train, y_train, test) 

rf_oof_train, rf_oof_test = get_oof(best_rf,X_train, y_train, test) 

cb_oof_train, cb_oof_test = get_oof(best_cb, X_train, y_train, test)

xgb_oof_train, xgb_oof_test = get_oof(best_xgb,X_train, y_train, test) 

lr_oof_train, lr_oof_test = get_oof(best_lr,X_train, y_train, test) 
X_train = np.concatenate((et_oof_train, rf_oof_train, cb_oof_train, xgb_oof_train, lr_oof_train), axis=1)

X_test = np.concatenate((et_oof_test, rf_oof_test, cb_oof_test, xgb_oof_test, lr_oof_test), axis=1)
best_model = XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, 

                    subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 

                    nthread= -1,scale_pos_weight=1).fit(X_train, y_train)
# predict the values for test dataset

y_test = best_model.predict(X_test)
# create and save predict dataframe

submission = pd.DataFrame({'PassengerId': list(range(892, 1310)), 'Survived': y_test})

submission['Survived'] = submission['Survived'].astype(int)

submission.to_csv('submission.csv', index=False)

print(submission)