import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix



from imblearn.over_sampling import SMOTE



from xgboost import XGBClassifier

from catboost import CatBoostClassifier

import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
data.info()
data.describe()
plt.figure(figsize=(20,12), dpi= 60)

plt.title('Distribution of Outcome variable')

plt.pie(data['Outcome'].value_counts(), labels = ['healthy','diabetic'], colors = ['gold', 'lightcoral'], autopct='%1.1f%%', shadow=True, startangle=140)

plt.show()
for i in data.columns:

    print(i, data[i][data[i] == 0].count())
plt.figure(figsize = (12, 8))

ax = sns.boxplot(data = data, orient = 'h', palette = 'Set2')

plt.title('Boxplot overview dataset')

plt.xlabel('values')

plt.xlim(-3, 300)

plt.show()
plt.figure(figsize = (12, 8))

sns.heatmap(data.corr(), annot = True)

plt.title('Correlation matrix')

plt.show()
def median_feature(feature):

    temp = data[data[feature] > 0]

    med_cat = temp.groupby('Outcome')[feature].median().reset_index()

    return med_cat
def preparing_feature(feature, median_data):

    data.loc[(data['Outcome'] == 0) & (data[feature] == 0), feature] = median_data[median_data['Outcome'] == 0][feature].median()

    data.loc[(data['Outcome'] == 1) & (data[feature] == 0), feature] = median_data[median_data['Outcome'] == 1][feature].median()
def kdeplot(feature, xlabel, title):

    plt.figure(figsize = (12, 8))

    ax = sns.kdeplot(data[feature][(data['Outcome'] == 0) & 

                             (data[feature].notnull())], color = 'darkturquoise', shade = True)

    ax = sns.kdeplot(data[feature][(data['Outcome'] == 1) & 

                             (data[feature].notnull())], color = 'lightcoral', shade= True)

    plt.xlabel(xlabel)

    plt.ylabel('frequency')

    plt.title(title)

    ax.grid()

    ax.legend(['healthy','diabetic'])

kdeplot('Glucose', 'concentration', 'Glucose')
median_feature_glucose = median_feature('Glucose')

median_feature_glucose
preparing_feature('Glucose', median_feature_glucose)
kdeplot('Insulin', 'mu U/ml', 'Insulin')
median_feature_insulin = median_feature('Insulin')

median_feature_insulin
data['Insulin'] = data['Insulin'].astype('float')

preparing_feature('Insulin', median_feature_insulin)
kdeplot('BloodPressure', 'mm Hg', 'BloodPressure')
median_feature_bpressure = median_feature('BloodPressure')

median_feature_bpressure
data['BloodPressure'] = data['BloodPressure'].astype('float')

preparing_feature('BloodPressure', median_feature_bpressure)
kdeplot('SkinThickness', 'mm', 'SkinThickness')
median_feature_skinthickness = median_feature('SkinThickness')

median_feature_skinthickness
preparing_feature('SkinThickness', median_feature_skinthickness)
kdeplot('BMI', 'weight in kg/(height in m)^2', 'BMI')
median_feature_bmi = median_feature('BMI')

median_feature_bmi
preparing_feature('BMI', median_feature_bmi)
for i in data.columns:

    print(i, data[i][data[i] == 0].count())
kdeplot('Age', 'years', 'Age')
kdeplot('DiabetesPedigreeFunction', 'diabetes pedigree function', 'DiabetesPedigreeFunction')
kdeplot('Pregnancies', 'number of times pregnant', 'Pregnancies')
X = data.drop(['Outcome'], axis = 1)

y = data['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 12345)
numeric = []

for i in X_train.columns:

        numeric += [i]

scaler = StandardScaler()

scaler.fit(X_train[numeric])

X_train[numeric] = scaler.transform(X_train[numeric])

X_test[numeric] = scaler.transform(X_test[numeric])
def confusion_m(model, title):

    cm = confusion_matrix(y_test, model.predict(X_test))

    f, ax = plt.subplots(figsize = (8, 6))

    sns.heatmap(cm, annot = True, linewidths = 0.5, cmap = 'Greens', fmt = '.0f', ax = ax)

    plt.xlabel('y_predicted')

    plt.ylabel('y_true')

    plt.title(title)

    plt.show()



def feature_importance(model, title):

    dataframe = pd.DataFrame(model, X_train.columns).reset_index()

    dataframe = dataframe.rename(columns = {'index':'features', 0:'coefficients'})

    dataframe = dataframe.sort_values(by = 'coefficients', ascending = False)

    plt.figure(figsize=(13,10), dpi= 60)

    ax = sns.barplot(x = 'coefficients', y = 'features', data = dataframe ,palette = 'viridis')

    plt.title(title, fontsize = 20)

    plt.grid()
lr = LogisticRegression(random_state = 12345)

parameters_lr = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 3, 5, 7, 10, 15, 20, 25, 30, 50], 

                 'penalty':['l1', 'l2', 'elasticnet', 'none'],

                 'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

                 'class_weight': [1, 3, 10],

                 'max_iter': [200, 500, 800, 1000, 2000]}

search_lr = RandomizedSearchCV(lr, parameters_lr, cv=5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_lr.fit(X_train, y_train)

best_lr = search_lr.best_estimator_

predict_lr = best_lr.predict(X_test)

auc_lr = cross_val_score(best_lr, X_test, y_test, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_lr = cross_val_score(best_lr, X_test, y_test, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for Logistic Regression on test dataset:', sum(auc_lr)/len(auc_lr))

print('Accuracy for Logistic Regression on test dataset:', sum(acc_lr)/len(acc_lr))
feature_importance(best_lr.coef_[0], 'Feature importance for Logistic Regression')

confusion_m(best_lr, 'Confusion matrix for Logistic Regression')
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

predict_rf = best_rf.predict(X_test)

auc_rf = cross_val_score(best_rf, X_test, y_test, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_rf = cross_val_score(best_rf, X_test, y_test, scoring = 'accuracy', cv = 10, n_jobs = -1) 

print('AUC-ROC for Random Forest on test dataset:', sum(auc_rf)/len(auc_rf))

print('Accuracy for Random Forest on test dataset:', sum(acc_rf)/len(acc_rf))
feature_importance(best_rf.feature_importances_, 'Feature importance for Random Forest')

confusion_m(best_rf, 'Confusion matrix for Random Forest')
xgb = XGBClassifier(random_state = 12345, eval_metric='auc')

parameters_xgb = {'eta': [0.01, 0.05, 0.1, 0.001, 0.005, 0.04, 0.2, 0.0001],  

                  'min_child_weight':range(1, 5), 

                  'max_depth':range(1, 6), 

                  'learning_rate': [0.01, 0.05, 0.1, 0.001, 0.005, 0.04, 0.2], 

                  'n_estimators':range(0, 2001, 50)}

search_xgb = RandomizedSearchCV(xgb, parameters_xgb, cv = 5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_xgb.fit(X_train, y_train)

best_xgb = search_xgb.best_estimator_

predict_xgb = best_xgb.predict(X_test)

auc_xgb = cross_val_score(best_xgb, X_test, y_test, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_xgb = cross_val_score(best_xgb, X_test, y_test, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for XGBoost on test dataset:', sum(auc_xgb)/len(auc_xgb))

print('Accuracy for XGBoost on test dataset:', sum(acc_xgb)/len(acc_xgb))
feature_importance(best_xgb.feature_importances_, 'Feature importance for XGBoost')

confusion_m(best_xgb, 'Confusion matrix for XGBoost')
cb = CatBoostClassifier(random_state = 12345, iterations = 300, eval_metric='Accuracy', verbose = 100)

parameters_cb = {'depth': range(6, 11),

                 'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.0001]}

search_cb = RandomizedSearchCV(cb, parameters_cb, cv = 5, scoring = 'accuracy', n_jobs = -1, random_state = 12345)

search_cb.fit(X_train, y_train, verbose = 100)

best_cb = search_cb.best_estimator_

predict_cb = best_cb.predict(X_test)

auc_cb = cross_val_score(best_cb, X_test, y_test, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_cb = cross_val_score(best_cb, X_test, y_test, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for CatBoost on test dataset:', sum(auc_cb)/len(auc_cb))

print('Accuracy for CatBoost on test dataset:', sum(acc_cb)/len(acc_cb))
feature_importance(best_cb.feature_importances_, 'Feature importance for CatBoost')

confusion_m(best_cb, 'Confusion matrix for CatBoost')
vc = VotingClassifier(estimators=[('lr', best_lr), ('xgb', best_xgb), ('rf', best_rf), ('cb', best_cb)], voting='soft')

vc.fit(X_train, y_train)

predict_vc = vc.predict(X_test)

auc_vc = cross_val_score(vc, X_test, y_test, scoring = 'roc_auc', cv = 10, n_jobs = -1)

acc_vc = cross_val_score(vc, X_test, y_test, scoring = 'accuracy', cv = 10, n_jobs = -1)

print('AUC-ROC for ensemble models on test dataset:', sum(auc_vc)/len(auc_vc))

print('Accuracy for ensemble models on test dataset:', sum(acc_vc)/len(acc_vc))
confusion_m(vc, 'Confusion matrix for VotingClassifier')
models = ['logistic_regression', 'random_forest',

          'xgboost', 'catboost', 'voting']

dict_values = {'auc_roc': [auc_lr.mean(), auc_rf.mean(),

                           auc_xgb.mean(), auc_cb.mean(), auc_vc.mean()],

              'accuracy': [acc_lr.mean(), acc_rf.mean(),

                            acc_xgb.mean(), acc_cb.mean(), acc_vc.mean()]}

df_score = pd.DataFrame(dict_values, index = models, columns = ['auc_roc', 'accuracy'])

df_score