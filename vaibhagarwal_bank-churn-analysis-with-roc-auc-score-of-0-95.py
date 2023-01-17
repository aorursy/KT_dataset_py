import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns



%matplotlib inline

from matplotlib import rcParams

sns.set_style("whitegrid")

sns.set_context("poster")
rcParams['figure.figsize'] = (8.0, 5.0)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
file_1 = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')
df_orig = pd.DataFrame(file_1)
df_orig.head()
df = df_orig.copy()
# Dropping the id and name columns.

df.drop('CustomerId', axis=1, inplace=True)

df.drop('Surname', axis=1, inplace=True)

df.head()
df.info()
# Converting NumOfProducts column to categorical.

df['NumOfProducts'] = df['NumOfProducts'].astype(int)

df['NumOfProducts'] = df['NumOfProducts'].astype(object)
# Creating seperate columns for categories

df = pd.get_dummies(df)

df.head()
# Dropping excess columns

df.drop('Geography_Spain', axis=1, inplace=True)

df.drop('Gender_Male', axis=1, inplace=True)

df.drop('NumOfProducts_2', axis=1, inplace=True)

df.head()
df.columns
df = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'HasCrCard',

       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',

       'Geography_France', 'Gender_Female', 'NumOfProducts_1',

       'NumOfProducts_4', 'NumOfProducts_3', 'Exited']] 
# Correlation Matrix

corr = df.corr()

corr.style.background_gradient()
# Converting all Balances more than 0 to 1

df['Balance'] = df['Balance'].clip(upper=1)
# Dropping insignificant features as decided during previous excercises.

# Age p-value = 0.0

# Credit Score p-value = 0.0085

# Balance p-value = 0.0

# Estimated Salary p-value = 0.1222

df.drop('EstimatedSalary', axis=1, inplace=True)

df.drop('HasCrCard', axis=1, inplace=True)

#df.drop('NumOfProducts', axis=1, inplace=True)

df.drop('Tenure', axis=1, inplace=True)

df.head()
df = df.applymap(np.int64)
df.loc[df.Balance == 0, 'Balance'] = -1

df.loc[df.IsActiveMember == 0, 'IsActiveMember'] = -1

df.loc[df.Geography_Germany == 0, 'Geography_Germany'] = -1

df.loc[df.Geography_France == 0, 'Geography_France'] = -1

df.loc[df.Gender_Female == 0, 'Gender_Female'] = -1

df.loc[df.NumOfProducts_1 == 0, 'NumOfProducts_1'] = -1

df.loc[df.NumOfProducts_3 == 0, 'NumOfProducts_3'] = -1

df.loc[df.NumOfProducts_4 == 0, 'NumOfProducts_4'] = -1

df.loc[df.Exited == 0, 'Exited'] = -1

df.head()
# Scaling the data

from sklearn.preprocessing import scale



df['CreditScore'] = scale(df['CreditScore'])

df['Age'] = scale(df['Age'])

#df['Tenure'] = scale(df['Tenure'])

#df['NumOfProducts'] = scale(df['NumOfProducts'])



df.head()
df.columns
X = df[['CreditScore', 'Age', 'Balance', 'IsActiveMember', 'Geography_Germany',

       'Geography_France', 'Gender_Female', 'NumOfProducts_1',

       'NumOfProducts_4', 'NumOfProducts_3']]

y = df['Exited']
# Using Lasso to know features significance.

from sklearn.linear_model import Lasso

names = df.columns

lasso = Lasso(alpha=0.1)

lasso_coef = lasso.fit(X, y).coef_

_ = plt.plot(range(len(names)-1), lasso_coef)

plt.xticks(range(len(names)-1), names, rotation=60)
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)
a = pd.Series(y_sm)

a.value_counts()
# Splitting the data in test data and train data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=51)
from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
# Fitting the data



logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

accuracy_score(y_pred, y_test)
# Using GridSearch to find the best parameters



grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

logreg_cv=GridSearchCV(logreg,grid,cv=10, scoring='accuracy', refit=True, n_jobs=-1)

logreg_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)

print("accuracy :",logreg_cv.best_score_)

print(logreg_cv.best_estimator_)
# ROC Curve



y_pred_prob = logreg_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='Logistic Regression')

plt.legend()
# ROC AUC score. The area under ROC curve.



roc_auc_score(y_test, y_pred_prob)


print(confusion_matrix(y_test, logreg_cv.predict(X_test)))



#Tp#Fp

#Fn#Tn
# Classification Report



print(classification_report(y_test, logreg_cv.predict(X_test)))
knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train,y_train)

accuracy_score(knn.predict(X_test), y_test)
knn = KNeighborsClassifier()

k_grid={'n_neighbors':np.arange(2,20)}

knn_cv=GridSearchCV(knn, k_grid, cv=10, refit=True, n_jobs=-1)

knn_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)

print("accuracy :",knn_cv.best_score_)

print(knn_cv.best_estimator_)
y_pred_prob = knn_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='k Nearest Neighbors')

plt.legend()
roc_auc_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, knn_cv.predict(X_test)))



#Tp#Fp

#Fn#Tn
print(classification_report(y_test, knn_cv.predict(X_test)))


Cs = [0.1, 1, 10, 100]

gammas = [0.001, .01, 0.1, 1, 10]

param_grid = {'C': Cs, 'gamma': gammas,'kernel': ['rbf'], 'probability':[True]}



SVM_rbf_cv = GridSearchCV(SVC(), param_grid, cv=3, refit=True, n_jobs=-1)

SVM_rbf_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",SVM_rbf_cv.best_params_)

print("accuracy :",SVM_rbf_cv.best_score_)

print(SVM_rbf_cv.best_estimator_)
y_pred_prob = SVM_rbf_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='SVM')

plt.legend()
roc_auc_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, SVM_rbf_cv.predict(X_test)))



#Tp#Fp

#Fn#Tn
print(classification_report(y_test, SVM_rbf_cv.predict(X_test)))


Cs = [0.1, 1, 10, 100]

gammas = [0.001, .01, 0.1, 0.5]



param_grid = {'C': Cs, 'gamma': gammas,'probability':[True],'kernel': ['poly'],'degree':[2,3] }

SVM_poly_cv = RandomizedSearchCV(estimator = SVC(), param_distributions = param_grid, n_iter = 10, cv = 3, random_state=51, n_jobs = -1, refit=True)

SVM_poly_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",SVM_poly_cv.best_params_)

print("accuracy :",SVM_poly_cv.best_score_)

print(SVM_poly_cv.best_estimator_)
y_pred_prob = SVM_poly_cv.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='SVM')

plt.legend()
roc_auc_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, SVM_poly_cv.predict(X_test)))



#Tp#Fp

#Fn#Tn
print(classification_report(y_test, SVM_poly_cv.predict(X_test)))


n_est = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 10)]

m_depth = [int(x) for x in np.linspace(5, 50, num = 5)]

min_samp = [3, 5, 6, 7, 10, 11]

m_ftr = ['auto']



param_grid = {'max_depth': m_depth, 'max_features': m_ftr,'n_estimators': n_est,'min_samples_split': min_samp}

RF_cv = RandomizedSearchCV(estimator = RandomForestClassifier(), n_iter=100, param_distributions =  param_grid, random_state=51, cv=3, n_jobs=-1, refit=True)

RF_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",RF_cv.best_params_)

print("accuracy :",RF_cv.best_score_)

print(RF_cv.best_estimator_)
n_est = [366]

m_depth = [38]

min_samp = [5]

m_ftr = ['auto']



param_grid = {'max_depth': m_depth, 'max_features': m_ftr,'n_estimators': n_est,'min_samples_split': min_samp}

RF_cv_10 = RandomizedSearchCV(estimator = RandomForestClassifier(), n_iter=200, param_distributions =  param_grid, random_state=51, cv=10, n_jobs=-1, refit=True)

RF_cv_10.fit(X_train,y_train)
y_pred_prob = RF_cv_10.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='Random Forest')

plt.legend()
roc_auc_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, RF_cv.predict(X_test)))



#Tp#Fp

#Fn#Tn
print(classification_report(y_test, RF_cv.predict(X_test)))


m_dep = [5,6,7,8]

gammas = [0.01,0.001,0.001]

min_c_wt = [1,5,10]

l_rate = [0.05,0.1, 0.2, 0.3]

n_est = [5,10,20,100]



param_grid = {'n_estimators': n_est, 'gamma': gammas, 'max_depth': m_dep,

              'min_child_weight': min_c_wt, 'learning_rate': l_rate}



xgb_cv = RandomizedSearchCV(estimator = XGBClassifier(), n_iter=100, param_distributions =  param_grid, random_state=51, cv=3, n_jobs=-1, refit=True)

xgb_cv.fit(X_train,y_train)



print("tuned hpyerparameters :(best parameters) ",xgb_cv.best_params_)

print("accuracy :",xgb_cv.best_score_)

print(xgb_cv.best_estimator_)
m_dep = [7]

gammas = [0.01]

min_c_wt = [1]

l_rate = [0.2]

n_est = [100]



param_grid = {'n_estimators': n_est, 'gamma': gammas, 'max_depth': m_dep,

              'min_child_weight': min_c_wt, 'learning_rate': l_rate}



xgb_cv_10 = RandomizedSearchCV(estimator = XGBClassifier(), n_iter=100, param_distributions =  param_grid, random_state=51, cv=10, n_jobs=-1, refit=True)

xgb_cv_10.fit(X_train,y_train)
y_pred_prob = xgb_cv_10.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label='xgb')

plt.legend()
roc_auc_score(y_test, y_pred_prob)
print(confusion_matrix(y_test, xgb_cv_10.predict(X_test)))



#Tp#Fp

#Fn#Tn
print(classification_report(y_test, xgb_cv_10.predict(X_test)))
algos = [logreg_cv, knn_cv, SVM_rbf_cv, SVM_poly_cv, RF_cv, xgb_cv]

labels = ['Logistic Regression', 'knn', 'SVM rbf', 'SVM poly', 'Random Forest', 'XGB']



plt.figure(figsize = (12,8))

plt.plot([0,1], [0,1], 'k--')



for i in range(len(algos)):

    y_pred_prob = algos[i].predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot(fpr, tpr, label=labels[i])



plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC Curve')

plt.legend(loc='best')