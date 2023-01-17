# importing libraries for data handling and analysis 

import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv',index_col='Id')

df.head()
# importing libraries for data visualisations

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# To get rid of duplicates present in the data

print(df.duplicated().sum())

df.drop_duplicates(inplace=True)
x = df.drop(['Attrition'],axis=1)

y = df.Attrition

x.shape
# to check whether mssing values present or not



sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.hist(figsize=(20,20))

plt.show()
types = df.columns.to_series().groupby(df.dtypes).groups

types
cols = df.columns

cols_ob = []

for j in cols :

    if df[j].dtypes=='object':

        cols_ob.append(j)

df.shape
plt.boxplot(x.MonthlyIncome)
plt.boxplot(x.DistanceFromHome)
import seaborn as sns

sns.set_style('whitegrid')

sns.countplot(x='Attrition',hue='BusinessTravel',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='Department',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='EducationField',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='Gender',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='JobRole',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='MaritalStatus',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='OverTime',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='Education',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='PerformanceRating',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='Education',data=df,palette='RdBu_r')
sns.countplot(x='Attrition',hue='StockOptionLevel',data=df,palette='RdBu_r')
x = df.drop(['Attrition'],axis=1)

y = df.Attrition

#y = y[x.MonthlyIncome<15000]  # if want to remove outliers from data

#x = x[x.MonthlyIncome<15000]
# checking skewness of numerical features 

df.skew()
skew_feas = ['MonthlyIncome','DistanceFromHome','TotalWorkingYears','YearsAtCompany','YearsSinceLastPromotion','YearsWithCurrManager','YearsInCurrentRole',

             'PerformanceRating']

x[skew_feas].skew()
# using sqrt() to reduce skewness in the data

x[skew_feas] = np.sqrt(x[skew_feas])

x[skew_feas].skew()
# sklearn modules for preprocessing

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

LE = LabelEncoder()

# Label Encoding will be used for columns with 2 or less unique values

le_count = 0

for col in x.columns[1:]:

    if x[col].dtype == 'object':

        if len(list(x[col].unique())) <= 2:

            LE.fit(x[col])

            x[col] = LE.transform(x[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))
# One hot encoding for categorical columns with more than 2 unique values

dummies = pd.get_dummies(x[list(set(cols_ob)-set(['Gender','OverTime']))])

dummies.head()
x_merged = pd.concat([x,dummies],axis=1)

x_merged.drop(['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'],axis=1,inplace=True)

x_merged.shape
c =['Behaviour','EmployeeNumber']#,'Gender','Education','JobRole_Sales Executive','JobRole_Research Scientist']#'PercentSalaryHike','PerformanceRatings']  

c = list(set(x_merged.columns)-set(c))

print(len(c))
x_merged_dum = x_merged

x_merged = (x_merged-x_merged.min())/(x_merged.max()-x_merged.min())

x_merged = x_merged*3

print(x_merged.shape)
#x_merged.describe().transpose()
x_test = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv',index_col='Id')

x_test.shape
x_test[skew_feas] = np.sqrt(x_test[skew_feas])

x_test[skew_feas].skew()
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

# Label Encoding will be used for columns with 2 or less unique values

le_count = 0

for col in x_test.columns[1:]:

    if x_test[col].dtype == 'object':

        if len(list(x_test[col].unique())) <= 2:

            LE.fit(x_test[col])

            x_test[col] = LE.transform(x_test[col])

            le_count += 1

print('{} columns were label encoded.'.format(le_count))
dummies_test = pd.get_dummies(x_test[list(set(cols_ob)-set(['Gender','OverTime']))])

dummies_test.head()
x_test_merged = pd.concat([x_test,dummies_test],axis=1)

x_test_merged.drop(['BusinessTravel','Department','EducationField','JobRole','MaritalStatus'],axis=1,inplace=True)

x_test_merged.shape
x_merged_col = list(x_test_merged.columns)

for col in x_merged_col:

    x_test_merged[col] = x_test_merged[col].astype(float)

    

x_test_merged.shape
x_test_merged = (x_test_merged-x_merged_dum.min())/(x_merged_dum.max()-x_merged_dum.min())

x_test_merged = x_test_merged*3

print(x_test_merged.shape)
#x_test_merged.describe().transpose()
# sklearn modules for ML model selection

from sklearn.model_selection import train_test_split  # import 'train_test_split'

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



# Libraries for data modelling

from sklearn import svm, tree

from sklearn import ensemble

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



# Common sklearn Model Helpers

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# sklearn modules for performance metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc, roc_auc_score
# selection of algorithms to consider and set performance measure

models = []

models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=5)))

models.append(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=5)))

models.append(('SVM', SVC(gamma='auto', random_state=7)))

models.append(('KNN', KNeighborsClassifier()))

models.append(('Decision Tree Classifier',DecisionTreeClassifier(random_state=7)))

models.append(('Gaussian NB', GaussianNB()))
from sklearn.model_selection import train_test_split

x_train,x_tes,y_train,y_tes = train_test_split(x_merged,y,test_size=0.1)

print(x_train.shape,x_tes.shape)
acc_results = []

auc_results = []

names = []

# set table to table to populate with performance results

col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 

       'Accuracy Mean', 'Accuracy STD']

df_results = pd.DataFrame(columns=col)

i = 0

# evaluate each model using cross-validation

for name, model in models:

    kfold = model_selection.KFold(

        n_splits=10, random_state=5)  # 10-fold cross-validation



    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring

        model, x_train[c], y_train, cv=kfold, scoring='accuracy')



    cv_auc_results = model_selection.cross_val_score(  # roc_auc scoring

        model, x_train[c], y_train, cv=kfold, scoring='roc_auc')



    acc_results.append(cv_acc_results)

    auc_results.append(cv_auc_results)

    names.append(name)

    df_results.loc[i] = [name,

                         round(cv_auc_results.mean()*100, 2),

                         round(cv_auc_results.std()*100, 2),

                         round(cv_acc_results.mean()*100, 2),

                         round(cv_acc_results.std()*100, 2)

                         ]

    i += 1

df_results.sort_values(by=['ROC AUC Mean'], ascending=False)
fig = plt.figure(figsize=(15, 7))

fig.suptitle('Algorithm Accuracy Comparison')

ax = fig.add_subplot(111)

plt.boxplot(acc_results)

ax.set_xticklabels(names)

plt.show()
fig = plt.figure(figsize=(15, 7))

fig.suptitle('Algorithm ROC AUC Comparison')

ax = fig.add_subplot(111)

plt.boxplot(auc_results)

ax.set_xticklabels(names)

plt.show()
from sklearn.svm import SVC

kfold = model_selection.KFold(n_splits=10, random_state=7)

modelCV = SVC(random_state=7, probability=True)

scoring = 'roc_auc'

results = model_selection.cross_val_score(

    modelCV, x_train[c], y_train, cv=kfold, scoring=scoring)

print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))
param_grid = {'C': [0.1, 0.5, 1, 2, 5, 10, 15, 20],

              'gamma' : ['auto','scale']} # hyper-parameter list to fine-tune

random = RandomizedSearchCV(estimator=modelCV,

                            param_distributions=param_grid,

                            scoring='roc_auc',

                            verbose=1, n_jobs=-1,

                            n_iter=10000,cv=10)



random_result = random.fit(x_merged[c], y)



print('Best Score: ', random_result.best_score_)

print('Best Params: ', random_result.best_params_)

"""print('='*20)

print("best params: " + str(log_gs.best_estimator_))

print("best params: " + str(log_gs.best_params_))

print('best score:', log_gs.best_score_)

print('='*20)"""
x_train,x_tes,y_train,y_tes = train_test_split(x_merged,y,test_size=0.1)

print(x_train.shape,x_tes.shape)
from sklearn.svm import SVC

SVC = SVC(C=0.5,gamma='scale',kernel='rbf',probability=True)

SVC.fit(x_merged[c],y)
from sklearn import metrics

y_pred_train = SVC.predict(x_merged[c])

print(metrics.confusion_matrix(y,y_pred_train))

print(metrics.roc_auc_score(y,SVC.predict_proba(x_merged[c])[:,1]))
#print(metrics.roc_auc_score(y_tes,SVC.predict_proba(x_tes[c])[:,1]))

#print(metrics.confusion_matrix(y_tes,SVC.predict(x_tes[c])))
y_pred_test_SVC = SVC.predict_proba(x_test_merged[c])
output_svc = pd.concat([pd.Series(x_test.index),pd.Series(y_pred_test_SVC[:,1])],axis=1)

output_svc.columns = ['Id','Attrition']

output_svc.head(15)
output_svc.to_csv('pred_attrition_svc',index=False)
from sklearn.linear_model import LogisticRegression

kfold = model_selection.KFold(n_splits=10, random_state=7)

modelCV = LogisticRegression(solver='liblinear',

                             class_weight="balanced", 

                             random_state=7)

scoring = 'roc_auc'

results = model_selection.cross_val_score(

    modelCV, x_train[c], y_train, cv=kfold, scoring=scoring)

print("AUC score (STD): %.2f (%.2f)" % (results.mean(), results.std()))
param_grid = {'C': np.arange(1e-03, 2, 0.01),

              'solver': ['lbfgs', 'liblinear']} # hyper-parameter list to fine-tune

random = GridSearchCV(estimator=modelCV,

                            param_grid=param_grid,

                            scoring='roc_auc',

                            verbose=1, n_jobs=-1,

                            cv=10)



random_result = random.fit(x_merged[c], y)



print('Best Score: ', random_result.best_score_)

print('Best Params: ', random_result.best_params_)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C = 0.05099999999999999 ,solver='lbfgs',class_weight=None,max_iter=1000)

LR.fit(x_train[c],y_train)
from sklearn import metrics

y_pred_train = LR.predict_proba(x_merged[c])

print(metrics.roc_auc_score(y,y_pred_train[:,1]))

print(metrics.confusion_matrix(y,LR.predict(x_merged[c])))
#y_pred_tes = LR.predict_proba(x_tes[c])

#print(metrics.roc_auc_score(y_tes,y_pred_tes[:,1]))

#print(metrics.confusion_matrix(y_tes,LR.predict(x_tes[c])))
y_pred_test_LR = LR.predict_proba(x_test_merged[c])
output_LR = pd.concat([pd.Series(x_test_merged.index),pd.Series(y_pred_test_LR[:,1])],axis=1)

output_LR.columns = ['Id','Attrition']

output_LR.head(15)
output_LR.to_csv('pred_LR',index=False)