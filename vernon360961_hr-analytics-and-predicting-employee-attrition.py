#Importing all the necessary libraries



import pandas as pd

import numpy as np

import seaborn as sns

import datetime as dt

import matplotlib.pyplot as plt

import xgboost as xgb



%matplotlib inline
#Importing the libraries to measure metrics on your model



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
#Ignoring Python warnings



import warnings

warnings.filterwarnings('ignore')
#Loading our IBM HR dataset



df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

#df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Viewing the first 5 rows of our dataset



df.head()
#Searching for null values in our dataset



pd.DataFrame(df.isna().sum(), columns=['Null Values Present']).T
#We can create new features to help improve our analysis and aid our machine learning process later on



#Calculating overall satisfaction using pre-existing columns in our DB

df['OverallSatisfaction'] = (df['RelationshipSatisfaction']  

                                  + df['EnvironmentSatisfaction'] 

                                  + df['JobSatisfaction'] 

                                  + df['JobInvolvement'] 

                                  + df['WorkLifeBalance'])/5



#Calculating mean job satisfaction using pre-existing columns in our DB

df['JobSatisf_mean'] = (df['JobSatisfaction'] + df['JobInvolvement']) / 2



#Calculating ratio of monthly income to distance

df['IncomeByDistance'] = df['MonthlyIncome'] / df['DistanceFromHome']



#Calculating employee progress in current company

df['CompanySatisfaction'] = df['YearsInCurrentRole'] / df['YearsAtCompany']





def dis_emp_satdis(df) :

    if df['OverallSatisfaction'] < 2.5 and df['DistanceFromHome'] > 10:

        return 1

    else:

        return 0

df['DisEmpSatDis'] = df.apply(lambda df:dis_emp_satdis(df) ,axis = 1)





def dis_emp(df) :

    if df['OverallSatisfaction'] < 2:

        return 1

    else:

        return 0

df['DisEmp'] = df.apply(lambda df:dis_emp(df) ,axis = 1)





def mar_age(df) :

    if df['MaritalStatus'] == 'Single' and df['Age'] > 35:

        return 1

    else:

        return 0

df['MarAge'] = df.apply(lambda df:mar_age(df) ,axis = 1)





def distance_wlbal(df) : 

    if  df['DistanceFromHome'] > 10 and df['WorkLifeBalance'] < 2 :

        return 1

    else : 

        return 0

df['LongDis_BadWLBal'] = df.apply(lambda df:distance_wlbal(df) ,axis = 1)





def distance(df) : 

    if  df['DistanceFromHome'] > 11:

        return 1

    else : 

        return 0

df['LongDistance'] = df.apply(lambda df:distance(df) ,axis = 1)



def job_inv_sat(df) : 

    if  df['JobInvolvement'] < 2 and df['JobSatisfaction'] < 2:

        return 1

    else : 

        return 0

df['RoleFit'] = df.apply(lambda df:job_inv_sat(df) ,axis = 1)





def job_lvl_ovr(df) : 

    if  df['JobLevel'] == 1 and df['OverTime'] == 1:

        return 1

    else : 

        return 0

df['JobLevelOvertime'] = df.apply(lambda df:job_lvl_ovr(df) ,axis = 1)



def job_lvl_mar(df) : 

    if  df['JobLevel'] == 1 and df['MaritalStatus'] == 'Single':

        return 1

    else : 

        return 0

df['JobLevelMarried'] = df.apply(lambda df:job_lvl_mar(df) ,axis = 1)
df_int = df.loc[:, df.dtypes == np.int64]
df_int = df_int.drop(['EmployeeCount','StandardHours'], axis = 1)
df_string = df.loc[:, df.dtypes == object]
plt.figure(figsize= (25,15))

sns.heatmap(df_int.corr().round(2), cmap="BuPu_r")
df_attrition = pd.DataFrame(df.Attrition.value_counts())
df_attrition['kind'] = df_attrition.index
df_attrition
plt.figure(figsize=(16,8))

sns.barplot(x="kind", y="Attrition", data=df_attrition)

plt.title('Employee Attrition Distribution')

plt.xlabel('Attrition')

plt.ylabel('Number of employees')

plt.grid()
# Sort the dataframe by Age

df_no = df.loc[df['Attrition'] == 'No']

df_yes = df.loc[df['Attrition'] == 'Yes']



plt.figure(figsize=(16,8))

sns.distplot(df_yes[['Age']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['Age']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.legend()

plt.title('Attrition by Age (Blue - No/ Green-Yes)')

plt.xlabel('Age')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['Education']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['Education']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by Education')

plt.legend()

plt.xlabel('Education')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['EnvironmentSatisfaction']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['EnvironmentSatisfaction']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by EnvironmentSatisfaction')

plt.legend()

plt.xlabel('EnvironmentSatisfaction')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['MonthlyIncome']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['MonthlyIncome']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by MonthlyIncome')

plt.legend()

plt.xlabel('MonthlyIncome')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['YearsAtCompany']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['YearsAtCompany']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by YearsAtCompany')

plt.legend()

plt.xlabel('YearsAtCompany')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['WorkLifeBalance']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['WorkLifeBalance']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by WorkLifeBalance')

plt.legend()

plt.xlabel('WorkLifeBalance')

plt.ylabel('Attrition')
plt.figure(figsize=(16,8))

sns.distplot(df_yes[['NumCompaniesWorked']],color='dodgerblue', label="Yes", hist_kws={'alpha':.5})

sns.distplot(df_no[['NumCompaniesWorked']], color='orange', label="No", hist_kws={'alpha':.5})

plt.grid()

plt.title('Attrition by NumCompaniesWorked')

plt.legend()

plt.xlabel('NumCompaniesWorked')

plt.ylabel('Attrition')
#Average values for employees that have not left the company and employees that have left

df.groupby(['Attrition']).mean().T
#Dropping the columns that don't provide much information



df = df.drop(['EmployeeCount','StandardHours','Over18'], axis = 1)
#Dropping the columns that don't provide much information



df_string.drop(['Over18'], axis = 1, inplace= True)
#Creating a new subset dataframe so that we can encode all the categorical variables together



dummy_cat = df_string.drop(['Attrition'], axis = 1)
dummy_cat = pd.get_dummies(dummy_cat)

dummy_cat.head()
#To avoid multicollinearity, we would need to remove one variable each for all categorical columns



dummy_cat.drop(['BusinessTravel_Non-Travel',

               'Department_Human Resources',

               'EducationField_Human Resources',

               'Gender_Female',

               'JobRole_Healthcare Representative',

               'MaritalStatus_Divorced',

               'OverTime_No',],axis = 1, inplace=True)
#We've separated the columns with binary values from the continuous values, we do this so that we can scale the continuous values separately without affecting



df_binary = df.loc[:,['MarAge','LongDis_BadWLBal','LongDistance','RoleFit','JobLevelOvertime','JobLevelMarried','DisEmpSatDis','DisEmp']]
#REmoving the binary columns from the data frame that holds the numerical columns



df_int.drop(['MarAge',

               'LongDis_BadWLBal',

               'LongDistance',

               'RoleFit',

               'JobLevelOvertime',

               'JobLevelMarried',

               'DisEmpSatDis',

               'DisEmp'],axis = 1, inplace=True)
#Combining the binary columns and categorical columns



df_final = pd.concat([df_binary, dummy_cat], axis=1)
#Converting the Employee Attrition column to numrical values



df.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)
df_int.drop(['EmployeeNumber'], axis =1, inplace=True)
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

scaled_df_int = std.fit_transform(df_int)

scaled_df_int = pd.DataFrame(scaled_df_int, columns=df_int.columns)
#Once we've scaled the numerical columns, we can now combine our final dataframe and the scaled dataframes



df_final = pd.concat([scaled_df_int, df_final], axis=1)
#Creating out target variable



y = df[['Attrition']]
#Our final set of features



X = df_final
# Threshold for removing correlated variables

threshold = 0.8



# Absolute value correlation matrix

corr_matrix = X.corr().abs()

corr_matrix.head()



# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()



# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#Dropping columns with strong correlation



X = X.drop(columns = to_drop)
#Dividing out DF into train and test sets



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 0)
classifier1 = xgb.XGBClassifier(n_jobs = -1)



# A parameter grid for XGBoost

params = {

        'n_estimators' : [100, 400, 500, 700, 800],

        'learning_rate' : [0.01, 0.01, 0.05, 0.1, 0.25, 0.5],

        'min_child_weight': [1, 2, 4, 5, 7, 9, 13],

        'gamma': [0.1, 0.5, 1, 1.5, 5],

        'subsample': [0.5, 0.7, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [2, 4, 6, 8, 10, 12]

        }



folds = 5

random_search = RandomizedSearchCV(classifier1, param_distributions=params, n_iter=800, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, random_state=42)

# and now we train over all the possible combinations of the parameters above

#random_search.fit(X_train, y_train)



# and we print the best score over the train set

#print(("best roc-auc from grid search: %.3f"

#       % random_search.score(X_train, y_train)))

#best roc-auc from grid search: 0.941
# we can print the best estimator parameters like this

#random_search.best_estimator_

#XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

#              colsample_bynode=1, colsample_bytree=0.6, gamma=0.1, gpu_id=-1,

#              importance_type='gain', interaction_constraints=None,

#              learning_rate=0.05, max_delta_step=0, max_depth=2,

#              min_child_weight=2, missing=nan, monotone_constraints=None,

#              n_estimators=500, n_jobs=-1, num_parallel_tree=1,

#              objective='binary:logistic', random_state=0, reg_alpha=0,

#              reg_lambda=1, scale_pos_weight=1, subsample=0.5, tree_method=None,

#              validate_parameters=False, verbosity=None)
# and find the best fit parameters like this

#random_search.best_params_
#Fitting our Train data to our XGB classifier with the best possible parameters



xgb_classifier = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.6, gamma=0.1,

              importance_type='gain',

              learning_rate=0.05, max_delta_step=0, max_depth=2,

              min_child_weight=2,

              n_estimators=500, n_jobs=-1, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=0.5,

              validate_parameters=False)



xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

y_prob = xgb_classifier.predict_proba(X_test)[:,1]
# let's explore the importance of the features



importance = pd.Series(xgb_classifier.feature_importances_)

importance.index = X.columns

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(25,9))
from sklearn.metrics import classification_report,confusion_matrix
print('---'*20)

print('Cross Validation Scores For the XGBoost Classifier')

print('---'*20)

print('Accuracy : ' + str(cross_val_score(xgb_classifier, X, y, cv = 5, scoring = 'accuracy').mean()))

print('Precision : ' + str(cross_val_score(xgb_classifier, X, y, cv = 5, scoring = 'precision').mean()))

print('Recall : ' + str(cross_val_score(xgb_classifier, X, y, cv = 5, scoring = 'recall').mean()))

print('---'*20)

print('Confusion Matrix For the XGBoost Classifier')

print('---'*20)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print('---'*20)

print('Classification Report For the XGBoost Classifier')

print('---'*20)

print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]





# A parameter grid for XGBoost

params = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



folds = 5

random_search_rfc = RandomizedSearchCV(rfc, param_distributions=params, n_iter=800, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, random_state=42)

# and now we train over all the possible combinations of the parameters above

#random_search_rfc.fit(X_train, y_train)
# and we print the best score over the train set

#random_search_rfc.score(X_train, y_train)

#0.9203109815354713
# we can print the best estimator parameters like this

#random_search_rfc.best_estimator_

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#                       max_depth=90, max_features='sqrt', max_leaf_nodes=None,

#                       min_impurity_decrease=0.0, min_impurity_split=None,

#                       min_samples_leaf=2, min_samples_split=10,

#                       min_weight_fraction_leaf=0.0, n_estimators=100,

#                       n_jobs=None, oob_score=False, random_state=None,

#                       verbose=0, warm_start=False)
# and find the best fit parameters like this

#random_search_rfc.best_params_

#{'n_estimators': 100,

# 'min_samples_split': 10,

# 'min_samples_leaf': 2,

# 'max_features': 'sqrt',

# 'max_depth': 90,

# 'bootstrap': True}
rfc_classifier = RandomForestClassifier(bootstrap=True, criterion='gini',

                       max_depth=90, max_features='sqrt',

                       min_impurity_decrease=0.0,

                       min_samples_leaf=2, min_samples_split=10,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       oob_score=False,

                       verbose=0, warm_start=False)



rfc_classifier.fit(X_train, y_train)

y_pred_rfc = rfc_classifier.predict(X_test)
# let's explore the importance of the features



importance = pd.Series(rfc_classifier.feature_importances_)

importance.index = X.columns

importance.sort_values(inplace=True, ascending=False)

importance.plot.bar(figsize=(25,9))
print('---'*20)

print('Cross Validation Scores For the Random Forest Classifier')

print('---'*20)

print('Accuracy : ' + str(cross_val_score(rfc_classifier, X, y, cv = 5, scoring = 'accuracy').mean()))

print('Precision : ' + str(cross_val_score(rfc_classifier, X, y, cv = 5, scoring = 'precision').mean()))

print('Recall : ' + str(cross_val_score(rfc_classifier, X, y, cv = 5, scoring = 'recall').mean()))

print('---'*20)

print('Confusion Matrix For the Random Forest Classifier')

print('---'*20)

cm_rfc = confusion_matrix(y_test, y_pred_rfc)

print(cm_rfc)

print('---'*20)

print('Classification Report For the Random Forest Classifier')

print('---'*20)

print(classification_report(y_test,y_pred_rfc))