# Basic Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import statements required for Plotly 
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from sklearn.tree import ExtraTreeClassifier
#preprocess.
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             roc_auc_score,
                             roc_curve,
                             confusion_matrix)
from sklearn.model_selection import (cross_val_score,
                                     GridSearchCV,
                                     RandomizedSearchCV,
                                     learning_curve,
                                     validation_curve,
                                     train_test_split)

from sklearn.pipeline import make_pipeline # For performing a series of operations

from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.info()
df.isnull().sum()
df.dtypes.unique() # There are the only available datatypes in our dataset
df.describe()
df.Attrition.describe()
df.Attrition.value_counts()
df.BusinessTravel.value_counts()
sns.distplot(df.Age) # Age is unimodal
df.Age.describe() # Age is Normally Distributed
df.columns
df.skew()
num_cat = df.select_dtypes(exclude='O')
num_cat_cols = num_cat.columns
num_cat_cols
fig,ax = plt.subplots(6,2,figsize=(9,9))
sns.distplot(df['TotalWorkingYears'],ax=ax[0,0])
sns.distplot(df['MonthlyIncome'],ax=ax[0,1])
sns.distplot(df['YearsAtCompany'], ax = ax[1,0]) 
sns.distplot(df['DistanceFromHome'], ax = ax[1,1]) 
sns.distplot(df['YearsInCurrentRole'], ax = ax[2,0]) 
sns.distplot(df['YearsWithCurrManager'], ax = ax[2,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[3,0]) 
sns.distplot(df['PercentSalaryHike'], ax = ax[3,1]) 
sns.distplot(df['YearsSinceLastPromotion'], ax = ax[4,0]) 
sns.distplot(df['TrainingTimesLastYear'], ax = ax[4,1]) 
sns.distplot(df['DailyRate'], ax = ax[5,0]) 
sns.distplot(df['HourlyRate'], ax = ax[5,1]) 
plt.tight_layout()
cat_df = df.select_dtypes(include='O')
cat_df.head()
cat_df.columns
# function to plot all categorical variables
def plot_cat(attr):
    #sns.factorplot(data=df,kind='count',size=5,aspect=1.5,x=attr)
    data = [go.Bar(
            x=df[attr].value_counts().index.values,
            y= df[attr].value_counts().values
    )]
    py.iplot(data, filename='basic-bar')


    
plot_cat('Attrition')
plot_cat(df.BusinessTravel.name)
plot_cat(df.EducationField.name)
plot_cat(df.Department.name)
plot_cat(df.Gender.name)
plot_cat(df.MaritalStatus.name)
plot_cat(df.JobRole.name)
plot_cat(df.Over18.name)
df.Over18.describe()
df.Over18.value_counts()
plot_cat(df.OverTime.name)
# def plot_num(attr):
#     sns.factorplot(data=df,kind='count',size=5,aspect=1.5,x=attr)
    
# for i in num_cat_cols:
#     plot_num(i)

cor_mat = df.corr()
np.amin(cor_mat) # No serious -ve correlation can be seen

mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,15)
sns.heatmap(data=cor_mat,mask=mask,fmt='.2f',linewidths=0.1,square=True,annot=True,cbar=True)
# Creating a function to plot against target variable
def plot_target(attr):
    if attr == df.Age.name:
        sns.factorplot(data=df,y='Age',x='Attrition',size=5,aspect=1,kind='box')
        return
    sns.factorplot(data=df,kind='count',x=df.Attrition.name,col = attr)
plot_target(df.Department.name)
#pd.crosstab(columns=df.Attrition,index=df.Department,values=df.Attrition,aggfunc='mean')
pd.crosstab(columns=df.Attrition,index=df.Department,normalize='index') # normailze = index gives row wise mean
plot_target('Age')

sns.factorplot(data=df,kind='bar',x='Attrition',y='MonthlyIncome')
plot_target(df.JobSatisfaction.name)
age_cross_tab = pd.crosstab(columns=df.Attrition,index=df.Age,margins_name='Total',margins=True)
age_cross_tab['Attrition_Ratio'] = age_cross_tab.Yes/age_cross_tab.Total
age_cross_tab
pd.crosstab(columns=[df.Attrition],index=[df.Gender],margins=True,normalize='index')
pd.crosstab(columns=df.Attrition,index=df.JobLevel,margins=True,normalize='index')
pd.crosstab(columns=df.Attrition,index=[df.JobLevel,df.JobSatisfaction],margins=True,normalize='index')
pd.crosstab(columns=[df.Attrition],index=[df.EnvironmentSatisfaction],margins=True,normalize='index')
pd.crosstab(columns=[df.Attrition],index=df.YearsWithCurrManager,margins=True,normalize='index')
pd.crosstab(columns=[df.Attrition],index=df.YearsSinceLastPromotion,margins=True,normalize='index')
pd.crosstab(columns=[df.Attrition],index=[df.WorkLifeBalance],margins=True,normalize='index')
pd.crosstab(columns=[df.Attrition],index=[df.BusinessTravel],margins=True,normalize='index')
# Using RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df.shape
df.head()
# Ecoding to low medium high based on ranges 
def encode_salary(salary):
    if salary>=1009 and salary < 7339:
        return 'Low'
    elif salary>=7339 and salary < 13669:
        return 'Medium'
    elif salary >=13669 and salary <= 19999:
        return 'High'
    
df['Income_Cat'] = df['MonthlyIncome'].apply(encode_salary)
df.Income_Cat.value_counts()
df.Income_Cat.shape
df.isnull().sum(axis=0)
df.Income_Cat.value_counts()
dic = {'Low':0,'Medium':1, 'High':2}
df.Income_Cat = df.Income_Cat.map(dic)
df.Income_Cat.head()
df.Income_Cat.value_counts()
df.shape
df.drop(['BusinessTravel','DailyRate','EmployeeCount','EmployeeNumber','HourlyRate','MonthlyRate'
          ,'NumCompaniesWorked','Over18','StandardHours', 'StockOptionLevel','TrainingTimesLastYear','MonthlyIncome'],axis=1,inplace=True)
df.head()
df.shape
df.columns
def feature_encode(feature):
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    print(le.classes_)
cat_df = df.select_dtypes(include='object')
cat_df.columns
for col in cat_df.columns:
    feature_encode(col)
df.head()
df.Income_Cat.value_counts()
df.dtypes
# scaler = StandardScaler()
# scaled_df = scaler.fit_transform(df.drop('Attrition',axis=1))
# X= scaled_df
# Y = df['Attrition'].to_numpy()
# X = df.loc[:,df.columns!='Attrition']
# X.head()
df.head()
# Y = df['Attrition']
# Y.head()
X_2 = df.loc[:, df.columns != "Attrition"].values # All columns except Attrition
y_2 = df.loc[:, df.columns == "Attrition"].values.flatten() # Attrition column and flatten to bring it to row format
X_2
y_2
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_2, y_2, test_size=0.2, stratify=y_2, random_state=1)
X_train_2
y_train_2
X_train_2[y_train_2 == 1].shape
X_train_u, y_train_u  = resample(X_train_2[y_train_2 == 1],
                                y_train_2[y_train_2==1],
                                 replace = True,
                                 n_samples=X_train_2[y_train_2 == 0].shape[0],
                                random_state=1
                                )
# Combine majority class with upsampled minority class
X_train_u = np.concatenate((X_train_2[y_train_2 == 0], X_train_u))
y_train_u = np.concatenate((y_train_2[y_train_2 == 0], y_train_u))

print("Original shape:", X_train_2.shape, y_train_2.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
# Build random forest classifier
methods_data = {"Original": (X_train_2, y_train_2),
                "Upsampled": (X_train_u, y_train_u)}

for method in methods_data.keys():
    pip_rf = make_pipeline(StandardScaler(),
                           RandomForestClassifier(n_estimators=500,
                                                  class_weight="balanced",
                                                  random_state=123))
    hyperparam_grid = {
        "randomforestclassifier__n_estimators": [10, 50, 100, 500],
        "randomforestclassifier__max_features": ["sqrt", "log2", 0.4, 0.5],
        "randomforestclassifier__min_samples_leaf": [1, 3, 5],
        "randomforestclassifier__criterion": ["gini", "entropy"]}
    
    gs_rf = GridSearchCV(pip_rf,
                         hyperparam_grid,
                         scoring="f1",
                         cv=10,
                         n_jobs=-1)
    
    gs_rf.fit(methods_data[method][0], methods_data[method][1])
    
    print("\033[1m" + "\033[0m" + "The best hyperparameters for {} data:".format(method))
    for hyperparam in gs_rf.best_params_.keys():
        print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_rf.best_params_[hyperparam])
    
    print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_rf.best_score_) * 100))
    
X_train_u[y_train_u == 0].shape, X_train_u[y_train_u == 1].shape
# Refit RF classifier using best params
clf_rf = make_pipeline(StandardScaler(),
                       RandomForestClassifier(n_estimators=500,
                                              criterion="gini",
                                              max_features='sqrt',
                                              min_samples_leaf=1,
                                              class_weight="balanced",
                                              n_jobs=-1,
                                              random_state=123))


clf_rf.fit(X_train_u, y_train_u)
# Plot confusion matrix and ROC curve
np.set_printoptions(precision=2)
disp = plot_confusion_matrix(clf_rf,X_test_2,y_test_2,display_labels=df.Attrition.name,cmap=plt.cm.Blues)
# Build Gradient Boosting classifier
pip_gb = make_pipeline(StandardScaler(),
                       GradientBoostingClassifier(loss="deviance",
                                                  random_state=123))

hyperparam_grid = {"gradientboostingclassifier__max_features": ["log2", 0.5],
                   "gradientboostingclassifier__n_estimators": [100, 300, 500],
                   "gradientboostingclassifier__learning_rate": [0.001, 0.01, 0.1],
                   "gradientboostingclassifier__max_depth": [1, 2, 3]}

gs_gb = GridSearchCV(pip_gb,
                      param_grid=hyperparam_grid,
                      scoring="f1",
                      cv=10,
                      n_jobs=-1)

gs_gb.fit(X_train_u, y_train_u)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_gb.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_gb.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_gb.best_score_) * 100))
# Build logistic model classifier
pip_logmod = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced"))

hyperparam_range = np.arange(0.5, 20.1, 0.5)

hyperparam_grid = {"logisticregression__penalty": ["l1", "l2"],
                   "logisticregression__C":  hyperparam_range,
                   "logisticregression__fit_intercept": [True, False]
                  }

gs_logmodel = GridSearchCV(pip_logmod,
                           hyperparam_grid,
                           scoring="accuracy",
                           cv=2,
                           n_jobs=-1)

gs_logmodel.fit(X_train_u, y_train_u)

print("\033[1m" + "\033[0m" + "The best hyperparameters:")
print("-" * 25)
for hyperparam in gs_logmodel.best_params_.keys():
    print(hyperparam[hyperparam.find("__") + 2:], ": ", gs_logmodel.best_params_[hyperparam])

print("\033[1m" + "\033[94m" + "Best 10-folds CV f1-score: {:.2f}%.".format((gs_logmodel.best_score_) * 100))
estimators = {"RF": clf_rf,
              "LR": gs_logmodel,
              "GBT": gs_gb
             }

# Print out accuracy score on test data
print("The accuracy rate and f1-score on test data are:")
for estimator in estimators.keys():
    print("{}: {:.2f}%, {:.2f}%.".format(estimator,
        accuracy_score(y_test_2, estimators[estimator].predict(X_test_2)) * 100,
         f1_score(y_test_2, estimators[estimator].predict(X_test_2)) * 100))
model_names=['RandomForestClassifier','Logistic Regression','GradientBoostingClassifier']
models = [clf_rf,gs_logmodel,gs_gb]
def compare_models(model):
    clf=model
    clf.fit(X_train_u,y_train_u)
    pred=clf.predict(X_test_2)
    
    # Calculating various metrics
    
    acc.append(accuracy_score(pred,y_test_2))
    prec.append(precision_score(pred,y_test_2))
    rec.append(recall_score(pred,y_test_2))
    auroc.append(roc_auc_score(pred,y_test_2))
acc=[]
prec=[]
rec=[]
auroc=[]
for model in models:
    compare_models(model)
d={'Modelling Algo':model_names,'Accuracy':acc,'Precision':prec,'Recall':rec,'Area Under ROC Curve':auroc}
met_df=pd.DataFrame(d)
met_df
clf_rf = RandomForestClassifier(n_estimators=500,
                                criterion="gini",
                                max_features='sqrt',
                                min_samples_leaf=1,
                                class_weight="balanced",
                                n_jobs=-1,
                                random_state=123)


clf_rf.fit(StandardScaler().fit_transform(X_train_u), y_train_u)

# Plot features importance
importances = clf_rf.feature_importances_
indices = np.argsort(clf_rf.feature_importances_)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(1, 24), importances[indices], align="center")
plt.xticks(range(1, 24), df.columns[df.columns != "Attrition"][indices], rotation=90)
plt.title("Feature Importance", {"fontsize": 16});
