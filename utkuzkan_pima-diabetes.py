# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale,StandardScaler, MinMaxScaler,Normalizer,RobustScaler
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Set Options
pd.options.display.max_columns
pd.options.display.max_rows = 500
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head(500)
df.info()




df.head(500)

df.isna().any()
dataset2 = df.drop(columns="Outcome")
fig = plt.figure(figsize=(15,12))
plt.suptitle("Histograms of Numerical Columns",fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6,3,i+1)
    f=plt.gca()
    f.set_title(dataset2.columns.values[i])
    
    vals=np.size(dataset2.iloc[:,i].unique())
    if vals >= 100:
        vals=100
    plt.hist(dataset2.iloc[:,i], bins=vals,color="#3F5D7D")
plt.tight_layout(rect=[0,0.03,1,0.95])    
    
# We observe zero values in Glucose, Blood Pressure,Skin Tickness,Inslulin and BMI. this values are wrong values
vars_with_zeros = [var for var in dataset2.columns if dataset2[dataset2[var] == 0].shape[0] > 0]
for column in vars_with_zeros:
    print("zero percentage of "+ column+" :" , dataset2[dataset2[column] == 0].shape[0] / dataset2.shape[0] )
for column in vars_with_zeros:
    print("zero percentage of "+ column+" :" , dataset2[dataset2[column] == 0].shape[0] / dataset2.shape[0] < 0.05 )
remove_columns = ["Glucose","BloodPressure","BMI"]

for remove in remove_columns:
    df= df[df[remove] != 0]
df.info()
dataset2.corrwith(df.Outcome).plot.bar(figsize=(20,10),title = "Correlation with Outcome",fontsize= 15, rot= 45,grid= True )
# we calculate the correlations using pandas corr
# and we round the values to 2 decimals
correlation_matrix = dataset2.corr().round(2) 

# plot the correlation matrix usng seaborn
# annot = True to print the correlation values
# inside the squares

figure = plt.figure(figsize=(12, 12))
sns.heatmap(data=correlation_matrix, annot=True)
df.info()
# let's trimm the dataset
insulin_outcome_one_lower_limit = np.min(df[(df["Outcome"]==1) & (df["Insulin"] > 0)]["Insulin"])
insulin_outcome_one_upper_limit = np.max(df[(df["Outcome"]==1) & (df["Insulin"] > 0)]["Insulin"])
insulin_outcome_one_lower_limit,insulin_outcome_one_upper_limit
# let's trimm the dataset
insulin_outcome_zero_lower_limit = np.min(df[(df["Outcome"]==0) & (df["Insulin"] > 0)]["Insulin"])
insulin_outcome_zero_upper_limit = np.max(df[(df["Outcome"]==0) & (df["Insulin"] > 0)]["Insulin"])
insulin_outcome_zero_lower_limit,insulin_outcome_zero_upper_limit
import random

# set Values For Insulin When outcome is zero

fill_missing_insulin = df.copy() 

for index, row in fill_missing_insulin.iterrows():
    if (row["Outcome"] == 1) & (row["Insulin"] == 0):
        fill_missing_insulin.loc[index, 'Insulin']  = random.randint(insulin_outcome_one_lower_limit,insulin_outcome_one_upper_limit)
    elif  (row["Outcome"] == 0) & (row["Insulin"] == 0):
        fill_missing_insulin.loc[index, 'Insulin']  = random.randint(insulin_outcome_zero_lower_limit,insulin_outcome_zero_upper_limit)
# let's trimm the dataset
skinThickness_outcome_zero_lower_limit = np.min(df[(df["Outcome"]==0) & (df["SkinThickness"] > 0)]["SkinThickness"])
skinThickness_outcome_zero_upper_limit = np.max(df[(df["Outcome"]==0) & (df["SkinThickness"] > 0)]["SkinThickness"])
skinThickness_outcome_zero_lower_limit,skinThickness_outcome_zero_upper_limit
# let's trimm the dataset
skinThickness_outcome_one_lower_limit = np.min(df[(df["Outcome"]==1) & (df["SkinThickness"] > 0)]["SkinThickness"])
skinThickness_outcome_one_upper_limit = np.max(df[(df["Outcome"]==1) & (df["SkinThickness"] > 0)]["SkinThickness"])
skinThickness_outcome_one_lower_limit,skinThickness_outcome_one_upper_limit 
for index, row in fill_missing_insulin.iterrows():
    if (row["Outcome"] == 1) & (row["SkinThickness"] == 0):
        fill_missing_insulin.loc[index, 'SkinThickness']  = random.randint(skinThickness_outcome_one_lower_limit,skinThickness_outcome_one_upper_limit)
    elif  (row["Outcome"] == 0) & (row["SkinThickness"] == 0):
        fill_missing_insulin.loc[index, 'SkinThickness']  = random.randint(skinThickness_outcome_zero_lower_limit,skinThickness_outcome_zero_upper_limit)
fill_missing_insulin.head(100)
df
fig = plt.figure(figsize=(15,12))
plt.suptitle("Histograms of Numerical Columns",fontsize=20)
for i in range(fill_missing_insulin.shape[1]):
    plt.subplot(6,3,i+1)
    f=plt.gca()
    f.set_title(fill_missing_insulin.columns.values[i])
    
    vals=np.size(fill_missing_insulin.iloc[:,i].unique())
    if vals >= 100:
        vals=100
    plt.hist(fill_missing_insulin.iloc[:,i], bins=vals,color="#3F5D7D")
plt.tight_layout(rect=[0,0.03,1,0.95]) 
#tüm sütünları yukardan bakarak  outlier feature'ları gözlemleme
sns.set(font_scale=0.7) 
fig, axes = plt.subplots(nrows=int(len(fill_missing_insulin.columns)/2), ncols=2,figsize=(7,12))
fig.tight_layout()
for ax,col in zip(axes.flatten(),fill_missing_insulin.columns):
    sns.boxplot(x=df[col],ax=ax)
from sklearn.neighbors import LocalOutlierFactor # çok değişkenli aykırı gözlem incelemesi

# LOF yöntemine göre aykırı değerlerin görselleştirilmesi
df_outlier = fill_missing_insulin.astype("float64")
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df_outlier)
df_scores = clf.negative_outlier_factor_
scores_df = pd.DataFrame(np.sort(df_scores))
scores_df.plot(stacked=True, xlim=[0,20], style='.-'); # ilk 20 gözlem
outlier_score=pd.DataFrame()
outlier_score['score']=df_scores
with pd.option_context("display.max_rows", 1000):
    display(outlier_score.sort_values(by='score', ascending=True))
# threshold -1.995415

np.sort(df_scores)[0]
th_val = np.sort(df_scores)[0]
outliers = df_scores > th_val
df = fill_missing_insulin.drop(df_outlier[~outliers].index)
df.shape
df
#tüm sütünları yukardan bakarak  outlier feature'ları gözlemleme
sns.set(font_scale=0.7) 
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2,figsize=(7,12))
fig.tight_layout()
for ax,col in zip(axes.flatten(),df.columns):
    sns.boxplot(x=df[col],ax=ax)
def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')


    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
distance = 1.5
def find_skewed_boundaries(df, variable):
    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)


    return upper_boundary, lower_boundary
# Set Limits For Features
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(df, 'Pregnancies')
RM_upper_limit, RM_lower_limit

df["Pregnancies"] = np.where(df["Pregnancies"] > RM_upper_limit,RM_upper_limit,
                            np.where(df["Pregnancies"] < RM_lower_limit,RM_lower_limit,df["Pregnancies"]))
# for the Q-Q plots
import scipy.stats as stats

# let's find outliers in Assists

diagnostic_plots(df, 'Pregnancies')
# Set Limits For Features
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(df, 'BloodPressure')
RM_upper_limit, RM_lower_limit

df["BloodPressure"] = np.where(df["BloodPressure"] > RM_upper_limit,RM_upper_limit,
                            np.where(df["BloodPressure"] < RM_lower_limit,RM_lower_limit,df["BloodPressure"]))
diagnostic_plots(df, 'BloodPressure')
# Set Limits For Features
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(df, 'Glucose')
RM_upper_limit, RM_lower_limit

df["Glucose"] = np.where(df["Glucose"] > RM_upper_limit,RM_upper_limit,
                            np.where(df["Glucose"] < RM_lower_limit,RM_lower_limit,df["Glucose"]))
diagnostic_plots(df, 'Glucose')
# Set Limits For Features
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(df, 'SkinThickness')
RM_upper_limit, RM_lower_limit

df["SkinThickness"] = np.where(df["SkinThickness"] > RM_upper_limit,RM_upper_limit,
                            np.where(df["SkinThickness"] < RM_lower_limit,RM_lower_limit,df["SkinThickness"]))
diagnostic_plots(df, 'SkinThickness')
# Set Limits For Features
#Find limits for Salary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(df, 'BMI')
RM_upper_limit, RM_lower_limit

df["BMI"] = np.where(df["BMI"] > RM_upper_limit,RM_upper_limit,
                            np.where(df["BMI"] < RM_lower_limit,RM_lower_limit,df["BMI"]))
diagnostic_plots(df, 'BMI')
X = df.drop(["Outcome"],axis=1)
X
y = df["Outcome"]
y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
X_train
logr_model = LogisticRegression().fit(X_train,y_train)
y_pred = logr_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

roc=roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(accuracy,roc,precision,recall)

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)
logr_model = LogisticRegression().fit(X_train,y_train)
y_pred = logr_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

roc=roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(accuracy,roc,precision,recall)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

models = []
models.append(('LOGR',LogisticRegression()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('RF',RandomForestClassifier()))
models.append(('SVC',SVC(kernel='linear',probability=True)))
models.append(('GBM',GradientBoostingClassifier()))
models.append(('XGBoost',XGBClassifier()))
models.append(('LightGBM',LGBMClassifier()))
models.append(('CatBoost',CatBoostClassifier()))

df_result = pd.DataFrame(columns=["model","accuracy_score","scale_method"])
index = 0
for name,model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    df_result.at[index,['model','accuracy_score','scale_method']] = [name,score,"NA"]
    index += 1
    
df_result
df_result.sort_values("accuracy_score",ascending=False)
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_val_score
df_result_cross = pd.DataFrame(columns=["model","accuracy_score","scale_method","cross_val"])
index = 0
scoring = 'accuracy'
for name,model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=12345)
    score = cross_val_score(estimator = model, X = X_train, y = y_train, cv = kfold,scoring=scoring)
    df_result_cross.at[index,['model','accuracy_score','scale_method',"cross_val"]] = [name,score.mean(),"SC",1]
    index += 1

df_result_cross
df_result_cross.sort_values("accuracy_score",ascending=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
from sklearn.preprocessing import StandardScaler
sc = RobustScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

df_result_scale = pd.DataFrame(columns=["model","accuracy_score","scale_method"])
index = 0
for name,model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    df_result_scale.at[index,['model','accuracy_score','scale_method']] = [name,score,"StandardScaler"]
    index += 1
    
df_result_scale
df_result_scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
xgb_model=XGBClassifier(silent=0, learning_rate=0.01, max_delta_step=5,
                            objective='reg:logistic',n_estimators=100, 
                            max_depth=3, eval_metric="logloss", gamma=3)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test,y_pred,digits=2))
print(accuracy_score(y_test, y_pred))