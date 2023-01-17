#Importing the required libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing

from scipy.stats import kurtosis
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score

# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsClassifier

#Importing the libraries for XGBoost.
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Suppress warnings
import warnings
warnings.filterwarnings('ignore')
import time
#To display a max of 40 columns of the dataframe
pd.set_option('display.max_columns',40)
#Reading the dataset to the dataframe df
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
#Head of the dataset
df.head()
#Dimension of the dataframe
df.shape
#More information of the data type of the features.
df.info()
df.describe()
#Checking the timeline of the transactions present in the dataframe
max(df['Time'])/(60*60)
classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100
# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
dist_df = pd.DataFrame({'Percentage':[normal_share,fraud_share]},index=['Normal','Fraudulent'])
sns.barplot(x=dist_df.index,y=dist_df['Percentage'],palette='RdYlGn')
plt.title('Fraudulent vs Non-Fraudulent')
plt.show()
dist_df
# Create a scatter plot to observe the distribution of classes with time
sns.scatterplot(x=df['Time'],y=df['Class'])
plt.show()
# Create a scatter plot to observe the distribution of classes with Amount
sns.scatterplot(x=df['Class'],y=df['Amount'])
plt.show()
# Drop unnecessary columns
df.drop(['Time'],axis=1,inplace=True)
sns.boxplot(y=df['Amount'])
plt.show()
df.loc[df['Amount']>=10000]
df.loc[df['Amount']>=10000]['Amount'].count()
#Before removing the records
df.shape
df = df.loc[df['Amount']<10000]
#We can recheck the stats for the amount column now again, after removing the Outliers.
print(df['Amount'].describe())
fig, ax = plt.subplots(1, 2, figsize=(18,4))

#plt.subplot(121)
sns.distplot(df[df['Class']==0]['Amount'],bins=50, ax=ax[0], color='brown')
ax[0].set_title('Amount distribution for Non-fraudulent transactions', fontsize=10)

#plt.subplot(122)
sns.distplot(df[df['Class']==1]['Amount'],bins=50, ax=ax[1], color='purple')
ax[1].set_title('Amount distribution for fraudulent transactions', fontsize=10)
#Finding the Min-Max values for fraudulent and valid transactions after the Outlier Removal.
print('Min amount for a fraudulent transaction:', df[df['Class']==1]['Amount'].min())
print('Max amount for a fraudulent transaction:', df[df['Class']==1]['Amount'].max())
print('Min amount for a valid transaction:', df[df['Class']==0]['Amount'].min())
print('Max amount for a valid transaction:', df[df['Class']==0]['Amount'].max())
print('No of transactions where amount is 0 for fraudulent transactions:', 
      df[(df['Class']==1) & (df['Amount']==0)].shape[0])
print('No of transactions where amount is 0 for valid transactions:', 
      df[(df['Class']==0) & (df['Amount']==0)].shape[0])
plt.figure(figsize=(17,8))
sns.heatmap(round(df.corr(),2),annot=True,cmap='Blues')
plt.show()
#Maintaining a copy of the dataframe
df_copy1 = df.copy()
X = df.drop(['Class'],axis=1)
y= df['Class']
#Number of Class 1 records
len(y.loc[y==1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,test_size=0.3,random_state=100)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()
len(y_train.loc[y==1])
len(y_test.loc[y==1])
# Defining a function that can be useful for violin plot
def pltViolin(subpltnum,colName,dfObj,hueCol=None):
    if ',' in str(subpltnum):
        nums=str(subpltnum).split(',')
        plt.subplot(int(nums[0]),int(nums[1]),int(nums[2]))
    else:
        plt.subplot(subpltnum)
    if hueCol==None:
        sns.violinplot(y=colName,data=dfObj)
    else:
        sns.violinplot(y=colName,data=dfObj,hue=hueCol)
    
# Defining a function that can be useful for Bar plot
def pltBar(subpltnum,xCol,yCol,dfObj):
    if ',' in str(subpltnum):
        nums=str(subpltnum).split(',')
        plt.subplot(int(nums[0]),int(nums[1]),int(nums[2]))
    else:
        plt.subplot(subpltnum)
    sns.barplot(x=xCol,y=yCol,data=dfObj)

# Defining a function that can be useful for Reg plot
def pltReg(subpltnum,xCol,yCol,dfObj):
    if ',' in str(subpltnum):
        nums=str(subpltnum).split(',')
        plt.subplot(int(nums[0]),int(nums[1]),int(nums[2]))
    else:
        plt.subplot(subpltnum)
    sns.regplot(x=xCol,y=yCol,data=dfObj)

# Defining a function that can be useful for box plot
def pltBox(subpltnum,xCol,yCol,dfObj,hueCol=None):
    if ',' in str(subpltnum):
        nums=str(subpltnum).split(',')
        plt.subplot(int(nums[0]),int(nums[1]),int(nums[2]))
    else:
        plt.subplot(subpltnum)
    if hueCol==None:
        sns.boxplot(x=xCol,y=yCol,data=dfObj)
    else:
        sns.boxplot(x=xCol,y=yCol,data=dfObj,hue=hueCol)
        
def pltCount(subpltnum,colName,dfObj,hueCol=None,orderCol=None):
    if ',' in str(subpltnum):
        nums=str(subpltnum).split(',')
        plt.subplot(int(nums[0]),int(nums[1]),int(nums[2]))
    else:
        plt.subplot(subpltnum)

    if hueCol==None:
        if orderCol == None:
            sns.countplot(x=colName,data=dfObj)
        else:
            sns.countplot(x=colName,data=dfObj,order=dfObj[orderCol].value_counts().index)
    else:
        if orderCol == None:
            sns.countplot(x=colName,data=dfObj,hue=hueCol)
        else:
            sns.countplot(x=colName,data=dfObj,hue=hueCol,order=dfObj[orderCol].value_counts().index)
# plot the histogram of a variable from the dataset to see the skewness
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 1

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 5

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 9

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 13

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 17

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 21

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
plt.figure(figsize=(16,20))

subplotNum = "4,2,"
pltNum = 1
colNumSuf = 25

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)
    pltBox(pltNo,'Class',col,df)
    
    pltNum = pltNum + 1

    plt.subplot(4,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    
    pltNum = pltNum + 1
X_train_Summary = pd.DataFrame(X_train.columns.to_list(), columns =['Variables'])
X_train_Summary['min'] = -100000
X_train_Summary['max'] = -100000
X_train_Summary['skew'] = -100000
X_train_Summary['kurtosis'] = -100000

for col in X_train.columns.to_list():
    X_train_Summary.loc[X_train_Summary.Variables == col,'min'] = np.round(X_train[col].min())
    X_train_Summary.loc[X_train_Summary.Variables == col,'max'] = np.round(X_train[col].max())
    X_train_Summary.loc[X_train_Summary.Variables == col,'skew'] = np.round(skew(X_train[col]),2)
    X_train_Summary.loc[X_train_Summary.Variables == col,'kurtosis'] = np.round(kurtosis(X_train[col]),2)
X_train_Summary
train_num_std = ['Amount']
train_num_yjt = X_train.columns
yj_trans = PowerTransformer(method='yeo-johnson')
scaler   = StandardScaler()
# standardization of Amount feature
X_train[train_num_std] = scaler.fit_transform(X_train[train_num_std].values)
X_test[train_num_std] = scaler.transform(X_test[train_num_std].values)
# power transform
X_train[train_num_yjt] = yj_trans.fit_transform(X_train[train_num_yjt].values)
X_test[train_num_yjt] = yj_trans.transform(X_test[train_num_yjt].values)
X_train_Summary_pw = pd.DataFrame(X_train.columns.to_list(), columns =['Variables'])
X_train_Summary_pw['min'] = -100000
X_train_Summary_pw['max'] = -100000
X_train_Summary_pw['skew'] = -100000
X_train_Summary_pw['kurtosis'] = -100000

for col in X_train.columns.to_list():
    X_train_Summary_pw.loc[X_train_Summary_pw.Variables == col,'min'] = np.round(X_train[col].min())
    X_train_Summary_pw.loc[X_train_Summary_pw.Variables == col,'max'] = np.round(X_train[col].max())
    X_train_Summary_pw.loc[X_train_Summary_pw.Variables == col,'skew'] = np.round(skew(X_train[col]),2)
    X_train_Summary_pw.loc[X_train_Summary_pw.Variables == col,'kurtosis'] = np.round(kurtosis(X_train[col]),2)
X_train_Summary_pw
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 1

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 5

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 9

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 13

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 17

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 21

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
plt.figure(figsize=(16,10))

subplotNum = "2,2,"
pltNum = 1
colNumSuf = 25

for i in range(0,4):
    col = "V" + str(colNumSuf)
    colNumSuf = colNumSuf + 1
    
    pltNo = subplotNum + str(pltNum)

    plt.subplot(2,2,pltNum)
    sns.distplot(X_train[col],bins=50)
    plt.xlabel(col + ', Skewness: ' + str(np.round(skew(X_train[col]),2)) + ', Kurtosis: ' + str(np.round(kurtosis(X_train[col]),2)))
    pltNum = pltNum + 1
#Backup of X_train
X_train_copy = X_train.copy()
y_train.shape
X_train.shape
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score
decision_tree = DecisionTreeClassifier(max_depth=5)
decision_tree.fit(X_train,y_train)
y_train_pred_DT = decision_tree.predict(X_train).astype(int)
decision_tree.score(X_train,y_train)
# Printing classification report
print(classification_report(y_train, y_train_pred_DT))
confusion1 = confusion_matrix(y_train,y_train_pred_DT)
confusion1
# Predicted     not_fraudulent    fraudulent
# Actual
# not_fraudulent    199016            11
# fraudulent            66           266
TP = confusion1[1,1] # true positive 
TN = confusion1[0,0] # true negatives
FP = confusion1[0,1] # false positives
FN = confusion1[1,0] # false negatives
#Sensitivity
TP / float(TP+FN)
#Specificity
TN / float(TN+FP)
#Precision
TP / float(TP+FP)
# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
start = time.time()
# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 3

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree,param_grid = param_grid,cv = n_folds,verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)

end = time.time()
elapsed = end - start
elapsed / 60
# printing the optimal accuracy score and hyperparameters
print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)
# model with optimal hyperparameters
dt_opt = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
dt_opt.fit(X_train,y_train)
y_train_pred_DT1 = dt_opt.predict(X_train).astype(int)
# Printing classification report
print(classification_report(y_train, y_train_pred_DT1))
confusion2 = confusion_matrix(y_train,y_train_pred_DT1)
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
confusion2
#Sensitivity
TP / float(TP+FN)
#Specificity
TN / float(TN+FP)
#Precision
TP / float(TP+FP)
# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 15, 5)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters,cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train,y_train)
end = time.time()
(end - start)/60
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training recall")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(500, 1500, 500)}
# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=7)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train,y_train)
end = time.time()
#print("Time took for the Hyperparameter tuning:",((end-start)/60))
(end - start)/60
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="test accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}
# instantiate the model
rf = RandomForestClassifier(max_depth=7)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train,y_train)
end = time.time()
(end - start)/60
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_features
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_leaf': range(100,300,50)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train,y_train)
end = time.time()
(end-start)/60
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_split': range(100, 300, 50)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train,y_train)
end = time.time()
(end-start)/60
 # scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Recall")
plt.legend()
plt.show()
rf = RandomForestClassifier(n_estimators=500,max_depth=7,max_features=13,min_samples_leaf=75,min_samples_split=150)
rf.fit(X_train,y_train)
y_train_pred_rf = rf.predict(X_train)
accuracy_score(y_train,y_train_pred_rf)
confusion3 = confusion_matrix(y_train,y_train_pred_rf)
TP = confusion3[1,1] # true positive 
TN = confusion3[0,0] # true negatives
FP = confusion3[0,1] # false positives
FN = confusion3[1,0] # false negatives
confusion3
#Sensitivity
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP+FP)
X_test.shape
y_test.shape
y_test_pred_rf = rf.predict(X_test)
accuracy_score(y_test,y_test_pred_rf)
confusion4 = confusion_matrix(y_test,y_test_pred_rf)
TP = confusion4[1,1] # true positive 
TN = confusion4[0,0] # true negatives
FP = confusion4[0,1] # false positives
FN = confusion4[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP+FP)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
import statsmodels.api as sm
model1 = sm.GLM(y_train,(sm.add_constant(X_train)),family=sm.families.Binomial())
model1.fit().summary()
X_train_logreg = X_train.copy()
X_train_logreg.drop(['Amount'],axis=1,inplace=True)
X_train_logreg.columns
model2 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model2.fit().summary()
X_train_logreg.drop(['V15'],axis=1,inplace=True)
model3 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model3.fit().summary()
X_train_logreg.drop(['V17'],axis=1,inplace=True)
model4 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model4.fit().summary()
X_train_logreg.drop(['V24'],axis=1,inplace=True)
model5 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model5.fit().summary()
X_train_logreg.drop(['V11'],axis=1,inplace=True)
model6 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model6.fit().summary()
X_train_logreg.drop(['V5'],axis=1,inplace=True)
model7 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model7.fit().summary()
X_train_logreg.drop(['V26'],axis=1,inplace=True)
model8 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model8.fit().summary()
X_train_logreg.drop(['V20'],axis=1,inplace=True)
model9 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model9.fit().summary()
X_train_logreg.drop(['V27'],axis=1,inplace=True)
model10 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model10.fit().summary()
X_train_logreg.drop(['V28'],axis=1,inplace=True)
model11 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model11.fit().summary()
X_train_logreg.drop(['V18'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model12.fit().summary()
X_train_logreg.drop(['V25'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model12.fit().summary()
X_train_logreg.drop(['V23'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model12.fit().summary()
X_train_logreg.drop(['V9'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model12.fit().summary()
X_train_logreg.drop(['V2'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
model12.fit().summary()
X_train_logreg.drop(['V6'],axis=1,inplace=True)
model12 = sm.GLM(y_train,(sm.add_constant(X_train_logreg)),family=sm.families.Binomial())
result = model12.fit()
result.summary()
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables that w used before and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_logreg.columns
vif['VIF'] = [variance_inflation_factor(X_train_logreg.values, i) for i in range(X_train_logreg.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_df_logreg = pd.DataFrame(data=y_train.values,columns=['y_train'])
y_df_logreg.head()
#X_train_logreg
y_train_pred_logreg = result.predict(sm.add_constant(X_train_logreg)).values.reshape(-1)
y_train_pred_logreg[:10]
y_df_logreg['y_train_pred_logreg'] = y_train_pred_logreg
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_df_logreg[i]= y_df_logreg.y_train_pred_logreg.map(lambda x: 1 if x > i else 0)
y_df_logreg.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_df_logreg.y_train, y_df_logreg[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df1 = pd.DataFrame( columns = ['probability','precision','recall'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_df_logreg.y_train, y_df_logreg[i] )
    total1=sum(sum(cm1))
    prec = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df1.loc[i] =[ i ,prec, rec]
print(cutoff_df1)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df1.plot.line(x='probability', y=['precision','recall'])
plt.show()
y_df_logreg['Final_predicted'] = y_df_logreg.y_train_pred_logreg.map( lambda x: 1 if x > 0.1 else 0)
y_df_logreg.head()
final_confusion = confusion_matrix(y_df_logreg['y_train'],y_df_logreg['Final_predicted'])
print(final_confusion)
TP = final_confusion[1,1] # true positive 
TN = final_confusion[0,0] # true negatives
FP = final_confusion[0,1] # false positives
FN = final_confusion[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP+FN)
#Precision
TP / float(TP + FP)
#Specificity
TN / float(TN + FP)
X_train_logreg_cols = list(X_train_logreg.columns)
#Building the test dataset with the relevant columns of our logistic regression model
X_test_logreg = X_test[X_train_logreg_cols]
#X_test_logreg
y_test
#X_train_logreg
y_test_pred_logreg = result.predict(sm.add_constant(X_test_logreg)).values.reshape(-1)
y_df_logreg_test = pd.DataFrame(data=y_test.values,columns=['y_test'])
y_df_logreg_test['y_test_prob'] = y_test_pred_logreg
y_df_logreg_test['final_predicted'] = y_df_logreg_test.y_test_prob.map( lambda x: 1 if x > 0.1 else 0)
final_confusion_test = confusion_matrix(y_df_logreg_test['y_test'],y_df_logreg_test['final_predicted'])
final_confusion_test
TP = final_confusion_test[1,1] # true positive 
TN = final_confusion_test[0,0] # true negatives
FP = final_confusion_test[0,1] # false positives
FN = final_confusion_test[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP+FN)
#Precision
TP / float(TP + FP)
#Specificity
TN / float(TN + FP)
from sklearn.neighbors import KNeighborsClassifier
start = time.time()
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
end = time.time()
(end-start)/60
y_train_pred_knn = knn.predict(X_train)
recall_score(y_train,y_train_pred_knn)
confusion_knn = confusion_matrix(y_train,y_train_pred_knn)
TP = confusion_knn[1,1] # true positive 
TN = confusion_knn[0,0] # true negatives
FP = confusion_knn[0,1] # false positives
FN = confusion_knn[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP + FP)
#Importing the libraries for XGBoost.
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
#!nvidia-smi
#dtrain = xgb.DMatrix(X_train,label=y_train)
#dtest  = xgb.DMatrix(X_test,label=y_test)
#import time
#setting tree and tree depth
#num_round = 50
#maxdepth = 8
#param = {
#  'colsample_bylevel': 1,
#  'colsample_bytree': 1,
#  'gamma': 0,
#  'learning_rate': 0.1, 
#  'random_state': 1010,
#  'objective': 'multi:softmax', 
#  'num_class': 7, 
#}
#param['tree_method'] = 'gpu_hist'
#param['grow_policy'] = 'depthwise'
#param['max_depth'] = maxdepth
#param['max_leaves'] = 0
#param['verbosity'] = 0
#param['gpu_id'] = 0
#param['updater'] = 'grow_gpu_hist'
#param['predictor'] = 'gpu_predictor'
#param['eval_metric'] = 'auc'

#gpu_result = {} 
#start_time = time.time()
# Training with the above parameters
#xgb_model = xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_result, verbose_eval=20)

#print("GPU Training Time: %s seconds" % (str(time.time() - start_time)))
#!nvidia-smi
import time
xgb_model = XGBClassifier()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}
# instantiate the model
xgb_model = XGBClassifier()
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train,y_train)
end = time.time()
(end-start)/60
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with max_depth
plt.figure()
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_train_score"], 
         label="training recall")
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(100, 800, 300)}
# instantiate the model (note we are specifying a max_depth)
xgb_model = XGBClassifier(max_depth=4)
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train,y_train)
end = time.time()
(end-start)/60
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with n_estimators
plt.figure()
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_train_score"], 
         label="train recall")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# hyperparameter tuning with XGBoost
# creating a KFold object 
folds = 3
# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=4, n_estimators=100)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'recall', 
                        cv = folds,  
                        verbose = 1,
                        return_train_score=True)  
model_cv.fit(X_train,y_train)
end = time.time()
(end-start)/60
# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')
cv_results.head()
#plotting
plt.figure(figsize=(16,6))
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 
for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('recall')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.2,
          'max_depth': 4, 
          'n_estimators':100,
          'subsample':0.3,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train,y_train)
# predict
y_pred = model.predict_proba(X_train)
y_pred[:10]
import sklearn
# roc_auc
auc = sklearn.metrics.roc_auc_score(y_train, y_pred[:, 1])
auc
xgb_df = pd.DataFrame(y_pred[:, 1],columns=["Predicted_probability"])
xgb_df['Churn'] = y_train.values
xgb_df
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    xgb_df[i]= xgb_df.Predicted_probability.map(lambda x: 1 if x > i else 0)
xgb_df.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df1 = pd.DataFrame( columns = ['probability','precision','recall'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    prec = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df1.loc[i] =[ i ,prec, rec]
print(cutoff_df1)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df1.plot.line(x='probability', y=['precision','recall'])
plt.show()
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_df['Final_predicted'] = xgb_df.Predicted_probability.map( lambda x: 1 if x > 0.1 else 0)
xgb_df.head()
y_pred_test = model.predict_proba(X_test)
y_pred_test[:10]
xgb_test_df_ca = pd.DataFrame(y_pred_test[:, 1],columns=["Predicted_probability"])
xgb_test_df_ca['y_test'] = y_test.values
xgb_test_df_ca
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_test_df_ca['Final_predicted'] = xgb_test_df_ca.Predicted_probability.map( lambda x: 1 if x > 0.1 else 0)
xgb_test_df_ca.head()
confusion_xgb = confusion_matrix(xgb_test_df_ca['y_test'],xgb_test_df_ca['Final_predicted'])
confusion_xgb
TP = confusion_xgb[1,1] # true positive 
TN = confusion_xgb[0,0] # true negatives
FP = confusion_xgb[0,1] # false positives
FN = confusion_xgb[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP + FP)
from imblearn.over_sampling import ADASYN
#Initialising the ADASYN object
ada = ADASYN(sampling_strategy='minority',random_state=101)
#fit_resample on the train sets to produce the new resampled sets.
X_train_ada,y_train_ada = ada.fit_resample(X_train,y_train)
X_train_ada.shape
y_train_ada.shape
X_train_ada = pd.DataFrame(X_train_ada,columns=X_train.columns)
y_train_ada = pd.DataFrame(y_train_ada,columns=['Fraud'])
#Data Inbalance check for the Converted column.
yes=y_train_ada[y_train_ada['Fraud']==1]['Fraud'].value_counts()
no=y_train_ada[y_train_ada['Fraud']==0]['Fraud'].value_counts()

converted=np.array((yes/len(y_train_ada))*100) 
not_converted=np.array((no/len(y_train_ada))*100) 
stat_summ=pd.DataFrame({'Percentage':[converted[0],not_converted[0]]},index=['Fraud','Not_Fraud'])
sns.barplot(x=stat_summ.index,y=stat_summ['Percentage'],palette='RdYlGn')
plt.title('Check on the data imbalance')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg,17)
rfe = rfe.fit(X_train_ada,y_train_ada)
cols = X_train_ada.columns[rfe.support_]
cols
X_train_ada_lr = X_train_ada[cols]
model1 = sm.GLM(y_train_ada,(sm.add_constant(X_train_ada_lr)),family=sm.families.Binomial())
model1.fit().summary()
# Create a dataframe that will contain the names of all the feature variables that w used before and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_ada_lr.columns
vif['VIF'] = [variance_inflation_factor(X_train_ada_lr.values, i) for i in range(X_train_ada_lr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_ada_lr.drop(['V10'],axis=1,inplace=True)
model2 = sm.GLM(y_train_ada,(sm.add_constant(X_train_ada_lr)),family=sm.families.Binomial())
model2.fit().summary()
# Create a dataframe that will contain the names of all the feature variables that w used before and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_ada_lr.columns
vif['VIF'] = [variance_inflation_factor(X_train_ada_lr.values, i) for i in range(X_train_ada_lr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_ada_lr.drop(['V17'],axis=1,inplace=True)
model3 = sm.GLM(y_train_ada,(sm.add_constant(X_train_ada_lr)),family=sm.families.Binomial())
model3.fit().summary()
# Create a dataframe that will contain the names of all the feature variables that w used before and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_ada_lr.columns
vif['VIF'] = [variance_inflation_factor(X_train_ada_lr.values, i) for i in range(X_train_ada_lr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_ada_lr.drop(['V14'],axis=1,inplace=True)
model4 = sm.GLM(y_train_ada,(sm.add_constant(X_train_ada_lr)),family=sm.families.Binomial())
result = model4.fit()
result.summary()
# Create a dataframe that will contain the names of all the feature variables that w used before and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_ada_lr.columns
vif['VIF'] = [variance_inflation_factor(X_train_ada_lr.values, i) for i in range(X_train_ada_lr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_df_logreg_ada = pd.DataFrame(data=y_train_ada.values,columns=['y_train_ada'])
#y_train_pred_logreg_ada
y_train_pred_logreg_ada = result.predict(sm.add_constant(X_train_ada_lr)).values.reshape(-1)
y_train_pred_logreg_ada[:10]
y_df_logreg_ada.shape
y_train_pred_logreg_ada.shape
y_df_logreg_ada['y_train_pred'] = y_train_pred_logreg_ada
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_df_logreg_ada[i]= y_df_logreg_ada.y_train_pred.map(lambda x: 1 if x > i else 0)
y_df_logreg_ada.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_df_logreg_ada.y_train_ada, y_df_logreg_ada[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df1 = pd.DataFrame( columns = ['probability','precision','recall'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(y_df_logreg_ada.y_train_ada, y_df_logreg_ada[i] )
    total1=sum(sum(cm1))
    prec = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df1.loc[i] =[ i ,prec, rec]
print(cutoff_df1)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df1.plot.line(x='probability', y=['precision','recall'])
plt.show()
X_train_ada_cols = list(X_train_ada_lr.columns)
#Building the test dataset with the relevant columns of our logistic regression model
X_test_logreg = X_test[X_train_ada_cols]
X_test_logreg.shape
y_test.shape
#X_train_logreg
y_test_pred_logreg = result.predict(sm.add_constant(X_test_logreg)).values.reshape(-1)
y_df_logreg_test = pd.DataFrame(data=y_test.values,columns=['y_test'])
y_df_logreg_test['y_test_prob'] = y_test_pred_logreg
y_df_logreg_test['final_predicted'] = y_df_logreg_test.y_test_prob.map( lambda x: 1 if x > 0.5 else 0)
final_confusion_test = confusion_matrix(y_df_logreg_test['y_test'],y_df_logreg_test['final_predicted'])
final_confusion_test
TP = final_confusion_test[1,1] # true positive 
TN = final_confusion_test[0,0] # true negatives
FP = final_confusion_test[0,1] # false positives
FN = final_confusion_test[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP + FP)
X_train_ada.shape
y_train_ada.shape
X_train_ada_new = X_train_ada.copy()
X_train_ada_new['Fraud'] = y_train_ada['Fraud']
X_train_ada_fraud = X_train_ada_new.loc[X_train_ada_new['Fraud']==1]
X_train_ada_non_fraud = X_train_ada_new.loc[X_train_ada_new['Fraud']==0]
X_train_ada_fraud.shape
X_train_ada_non_fraud.shape
X_train_ada_fraud = X_train_ada_fraud.sample(frac=1)
X_train_ada_non_fraud = X_train_ada_non_fraud.sample(frac=1)
X_train_ada_fraud = X_train_ada_fraud[:50000]
X_train_ada_non_fraud = X_train_ada_non_fraud[:50000]
frames = [X_train_ada_fraud,X_train_ada_non_fraud]
X_train_ada_final = pd.concat(frames)
X_train_ada_final = X_train_ada_final.sample(frac=1)
X_train_ada_final
X_train_ada_sampled = X_train_ada_final.drop(['Fraud'],axis=1)
y_train_ada_sampled = X_train_ada_final['Fraud']
from sklearn.ensemble import RandomForestClassifier
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 15, 3)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters,cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training recall")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(200, 1500, 500)}
# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=6)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="test accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}
# instantiate the model
rf = RandomForestClassifier(max_depth=6)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_features
plt.figure()
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_leaf': range(50,300,50)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf
plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_split': range(100, 300, 50)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
 # scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split
plt.figure()
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Recall")
plt.legend()
plt.show()
rf = RandomForestClassifier(n_estimators=200,max_depth=6,max_features=5,min_samples_leaf=250,min_samples_split=500)
rf.fit(X_train_ada_sampled,y_train_ada_sampled)
y_train_pred_rf_ada = rf.predict(X_train_ada_sampled)
accuracy_score(y_train_ada_sampled,y_train_pred_rf_ada)
confusion = confusion_matrix(y_train_ada_sampled,y_train_pred_rf_ada)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
confusion
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP+FP)
X_test.shape
y_test.shape
y_test_pred_rf_ada = rf.predict(X_test)
accuracy_score(y_test,y_test_pred_rf_ada)
confusion_rf = confusion_matrix(y_test,y_test_pred_rf_ada)
confusion_rf
TP = confusion_rf[1,1] # true positive 
TN = confusion_rf[0,0] # true negatives
FP = confusion_rf[0,1] # false positives
FN = confusion_rf[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP+FP)
#Importing the libraries for XGBoost.
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 20, 5)}
# instantiate the model
xgb_model = XGBClassifier()
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with max_depth
plt.figure()
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_train_score"], 
         label="training recall")
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(100, 1000, 200)}
# instantiate the model (note we are specifying a max_depth)
xgb_model = XGBClassifier(max_depth=4)
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with n_estimators
plt.figure()
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_train_score"], 
         label="train recall")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# hyperparameter tuning with XGBoost
# creating a KFold object 
folds = 3
# specify range of hyperparameters
param_grid = {'learning_rate': [0.2,0.4,0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=4, n_estimators=100)
# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'recall', 
                        cv = folds,  
                        verbose = 1,
                        return_train_score=True)  
model_cv.fit(X_train_ada_sampled,y_train_ada_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')
cv_results.head()
#plotting
plt.figure(figsize=(16,6))
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]} 
for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('recall')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.995, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.2,
          'max_depth': 4, 
          'n_estimators':100,
          'subsample':0.3,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train_ada_sampled,y_train_ada_sampled)
# predict
y_pred = model.predict_proba(X_train_ada_sampled)
y_pred[:10]
import sklearn
# roc_auc
auc = sklearn.metrics.roc_auc_score(y_train_ada_sampled, y_pred[:, 1])
auc
xgb_df = pd.DataFrame(y_pred[:, 1],columns=["Predicted_probability"])
xgb_df['Churn'] = y_train_ada_sampled.values
xgb_df
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    xgb_df[i]= xgb_df.Predicted_probability.map(lambda x: 1 if x > i else 0)
xgb_df.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df1 = pd.DataFrame( columns = ['probability','precision','recall'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    prec = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df1.loc[i] =[ i ,prec, rec]
print(cutoff_df1)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df1.plot.line(x='probability', y=['precision','recall'])
plt.show()
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_df['Final_predicted'] = xgb_df.Predicted_probability.map( lambda x: 1 if x > 0.55 else 0)
xgb_df.head()
#Function to plot for the Reciever Operating Characteristics (ROC) and to find out the AUC.
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve(xgb_df['Churn'],xgb_df['Final_predicted'], drop_intermediate = False )
draw_roc(xgb_df['Churn'],xgb_df['Final_predicted'])
y_pred_test = model.predict_proba(X_test)
y_pred_test[:10]
xgb_test_df = pd.DataFrame(y_pred_test[:, 1],columns=["Predicted_probability"])
xgb_test_df['y_test'] = y_test.values
xgb_test_df.head()
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_test_df['Final_predicted'] = xgb_test_df.Predicted_probability.map( lambda x: 1 if x > 0.55 else 0)
xgb_test_df.head()
confusion_xgb = confusion_matrix(xgb_test_df['y_test'],xgb_test_df['Final_predicted'])
confusion_xgb
TP = confusion_xgb[1,1] # true positive 
TN = confusion_xgb[0,0] # true negatives
FP = confusion_xgb[0,1] # false positives
FN = confusion_xgb[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP + FP)
start = time.time()

#
rf_ro = RandomForestClassifier(class_weight={0:0.001, 1:0.999},n_jobs=1)

params = {
        'max_depth': range(2, 14, 2)
}

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

rf_ro = GridSearchCV(estimator = rf_ro,
                    cv=folds,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1)

# fit
rf_ro.fit(X_train, y_train)

end = time.time()

print('I took:',np.round((end-start)/60,2),' minutes to complete !!')
start = time.time()

#
rf_ro = RandomForestClassifier(max_depth=4, class_weight={0:0.001, 1:0.999},n_jobs=1)

params = {
        'n_estimators': range(200, 801, 200)
}

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

rf_ro = GridSearchCV(estimator = rf_ro,
                    cv=folds,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1)

# fit
rf_ro.fit(X_train, y_train)

end = time.time()

print('I took:',np.round((end-start)/60,2),' minutes to complete !!')
rf_ro.best_params_
start = time.time()

#
rf_ro3 = RandomForestClassifier(max_depth=4,
                               n_estimators=200,
                               class_weight={0:0.001, 1:0.999},
                               n_jobs=1)

params = {
        'max_features': range(5,21,5)
}

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

rf_ro3 = GridSearchCV(estimator = rf_ro3,
                    cv=folds,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1)

# fit
rf_ro3.fit(X_train, y_train)

end = time.time()

print('I took:',np.round((end-start)/60,2),' minutes to complete !!')
rf_ro3.best_params_
start = time.time()

#
rf_ro4 = RandomForestClassifier(max_depth=4,
                               n_estimators=200,
                               max_features=10,
                               class_weight={0:0.001, 1:0.999},
                               n_jobs=1)

params = {
        'min_samples_leaf': range(5, 21, 5)
}

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

rf_ro4 = GridSearchCV(estimator = rf_ro4,
                    cv=folds,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1)

# fit
rf_ro4.fit(X_train, y_train)

end = time.time()

print('I took:',np.round((end-start)/60,2),' minutes to complete !!')
rf_ro4.best_params_
start = time.time()

#
rf_ro5 = RandomForestClassifier(max_depth=4,
                               n_estimators=200,
                               max_features=10,
                               min_samples_leaf=15,
                               class_weight={0:0.001, 1:0.999},
                               n_jobs=1)

params = {
        'min_samples_split': range(10, 101, 20)
}

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 100)

rf_ro5 = GridSearchCV(estimator = rf_ro5,
                    cv=folds,
                    param_grid=params,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1)

# fit
rf_ro5.fit(X_train, y_train)

end = time.time()

print('I took:',np.round((end-start)/60,2),' minutes to complete !!')
rf_ro5.best_params_
rf_roTuned = RandomForestClassifier(max_depth=4,
                               n_estimators=200,
                               max_features=10,
                               min_samples_leaf=15,
                               min_samples_split=90,
                               class_weight={0:0.001, 1:0.999},
                               n_jobs=1,
                               random_state = 100)
rf_roTuned.fit(X_train, y_train)

y_train_pred = rf_roTuned.predict(X_train)
y_test_pred = rf_roTuned.predict(X_test)
# Printing classification report
print(classification_report(y_train, y_train_pred))
print(metrics.accuracy_score(y_train, y_train_pred))
# find precision score and recall score
precisionScore = precision_score(y_train, y_train_pred)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_train, y_train_pred)
print('Recall score is:',recallScore)
# Printing classification report
print(classification_report(y_test, y_test_pred))
print(metrics.accuracy_score(y_test, y_test_pred))
# find precision score and recall score
precisionScore = precision_score(y_test, y_test_pred)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_test, y_test_pred)
print('Recall score is:',recallScore)
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# Confusion matrix 
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train, y_train_pred)
trainPrecision = precision_score(y_train, y_train_pred)
trainRecall = recall_score(y_train, y_train_pred)
testAccuracy = metrics.accuracy_score(y_test, y_test_pred)
testPrecision = precision_score(y_test, y_test_pred)
testRecall = recall_score(y_test, y_test_pred)

hyperparams = 'max_depth=4,n_estimators=200,max_features=10,min_samples_leaf=15,min_samples_split=90'
#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['Random_Forest_RandomOversampling_Tuned', hyperparams, 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]
dfAllModelMetrics.head()
import warnings
warnings.filterwarnings("ignore")


smotesampl = SMOTE(random_state=0)
X_train_smote, y_train_smote = smotesampl.fit_resample(X_train, y_train)
# Artificial minority samples and corresponding minority labels from SMOTE are appended
# below X_train and y_train respectively
# So to exclusively get the artificial minority samples from SMOTE, we do
X_train_smote_1 = X_train_smote[X_train.shape[0]:]

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]


plt.rcParams['figure.figsize'] = [20, 20]
fig = plt.figure()

plt.subplot(3, 1, 1)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_smote_1[:X_train_1.shape[0], 0], X_train_smote_1[:X_train_1.shape[0], 1],
            label='Artificial SMOTE Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], 0], X_train_0[:X_train_1.shape[0], 1], label='Actual Class-0 Examples')
plt.legend()
print('After OverSampling, the shape of train_X: {}'.format(X_train_smote.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_smote.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_smote==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_smote==0)))
#Creating dataframe from resampled data
X_train_smote_res=pd.DataFrame(data=X_train_smote[0:,0:],
                               index=[i for i in range(X_train_smote.shape[0])],
                               columns=X_train.columns)

y_train_smote_res=pd.DataFrame(data=y_train_smote,
                               index=[i for i in range(y_train_smote.shape[0])],
                               columns=['Class'])
#Taking backup of the data to use for different models
X_train_smote_res_bkup = X_train_smote_res.copy()
y_train_smote_res_bkup = y_train_smote_res.copy()
# Logistic regression model
logm1 = sm.GLM(y_train_smote_res,(sm.add_constant(X_train_smote_res)), family = sm.families.Binomial())
logm1.fit().summary()
X_train_smote_res = X_train_smote_res.drop(['V23'], axis = 1)
# Logistic regression model
logm1 = sm.GLM(y_train_smote_res,(sm.add_constant(X_train_smote_res)), family = sm.families.Binomial())
logm1.fit().summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_smote_res.columns
vif['VIF'] = [variance_inflation_factor(X_train_smote_res.values, i) for i in range(X_train_smote_res.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Using RFE for feature selection
logreg = LogisticRegression()
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train_smote_res, y_train_smote_res)
list(zip(X_train_smote_res.columns, rfe.support_, rfe.ranking_))
# Lets create a list with rfe support columns list
col = X_train_smote_res.columns[rfe.support_]
# Applying logistic regression on RFE supported features
X_train_smote_res = sm.add_constant(X_train_smote_res[col])
LR2 = sm.GLM(y_train_smote_res,X_train_smote_res, family = sm.families.Binomial())
res = LR2.fit()
res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_smote_res[col].columns
vif['VIF'] = [variance_inflation_factor(X_train_smote_res[col].values, i) for i in range(X_train_smote_res[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping one by one progressively and rechecking VIF and subsequently dropping
# X_train_smote_res = X_train_smote_res.drop(['V10','V17','V14','V12','const'], axis = 1)
X_train_smote_res = X_train_smote_res.drop(['V12'], axis = 1)
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_smote_res.columns
vif['VIF'] = [variance_inflation_factor(X_train_smote_res.values, i) for i in range(X_train_smote_res.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Applying logistic regression on RFE supported features
X_train_smote_res = sm.add_constant(X_train_smote_res)
logReg_Final = sm.GLM(y_train_smote_res,X_train_smote_res, family = sm.families.Binomial())
res = logReg_Final.fit()
res.summary()
# Lets predict churn on Train data
y_train_smote_res_pred = res.predict(X_train_smote_res).values.reshape(-1)
# Lets create a data frame with Original Class values and predicted Class probability values 
y_train_smote_res_pred_final = pd.DataFrame({'Class':y_train_smote_res['Class'], 'Class_Prob':y_train_smote_res_pred})
y_train_smote_res_pred_final.head()
# Precison - recall curve
p, r, thresholds = precision_recall_curve(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.Class_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.grid(b=None, which='major', axis='both')
plt.show()
# Lets see how the predicted va;lues turn out with 50% probability cut off
y_train_smote_res_pred_final['predicted'] = y_train_smote_res_pred_final.Class_Prob.map(lambda x: 1 if x > 0.39 else 0)

# Let's see the head
y_train_smote_res_pred_final.head()
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted))
# find precision score and recall score
precisionScore = precision_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
print('Recall score is:',recallScore)
# Lets find out F1 score
f1Score = 2 * (precisionScore * recallScore)/(precisionScore + recallScore)
print('F1 Score is:',f1Score)
# Creating a function to draw ROC curve
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC (AUC) = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ROC')
    plt.legend(loc="lower right")
    plt.grid(b=None, which='major', axis='both')
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve( y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.Class_Prob, drop_intermediate = False )

# Lets call the ROC plot fucnction on our model.
draw_roc(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.Class_Prob)
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's calculate some advanced metrics like sensitivity, specificity etc of our logistic regression model
print('Sensitivity or recall of the model is:',np.round(TP / float(TP+FN),2))
print('Specificity of the model is:',np.round(TN / float(TN+FP),2))
print('False Positive Rate for model is:',np.round(FP/ float(TN+FP),2))
print('Positive predictive value or precision is:',np.round(TP / float(TP+FP),2))
print('Negative predictive value is:',np.round(TN / float(TN+ FN),2))
X_test.columns
cols = X_train_smote_res.drop('const',axis=1).columns
X_test_lr = X_test[cols]
X_test_lr.head()
# lets predict churn on Test data
X_test_lr = sm.add_constant(X_test_lr)
y_test_smote_res_pred = res.predict(X_test_lr).values.reshape(-1)
y_test_smote_res_pred[:10]
# Creating a data frame with original churn values and predicted churn probability values
y_test_smote_res_pred_final = pd.DataFrame({'Class':y_test, 'Class_Prob':y_test_smote_res_pred})
y_test_smote_res_pred_final.head()
# We will take same 0,5 as probability cut off on test set as well as this is the probability we have choosen on train set as well
y_test_smote_res_pred_final['predicted'] = y_test_smote_res_pred_final.Class_Prob.map(lambda x: 1 if x > 0.41 else 0)

# Let's see the head
y_test_smote_res_pred_final.head()
# Let's check the overall accuracy.
print('Accuracy:',metrics.accuracy_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted))
# find precision score and recall score
precisionScore = precision_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)
print('Recall score is:',recallScore)
# Confusion matrix 
confusion = metrics.confusion_matrix(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's calculate some advanced metrics like sensitivity, specificity etc of our logistic regression model
print('Sensitivity or recall of the model is:',np.round(TP / float(TP+FN),2))
print('Specificity of the model is:',np.round(TN / float(TN+FP),2))
print('False Positive Rate for model is:',np.round(FP/ float(TN+FP),2))
print('Positive predictive value or precision is:',np.round(TP / float(TP+FP),2))
print('Negative predictive value is:',np.round(TN / float(TN+ FN),2))
# Lets call the ROC plot fucnction on our model.
draw_roc(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.Class_Prob)
# Creating a data frame to store metrics of all the models for easy comparision later 
dfAllModelMetrics = pd.DataFrame(columns=['ModelName','Hyperparams',
                                          'TestTN','TestFP','TestFN','TestTP',
                                          'TrainAccuracy','TrainPrecision','TrainRecall',
                                          'TestAccuracy','TestPrecision','TestRecall'])
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
trainPrecision = precision_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
trainRecall = recall_score(y_train_smote_res_pred_final.Class, y_train_smote_res_pred_final.predicted)
testAccuracy = metrics.accuracy_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)
testPrecision = precision_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)
testRecall = recall_score(y_test_smote_res_pred_final.Class, y_test_smote_res_pred_final.predicted)

#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['LogisticRegression_SMOTE', '', 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]
# Visualizing the metrics dataframe
dfAllModelMetrics
X_train_smote_res_rf = X_train_smote_res_bkup.copy()
X_test_smote_res_rf = X_test.copy()
y_train_smote_res_rf = y_train_smote_res_bkup.copy()
y_test_smote_res_rf = y_test.copy()
# Running the random forest with default parameters.
rfc = RandomForestClassifier(class_weight='balanced')

# fit
rfc.fit(X_train_smote_res_rf,y_train_smote_res_rf)

# Making predictions
y_train_pred = rfc.predict(X_train_smote_res_rf)
y_test_pred = rfc.predict(X_test_smote_res_rf)
# Printing classification report
print(classification_report(y_train_smote_res_rf, y_train_pred))
print(metrics.accuracy_score(y_train_smote_res_rf, y_train_pred))
# find precision score and recall score
precisionScore = precision_score(y_train_smote_res_rf, y_train_pred)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_train_smote_res_rf, y_train_pred)
print('Recall score is:',recallScore)
# Printing classification report
print(classification_report(y_test_smote_res_rf, y_test_pred))
print(metrics.accuracy_score(y_test_smote_res_rf, y_test_pred))
# find precision score and recall score
precisionScore = precision_score(y_test_smote_res_rf, y_test_pred)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_test_smote_res_rf, y_test_pred)
print('Recall score is:',recallScore)
# Confusion matrix 
confusion = metrics.confusion_matrix(y_test_smote_res_rf, y_test_pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
trainAccuracy = metrics.accuracy_score(y_train_smote_res_rf, y_train_pred)
trainPrecision = precision_score(y_train_smote_res_rf, y_train_pred)
trainRecall = recall_score(y_train_smote_res_rf, y_train_pred)
testAccuracy = metrics.accuracy_score(y_test_smote_res_rf, y_test_pred)
testPrecision = precision_score(y_test_smote_res_rf, y_test_pred)
testRecall = recall_score(y_test_smote_res_rf, y_test_pred)

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['RandomForest_SMOTE_Default', '', 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]
dfAllModelMetrics.head()
from imblearn.over_sampling import SMOTE
#Initialising the ADASYN object
smotesampl = SMOTE(sampling_strategy='minority',random_state=101)
#fit_resample on the train sets to produce the new resampled sets.
X_train_smote,y_train_smote = smotesampl.fit_resample(X_train,y_train)
X_train_smote.shape
y_train_smote.shape
X_train_smote = pd.DataFrame(X_train_smote,columns=X_train.columns)
y_train_smote = pd.DataFrame(y_train_smote,columns=['Fraud'])
#Data Inbalance check for the Converted column.
yes=y_train_smote[y_train_smote['Fraud']==1]['Fraud'].value_counts()
no=y_train_smote[y_train_smote['Fraud']==0]['Fraud'].value_counts()

converted=np.array((yes/len(y_train_smote))*100) 
not_converted=np.array((no/len(y_train_smote))*100) 
stat_summ=pd.DataFrame({'Percentage':[converted[0],not_converted[0]]},index=['Fraud','Not_Fraud'])
plt.figure(figsize=(5,5))
sns.barplot(x=stat_summ.index,y=stat_summ['Percentage'],palette='RdYlGn')
plt.title('Check on the data imbalance')
plt.show()
X_train_smote.shape
y_train_smote.shape
X_train_smote_new = X_train_smote.copy()
X_train_smote_new['Fraud'] = y_train_smote['Fraud']
X_train_smote_fraud = X_train_smote_new.loc[X_train_smote_new['Fraud']==1]
X_train_smote_non_fraud = X_train_smote_new.loc[X_train_smote_new['Fraud']==0]
X_train_smote_fraud.shape
X_train_smote_non_fraud.shape
X_train_smote_fraud = X_train_smote_fraud.sample(frac=1)
X_train_smote_non_fraud = X_train_smote_non_fraud.sample(frac=1)
X_train_smote_fraud = X_train_smote_fraud[:50000]
X_train_smote_non_fraud = X_train_smote_non_fraud[:50000]
frames = [X_train_smote_fraud,X_train_smote_non_fraud]
X_train_smote_final = pd.concat(frames)
X_train_smote_final = X_train_smote_final.sample(frac=1)
X_train_smote_final
X_train_smote_sampled = X_train_smote_final.drop(['Fraud'],axis=1)
y_train_smote_sampled = X_train_smote_final['Fraud']
from sklearn.ensemble import RandomForestClassifier
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 12, 3)}
# instantiate the model
rf = RandomForestClassifier()
# fit tree on training data
rf = GridSearchCV(rf, parameters,cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_depth
plt.figure(figsize=(5,5))
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training recall")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(200, 900, 200)}
# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=8)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds,scoring="recall",return_train_score=True)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with n_estimators
plt.figure(figsize=(5,5))
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="test accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}
# instantiate the model
rf = RandomForestClassifier(max_depth=8,
                            n_estimators=200)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_features
plt.figure(figsize=(5,5))
plt.plot(scores["param_max_features"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_features"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_features")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_leaf': range(50,300,50)}
# instantiate the model
rf = RandomForestClassifier(max_depth=8,
                            n_estimators=200,
                            max_features=14)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_leaf
plt.figure(figsize=(5,5))
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'min_samples_split': range(100, 300, 50)}
# instantiate the model
rf = RandomForestClassifier(max_depth=8,
                            n_estimators=200,
                            max_features=14,
                            min_samples_leaf=150)
# fit tree on training data
rf = GridSearchCV(rf, parameters, cv=n_folds, scoring="recall",return_train_score=True)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
 # scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with min_samples_split
plt.figure(figsize=(5,5))
plt.plot(scores["param_min_samples_split"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_split"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_split")
plt.ylabel("Recall")
plt.legend()
plt.show()
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=8,
                            max_features=14,
                            min_samples_leaf=150,
                            min_samples_split=200)
rf.fit(X_train_smote_sampled,y_train_smote_sampled)
y_train_pred_rf_smote = rf.predict(X_train_smote_sampled)
accuracy_score(y_train_smote_sampled,y_train_pred_rf_smote)
# Printing classification report
print(classification_report(y_train_smote_sampled, y_train_pred_rf_smote))
print(metrics.accuracy_score(y_train_smote_sampled, y_train_pred_rf_smote))
# find precision score and recall score
precisionScore = precision_score(y_train_smote_sampled, y_train_pred_rf_smote)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_train_smote_sampled, y_train_pred_rf_smote)
print('Recall score is:',recallScore)
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_smote_sampled,y_train_pred_rf_smote)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
confusion
X_test.shape
y_test.shape
y_test_pred_rf_smote = rf.predict(X_test)
# Printing classification report
print(classification_report(y_test, y_test_pred_rf_smote))
print(metrics.accuracy_score(y_test, y_test_pred_rf_smote))
# find precision score and recall score
precisionScore = precision_score(y_test, y_test_pred_rf_smote)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_test, y_test_pred_rf_smote)
print('Recall score is:',recallScore)
# Confusion matrix 
confusion_rf_smote = metrics.confusion_matrix(y_test, y_test_pred_rf_smote)
print(confusion_rf_smote)

plt.figure(figsize=(3,2))
sns.heatmap(confusion_rf_smote, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion_rf_smote[1,1] # true positive 
TN = confusion_rf_smote[0,0] # true negatives
FP = confusion_rf_smote[0,1] # false positives
FN = confusion_rf_smote[1,0] # false negatives
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP+FP)
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train_smote_sampled, y_train_pred_rf_smote)
trainPrecision = precision_score(y_train_smote_sampled, y_train_pred_rf_smote)
trainRecall = recall_score(y_train_smote_sampled, y_train_pred_rf_smote)
testAccuracy = metrics.accuracy_score(y_test, y_test_pred_rf_smote)
testPrecision = precision_score(y_test, y_test_pred_rf_smote)
testRecall = recall_score(y_test, y_test_pred_rf_smote)

params = 'n_estimators=200,max_depth=8,max_features=14,min_samples_leaf=150,min_samples_split=200'
#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['Random_Forest_SMOTE_Tuned', params, 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]

dfAllModelMetrics.head()
X_train_smote_sampled.shape
y_train_smote_sampled.shape
#Importing the libraries for XGBoost.
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import time
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'max_depth': range(2, 20, 4)}
# instantiate the model
xgb_model = XGBClassifier()
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with max_depth
plt.figure(figsize=(5,5))
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_train_score"], 
         label="training recall")
plt.plot(cv_results["param_max_depth"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.xlabel("max_depth")
plt.ylabel("recall")
plt.legend()
plt.show()
start = time.time()
# specify number of folds for k-fold CV
n_folds = 3
# parameters to build the model on
parameters = {'n_estimators': range(100, 900, 200)}
# instantiate the model (note we are specifying a max_depth)
xgb_model = XGBClassifier(max_depth=6)
# fit tree on training data
xgb_model1 = GridSearchCV(xgb_model, parameters, cv=n_folds, scoring="recall",return_train_score=True)
xgb_model1.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
cv_results = pd.DataFrame(xgb_model1.cv_results_)
cv_results
# plotting accuracies with n_estimators
plt.figure(figsize=(5,5))
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_test_score"], 
         label="test recall")
plt.plot(cv_results["param_n_estimators"], 
         cv_results["mean_train_score"], 
         label="train recall")
plt.xlabel("n_estimators")
plt.ylabel("Recall")
plt.legend()
plt.show()
start = time.time()
# hyperparameter tuning with XGBoost
# creating a KFold object 
folds = 3
# specify range of hyperparameters
param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=6,
                          n_estimators=300)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'recall', 
                        cv = folds,  
                        verbose = 1,
                        return_train_score=True)  
model_cv.fit(X_train_smote_sampled,y_train_smote_sampled)
end = time.time()
print("Amount of time taken for the above query in minutes:",round((end-start)/60 ,2))
# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results
# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')
cv_results.head()
cv_results['mean_test_score'].value_counts()
cv_results['mean_train_score'].value_counts()
#plotting
plt.figure(figsize=(16,6))
param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4], 
             'subsample': [0.3, 0.6, 0.9]} 
for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('recall')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')
# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate':0.1,
          'max_depth':6, 
          'n_estimators':300,
          'subsample':0.3,
         'objective':'binary:logistic'}

# fit model on training data
model = XGBClassifier(params = params)
model.fit(X_train_smote_sampled,y_train_smote_sampled)
# predict
y_pred = model.predict_proba(X_train_smote_sampled)
y_pred[:10]
import sklearn
# roc_auc
auc = sklearn.metrics.roc_auc_score(y_train_smote_sampled, y_pred[:, 1])
auc
xgb_df = pd.DataFrame(y_pred[:, 1],columns=["Predicted_probability"])
xgb_df['Churn'] = y_train_smote_sampled.values
xgb_df
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    xgb_df[i]= xgb_df.Predicted_probability.map(lambda x: 1 if x > i else 0)
xgb_df.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
plt.figure(figsize=(5,5))
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])
plt.show()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df1 = pd.DataFrame( columns = ['probability','precision','recall'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = confusion_matrix(xgb_df.Churn, xgb_df[i] )
    total1=sum(sum(cm1))
    prec = cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df1.loc[i] =[ i ,prec, rec]
print(cutoff_df1)
# Let's plot accuracy sensitivity and specificity for various probabilities.
plt.figure(figsize=(5,5))
cutoff_df1.plot.line(x='probability', y=['precision','recall'])
plt.show()
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_df['Final_predicted'] = xgb_df.Predicted_probability.map( lambda x: 1 if x > 0.4 else 0)
xgb_df.head()
y_pred_test = model.predict_proba(X_test)
y_pred_test[:10]
xgb_test_df = pd.DataFrame(y_pred_test[:, 1],columns=["Predicted_probability"])
xgb_test_df['y_test'] = y_test.values
xgb_test_df
#Selecting the cut-off probability as 0.4 and predicting for Churn.
xgb_test_df['Final_predicted'] = xgb_test_df.Predicted_probability.map( lambda x: 1 if x > 0.4 else 0)
xgb_test_df.head()
# Printing classification report
print(classification_report(y_test, xgb_test_df.Final_predicted))
print(metrics.accuracy_score(y_test, xgb_test_df.Final_predicted))
# find precision score and recall score
precisionScore = precision_score(y_test, xgb_test_df.Final_predicted)
print('Precision score is:',precisionScore)
recallScore = recall_score(y_test, xgb_test_df.Final_predicted)
print('Recall score is:',recallScore)
confusion_xgb = confusion_matrix(xgb_test_df['y_test'],xgb_test_df['Final_predicted'])
confusion_xgb
TP = confusion_xgb[1,1] # true positive 
TN = confusion_xgb[0,0] # true negatives
FP = confusion_xgb[0,1] # false positives
FN = confusion_xgb[1,0] # false negatives
# Confusion matrix 
confusion_xgb = confusion_matrix(xgb_test_df['y_test'],xgb_test_df['Final_predicted'])
print(confusion_xgb)

plt.figure(figsize=(3,2))
sns.heatmap(confusion_xgb, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
#Sensitivity / Recall
TP / float(TP + FN)
#Specificity
TN / float(TN + FP)
#Precision
TP / float(TP + FP)
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train_smote_sampled, y_train_pred_rf_smote)
trainPrecision = precision_score(y_train_smote_sampled, y_train_pred_rf_smote)
trainRecall = recall_score(y_train_smote_sampled, y_train_pred_rf_smote)
testAccuracy = metrics.accuracy_score(y_test, y_test_pred_rf_smote)
testPrecision = precision_score(y_test, y_test_pred_rf_smote)
testRecall = recall_score(y_test, y_test_pred_rf_smote)

params = 'n_estimators=200,max_depth=8,max_features=14,min_samples_leaf=150,min_samples_split=200'
#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['Random_Forest_SMOTE_Tuned', params, 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]

dfAllModelMetrics.head()
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning':0})

from sklearn.model_selection import StratifiedKFold

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN

import lightgbm as lgb

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


from sklearn.metrics import roc_auc_score,classification_report

import warnings
warnings.filterwarnings('ignore')

!pip install bayesian-optimization
from bayes_opt import BayesianOptimization
y_train_df = pd.DataFrame(y_train)
# Scaling weight = Ratio of number of 0's to 1's so it can scale the weight of 1's
scale_pos_weight_factor = y_train[y_train_df.Class==0].shape[0] / y_train[y_train_df.Class==1].shape[0]
def xgb_best_params(X,y,opt_params,init_points=10, optimization_round=5, n_folds=3, random_seed=0, cv_estimators=600): 
    # prepare dataset
    training_data = xgb.DMatrix(X, y)
    
    def xgb_run(learning_rate,max_depth,min_child_weight,gamma,subsample,colsample_bytree,reg_alpha,reg_lambda):
        params = {'objective':'binary:logistic','n_estimators':cv_estimators,'scale_pos_weight':scale_pos_weight_factor ,'early_stopping_round':int(cv_estimators/20), 'metric':'auc'}
        params["gamma"] = gamma
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = reg_alpha
        params['reg_lambda'] = reg_lambda
        params['min_child_weight'] = int(min_child_weight)
        params['learning_rate'] = learning_rate
        cv_result = xgb.cv(params, training_data, nfold=n_folds, seed=random_seed ,stratified=True,shuffle=True ,verbose_eval =int(cv_estimators/20),num_boost_round=cv_estimators, metrics='auc')

        return cv_result['test-auc-mean'].max()
    
    
    params_finder = BayesianOptimization(xgb_run, opt_params, random_state=100)
    # optimize
    params_finder.maximize(init_points=init_points, n_iter=optimization_round)

    # return best parameters
    return params_finder.max
folds = 3

bounds = {
    'learning_rate': (0.002, 0.2),
    'max_depth':(1,20),
    'min_child_weight':(1,100),
    'gamma':(0,1),
    'subsample':(0.1,1),
    'colsample_bytree':(0.1,0.8),
    'reg_alpha':(0.1,20),
    'reg_lambda':(0.1,20)
}
#a = []
#while(1):
#    a.append('1000000')
best_params= []
cv_estimators = [500]
optimization_round = 10
init_points = 10
random_seed = 0
    
    
for cv_estimator in cv_estimators:
    opt_params = xgb_best_params(X, y,bounds, init_points=init_points, optimization_round=optimization_round, n_folds=folds, random_seed=random_seed, cv_estimators=cv_estimator)
    opt_params['params']['iteration'] = cv_estimator
    opt_params['params']['fold'] = folds
    opt_params['params']['auc'] = opt_params['target']
    best_params.append(opt_params['params'])

max_auc = 0
max_auc_index = -1
for idx in range(0,len(best_params)):
  if best_params[idx]['auc'] > max_auc:
    max_auc = best_params[idx]['auc']
    max_auc_index = idx

xgb_best_params = best_params[max_auc_index]

print('***** PARAMETERS WITH TOP AUC SCORE *****')
for key in xgb_best_params:
  print(key,':',xgb_best_params[key])
xgb_best_params
xgb_tuned_params = xgb_best_params.copy()
del xgb_tuned_params['auc']
xgb_tuned_params['metric'] = 'auc'
xgb_tuned_params['max_depth'] = int(xgb_tuned_params['max_depth'])
xgb_tuned_params['min_child_weight'] = int(xgb_tuned_params['min_child_weight'])
training_data = xgb.DMatrix(X_train, y_train)
xgb_tuned_model = xgb.train(xgb_tuned_params,training_data)
xgb_train_data = xgb.DMatrix(X_train[X.columns])
y_train_pred_bo_xgb = xgb_tuned_model.predict(xgb_train_data)
xgb_test_data = xgb.DMatrix(X_test[X.columns])
y_test_pred_bo_xgb = xgb_tuned_model.predict(xgb_test_data)
print(roc_auc_score(y_train,y_train_pred_bo_xgb))
print(roc_auc_score(y_test,y_test_pred_bo_xgb))
# Lets create a data frame with Original Class values and predicted Class probability values 
y_train_pred_bo_xgb_final = pd.DataFrame({'Class':y_train, 'Class_Prob':y_train_pred_bo_xgb})
y_test_pred_bo_xgb_final = pd.DataFrame({'Class':y_test, 'Class_Prob':y_test_pred_bo_xgb})
y_train_pred_bo_xgb_final.head()
# Precison - recall curve
p, r, thresholds = precision_recall_curve(y_train_pred_bo_xgb_final.Class, y_train_pred_bo_xgb_final.Class_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.grid(b=None, which='major', axis='both')
plt.show()
# Lets see how the predicted values turn out with 42% probability cut off
y_train_pred_bo_xgb_final['Class_Pred'] = y_train_pred_bo_xgb_final.Class_Prob.map(lambda x: 1 if x > 0.42 else 0)
y_test_pred_bo_xgb_final['Class_Pred'] = y_test_pred_bo_xgb_final.Class_Prob.map(lambda x: 1 if x > 0.42 else 0)
# Confusion matrix for training
confusion = metrics.confusion_matrix(y_train_pred_bo_xgb_final.Class, y_train_pred_bo_xgb_final.Class_Pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# Confusion matrix for test
confusion = metrics.confusion_matrix(y_test_pred_bo_xgb_final.Class, y_test_pred_bo_xgb_final.Class_Pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train_pred_bo_xgb_final.Class, y_train_pred_bo_xgb_final.Class_Pred)
trainPrecision = precision_score(y_train_pred_bo_xgb_final.Class, y_train_pred_bo_xgb_final.Class_Pred)
trainRecall = recall_score(y_train_pred_bo_xgb_final.Class, y_train_pred_bo_xgb_final.Class_Pred)
testAccuracy = metrics.accuracy_score(y_test_pred_bo_xgb_final.Class, y_test_pred_bo_xgb_final.Class_Pred)
testPrecision = precision_score(y_test_pred_bo_xgb_final.Class, y_test_pred_bo_xgb_final.Class_Pred)
testRecall = recall_score(y_test_pred_bo_xgb_final.Class, y_test_pred_bo_xgb_final.Class_Pred)

#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['XGB_Bayesian_Optimization_Tuned', str(xgb_tuned_params), 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]
dfAllModelMetrics.head()
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning':0})
from sklearn.model_selection import StratifiedKFold

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from imblearn.combine import SMOTETomek,SMOTEENN
import lightgbm as lgb

#import xgboost as xgb
#from xgboost.sklearn import XGBClassifier


from sklearn.metrics import roc_auc_score,classification_report

import warnings
warnings.filterwarnings('ignore')

#!pip install bayesian-optimization
from bayes_opt import BayesianOptimization
import lightgbm as lgb
y_train_df = pd.DataFrame(y_train)
# Scaling weight = Ratio of number of 0's to 1's so it can scale the weight of 1's
scale_pos_weight_factor = y_train[y_train_df.Class==0].shape[0] / y_train[y_train_df.Class==1].shape[0]
print('scale_pos_weight_factor value:',scale_pos_weight_factor)
def lightgbm_best_params(X, y,opt_params,init_points=10, optimization_round=5, n_folds=3, random_seed=0, cv_estimators=500): 
    # prepare dataset
    training_data = lgb.Dataset(X_train, y_train)
    
    def lightgbm_run(max_depth,
                     num_leaves,
                     #colsample_bytree,
                     reg_alpha,  #also called lambdal1
                     reg_lambda, # also called lambdal2
                     min_child_samples, #Minimum number of data needed in a child (leaf).
                     learning_rate,
                     feature_fraction
                     ): 
        params = {'objective':'binary',
                  'n_estimators':cv_estimators, #Number of boosted trees to fit.
                  'early_stopping_round':int(cv_estimators/20),
                  'metric':'auc',
                  'subsample_freq':5, #Frequence of subsample - Need to understand more (<=0 means no enable.)
                  'bagging_seed':42,
                  'verbosity':-1,
                  'num_threads':20,
                  #Not using class_weight as per documentation; since class_weight is for multi class classification. For binary classification
                  #can use is_unbalance = True OR scale_pos_weight parameter. Using is_unbalance
                  'is_unbalance':True 
                  }
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        #params['colsample_bytree'] = colsample_bytree
        params['reg_alpha'] = reg_alpha
        params['reg_lambda'] = reg_lambda
        params['min_child_samples'] = int(min_child_samples) 
        # params['min_data_in_leaf'] = int(min_data_in_leaf)
        params['learning_rate'] = learning_rate
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        cv_result = lgb.cv(params,
                           training_data,
                           nfold=n_folds,
                           seed=random_seed,
                           stratified=True,
                           metrics=['auc'])

        print('cv_result',cv_result)
        return max(cv_result['auc-mean'])
    
    
    params_finder = BayesianOptimization(lightgbm_run, opt_params, random_state=100)
    # optimize
    params_finder.maximize(init_points=init_points, n_iter=optimization_round)

    # return best parameters
    return params_finder.max
bounds = {
    'max_depth':(3,40),
    'num_leaves':(25, 4000), #Maximum tree leaves for base learners.
    #'colsample_bytree':(0.1,0.8),
    'reg_alpha':(0.1,20),
    'reg_lambda':(0.1,20),
    'min_child_samples':(50, 4000),
    'learning_rate': (0.002, 0.2),
    'feature_fraction':(0.1,1) #Alias colsample_bytree, sub_feature -> helps reduce overfitting and increase speed, let me use only 10 to 80% of features
}
best_params= []
cv_estimators = [200,400,600,800,1000]
optimization_round = 10
init_points = 10
random_seed = 0
folds = 3
for cv_estimator in cv_estimators:
    opt_params = lightgbm_best_params(X_train,
                                 y_train,
                                 bounds,
                                 init_points=init_points,
                                 optimization_round=optimization_round,
                                 n_folds=folds,
                                 random_seed=random_seed,
                                 cv_estimators=cv_estimator)
    print(opt_params)
    opt_params['params']['iteration'] = cv_estimator
    opt_params['params']['fold'] = folds
    opt_params['params']['auc'] = opt_params['target']
    best_params.append(opt_params['params'])
    print('best_params as of now:',best_params)
max_auc = 0
max_auc_index = -1
for idx in range(0,len(best_params)):
  if best_params[idx]['auc'] > max_auc:
    max_auc = best_params[idx]['auc']
    max_auc_index = idx

gbm_best_params = best_params[max_auc_index]

print('***** PARAMETERS WITH TOP AUC SCORE *****')
for key in gbm_best_params:
  print(key,':',gbm_best_params[key])
gbm_tuned_params = gbm_best_params.copy()
del gbm_tuned_params['auc']
gbm_tuned_params['metric'] = 'auc'
gbm_tuned_params['max_depth'] = int(gbm_tuned_params['max_depth'])
gbm_tuned_params['num_leaves'] = int(gbm_tuned_params['num_leaves'])
gbm_tuned_params['min_child_samples'] = int(gbm_tuned_params['min_child_samples'])
training_data = lgb.Dataset(X_train, y_train)
gbm_tuned_model = lgb.train(gbm_tuned_params,training_data)
y_train_pred_bo_gbm = gbm_tuned_model.predict(X_train)
y_test_pred_bo_gbm = gbm_tuned_model.predict(X_test)
print(roc_auc_score(y_train,y_train_pred_bo_gbm))
print(roc_auc_score(y_test,y_test_pred_bo_gbm))
# Lets create a data frame with Original Class values and predicted Class probability values 
y_train_pred_bo_gbm_final = pd.DataFrame({'Class':y_train, 'Class_Prob':y_train_pred_bo_gbm})
y_test_pred_bo_gbm_final = pd.DataFrame({'Class':y_test, 'Class_Prob':y_test_pred_bo_gbm})
y_train_pred_bo_gbm_final.head()
# Precison - recall curve
p, r, thresholds = precision_recall_curve(y_train_pred_bo_gbm_final.Class, y_train_pred_bo_gbm_final.Class_Prob)
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.grid(b=None, which='major', axis='both')
plt.show()
# Lets see how the predicted values turn out with 2% probability cut off (0.02)
y_train_pred_bo_gbm_final['Class_Pred'] = y_train_pred_bo_gbm_final.Class_Prob.map(lambda x: 1 if x > 0.02 else 0)
y_test_pred_bo_gbm_final['Class_Pred'] = y_test_pred_bo_gbm_final.Class_Prob.map(lambda x: 1 if x > 0.02 else 0)
# Confusion matrix for training
confusion = metrics.confusion_matrix(y_train_pred_bo_gbm_final.Class, y_train_pred_bo_gbm_final.Class_Pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
# Confusion matrix for test
confusion = metrics.confusion_matrix(y_test_pred_bo_gbm_final.Class, y_test_pred_bo_gbm_final.Class_Pred)
print(confusion)

plt.figure(figsize=(3,2))
sns.heatmap(confusion, annot=True, xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],fmt='.0f')
plt.ylabel('True label')
plt.xlabel('Predicted label')
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Storing all metics in the model metrics data frame
trainAccuracy = metrics.accuracy_score(y_train_pred_bo_gbm_final.Class, y_train_pred_bo_gbm_final.Class_Pred)
trainPrecision = precision_score(y_train_pred_bo_gbm_final.Class, y_train_pred_bo_gbm_final.Class_Pred)
trainRecall = recall_score(y_train_pred_bo_gbm_final.Class, y_train_pred_bo_gbm_final.Class_Pred)
testAccuracy = metrics.accuracy_score(y_test_pred_bo_gbm_final.Class, y_test_pred_bo_gbm_final.Class_Pred)
testPrecision = precision_score(y_test_pred_bo_gbm_final.Class, y_test_pred_bo_gbm_final.Class_Pred)
testRecall = recall_score(y_test_pred_bo_gbm_final.Class, y_test_pred_bo_gbm_final.Class_Pred)

#Adding metrics into the dataframe
dfAllModelMetrics.loc[dfAllModelMetrics.shape[0]] = ['LightGBM_Bayesian_Optimization_Tuned', str(gbm_tuned_params), 
                TN, FP, FN, TP,
                trainAccuracy,trainPrecision,trainRecall,
                testAccuracy,testPrecision,testRecall]
dfAllModelMetrics.head()
xgb_test_df_ca.head()
X_test_copy.head()
cost_analysis_df = xgb_test_df_ca.copy()
cost_analysis_df['Amount'] = X_test_copy['Amount'].values
#Final cost analsys dataframe after combining the df having y_test and pred values with the Amount column.
cost_analysis_df.head()
#We can drop the probability column which is not needed.
cost_analysis_df.drop(['Predicted_probability'],axis=1,inplace=True)
cost_analysis_df.head()
confusion_cost_analysis = confusion_matrix(cost_analysis_df['y_test'],cost_analysis_df['Final_predicted'])
confusion_cost_analysis
TP = confusion_cost_analysis[1,1] # true positive 
TN = confusion_cost_analysis[0,0] # true negatives
FP = confusion_cost_analysis[0,1] # false positives
FN = confusion_cost_analysis[1,0] # false negatives
Total_fraud_predictions = TP + FP
Total_fraud_predictions
def confusion_col(s):
    if(s['y_test']==1 and s['Final_predicted']==1):
        return 'TP'
    elif(s['y_test']==0 and s['Final_predicted']==0):
        return 'TN'
    if(s['y_test']==0 and s['Final_predicted']==1):
        return 'FP'
    elif(s['y_test']==1 and s['Final_predicted']==0):
        return 'FN'
cost_analysis_df['Confusion_label'] = cost_analysis_df.apply(confusion_col,axis=1)
cost_analysis_df.head()
total_amnt_crct_pred = cost_analysis_df.loc[(cost_analysis_df['Confusion_label']=='TP'),'Amount'].sum()
total_amnt_crct_pred
total_amnt_incrct_pred = cost_analysis_df.loc[(cost_analysis_df['Confusion_label']=='FN'),'Amount'].sum()
total_amnt_incrct_pred
Total_savings = round(total_amnt_crct_pred - ((Total_fraud_predictions * 10 ) + total_amnt_incrct_pred),3)
print(Total_savings)
