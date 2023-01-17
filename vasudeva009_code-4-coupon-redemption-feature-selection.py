# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn.preprocessing import LabelEncoder

from scipy.stats import mode

import pandas_profiling

from sklearn.decomposition import PCA



from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold

from sklearn.metrics import classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier

from sklearn.metrics import auc



import os

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_colwidth',500)

pd.set_option('display.max_columns',5000)



encoder = LabelEncoder()

from IPython.display import Image

import os

!ls ../input/
train = pd.read_csv('../input/train.csv')

campaign = pd.read_csv('../input/campaign_data.csv')

items = pd.read_csv('../input/item_data.csv')

coupons = pd.read_csv('../input/coupon_item_mapping.csv')

cust_demo = pd.read_csv('../input/customer_demographics.csv')

cust_tran = pd.read_csv('../input/customer_transaction_data.csv')

test = pd.read_csv('../input/test.csv')

df=pd.read_csv('../input/final_train.csv')

data=pd.read_csv('../input/smote.csv')
train.shape, campaign.shape, items.shape, coupons.shape, cust_demo.shape, cust_tran.shape, test.shape,df.shape,data.shape
print('Train Dataframe')

print(train.isnull().sum())

print('======================')

print('Campaign Dataframe')

print(campaign.isnull().sum())

print('======================')

print('Items Dataframe')

print(items.isnull().sum())

print('======================')

print('Coupons Dataframe')

print(coupons.isnull().sum())

print('======================')

print('Customer Demographics Dataframe')

print(cust_demo.isnull().sum())

print('======================')

print('Customer Transaction Dataframe')

print(cust_tran.isnull().sum())

print('======================')



print(test.isnull().sum())

print('======================')

print(df.isnull().sum())

print('======================')

print(data.isnull().sum())
Image("../input/Schema.png")
df.shape
df.columns
df.drop(columns=['Unnamed: 0'],inplace=True)

df.head()
df.shape
from sklearn.feature_selection import RFE

X = df.drop(['redemption_status'], axis = 1)

y = df['redemption_status']
from sklearn.linear_model import LinearRegression

model = LinearRegression()

rfe = RFE(model,41)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)
from sklearn.metrics import roc_auc_score
# no of features

nof_list = np.arange(1,41)

high_score = 0

#Variable to store the optimum feature

nof = 0

score_list=[]

for n in range(len(nof_list)):

    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=0)

    model = LinearRegression()

    rfe = RFE(model,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,Y_train)

    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe,Y_train)

    Y_pred = model.predict(X_test_rfe)

    score = roc_auc_score(Y_test,Y_pred)

    score_list.append(score)

    if(score>high_score):

        high_score=score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" %(nof, high_score))
cols = list(X.columns)

model=LinearRegression()

#Initializing RFE Model

rfe = RFE(model,32)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)

#Fitting data to model

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)
col_names=['coupon_id', 'brand', 'brand_type', 'category', 'cd_sum','coupon_discount_x', 'coupon_used_x', 'item_counts', 'no_of_customers',

   'od_sum', 'other_discount_x', 'quantity_x', 'selling_price_x','total_discount_mean',

   'total_discount_sum', 'campaign_type','campaign_duration',

   'age_range', 'marital_status', 'rented','family_size',

   'no_of_children', 'income_bracket', 'coupon_discount_y','coupon_used_y','day', 'dow',

   'month', 'other_discount_y','quantity_y', 'selling_price_y', 'cdd_sum']
x=df[col_names]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1990)
LR = LogisticRegression()

LR.fit(x_train,y_train)

y_pred_LR = LR.predict(x_test)

print(classification_report(y_test,y_pred_LR))
print(roc_auc_score(y_test,y_pred_LR))

Model = ['Logistic Regression']

ROC_AUC_Accuracy = [roc_auc_score(y_test,y_pred_LR)]
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

results=confusion_matrix(y_test,y_pred_LR)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_LR) )

print ('Report : ')

print (classification_report(y_test,y_pred_LR) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_LR ):

    cm = metrics.confusion_matrix( y_test,y_pred_LR )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_LR )



print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_LR):

    if (len(y_test.shape) != len(y_pred_LR.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_LR.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_LR)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
    conf_mat = create_conf_mat(y_test,y_pred_LR)

    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

    plt.xlabel('Predicted Values')

    plt.ylabel('Actual Values')

    plt.title('Actual vs. Predicted Confusion Matrix')

    plt.show()
#Get predicted probabilites

target_probailities_log = LR.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)

#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)

print(roc_auc_score(y_test,y_pred_nb))

Model.append('Naive Bayes')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
results=confusion_matrix(y_test,y_pred_nb)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_nb) )

print ('Report : ')

print (classification_report(y_test,y_pred_nb) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_nb ):

    cm = metrics.confusion_matrix( y_test,y_pred_nb )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_nb )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_nb):

    if (len(y_test.shape) != len(y_pred_nb.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_nb.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_nb)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_nb)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = nb.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()

Model,ROC_AUC_Accuracy
params = {

    

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':range(1,10),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



dt = DecisionTreeClassifier()



rs = RandomizedSearchCV(estimator=dt,n_jobs=-1,cv=3,param_distributions=params,scoring='recall')

rs.fit(x,y)
dt = DecisionTreeClassifier(**rs.best_params_)

dt.fit(x_train,y_train)

y_pred_dt = dt.predict(x_test)

print(roc_auc_score(y_test,y_pred_dt))

Model.append('Decision Tree')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_dt):

    cm = metrics.confusion_matrix( y_test,y_pred_dt )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
results=confusion_matrix(y_test,y_pred_dt)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_dt) )

print ('Report : ')

print (classification_report(y_test,y_pred_dt) )
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_dt )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_dt):

    if (len(y_test.shape) != len(y_pred_dt.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_dt.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_dt)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_dt)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = dt.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()

Model,ROC_AUC_Accuracy
params = {

    

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

    'max_depth':range(2,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



rf = RandomForestClassifier()



rs = RandomizedSearchCV(estimator=rf,param_distributions=params,cv=3,scoring='recall',n_jobs=-1)

rs.fit(x,y)
rf = RandomForestClassifier(**rs.best_params_)

rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print(roc_auc_score(y_test,y_pred_rf))

Model.append('Random Forest')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
results=confusion_matrix(y_test,y_pred_rf)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_rf) )

print ('Report : ')

print (classification_report(y_test,y_pred_rf) )

#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_rf ):

    cm = metrics.confusion_matrix( y_test,y_pred_rf )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_rf )

print("confusion matrix = \n",mat_pruned)

def create_conf_mat(y_test,y_pred_rf):

    if (len(y_test.shape) != len(y_pred_rf.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_rf.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_rf)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_rf)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = rf.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)

#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
Model,ROC_AUC_Accuracy
final_result = pd.DataFrame({'Model':Model,'Accuracy':ROC_AUC_Accuracy})

final_result
LR_Bag = BaggingClassifier(base_estimator=LR,n_estimators=100,n_jobs=-1,random_state=1)

nb_Bag = BaggingClassifier(base_estimator=nb,n_estimators=100,n_jobs=-1,random_state=1)

dt_Bag = BaggingClassifier(base_estimator=dt,n_estimators=100,n_jobs=-1,random_state=1)
x = np.array(x)
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = LR_Bag

name = 'Bagged-LR'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Bag

name = 'Bagged-NB'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = dt_Bag

name = 'Bagged-DT'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMModel,LGBMClassifier
LR_Boost = AdaBoostClassifier(base_estimator=LR,n_estimators=100,learning_rate=0.01,random_state=1)

nb_Boost = AdaBoostClassifier(base_estimator=nb,n_estimators=100,learning_rate=0.01,random_state=1)

dt_Boost = AdaBoostClassifier(base_estimator=dt,n_estimators=100,learning_rate=0.01,random_state=1)

rf_Boost = AdaBoostClassifier(base_estimator=rf,n_estimators=100,learning_rate=0.01,random_state=1)

gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

lgbm = LGBMClassifier(objective='binary',n_estimators=100,reg_alpha=2,reg_lambda=5,random_state=1,learning_rate=0.01,is_unbalance=True)
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = LR_Boost

name = 'Boosted-LR'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Boost

name = 'Boosted-NB'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = dt_Boost

name = 'Boosted-DT'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = lgbm

name = 'LGBM'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
final_result = pd.DataFrame({'Model - RFE Data':Model,'Accuracy':ROC_AUC_Accuracy})

final_result
from statsmodels.stats.outliers_influence import variance_inflation_factor



[variance_inflation_factor(X.values, j) for j in range(1, X.shape[1])]
# function definition



def calculate_vif(X):

    thresh = 5.0

    output = pd.DataFrame()

    k = X.shape[1]

    vif = [variance_inflation_factor(X.values, j) for j in range(X.shape[1])]

    for i in range(1,k):

        print("Iteration no.",i)

        print(vif)

        a = np.argmax(vif)

        print("Max VIF is for variable no.:",a)

        if vif[a] <= thresh :

            break

        if i == 1 :          

            output = X.drop(X.columns[a], axis = 1)

            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

        elif i > 1 :

            output = output.drop(output.columns[a],axis = 1)

            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

    return(output)
train_out = calculate_vif(X)
## includes only the relevant features.

train_out.head()
train_out.columns
train_out.shape
x_train,x_test,y_train,y_test = train_test_split(train_out,y,test_size=0.3,random_state=1990)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
LR = LogisticRegression()

LR.fit(x_train,y_train)

y_pred_LR = LR.predict(x_test)

print(classification_report(y_test,y_pred_LR))
roc_auc_score(y_test,y_pred_LR)

Model = ['Logistic Regression']

ROC_AUC_Accuracy = [roc_auc_score(y_test,y_pred_LR)]
results=confusion_matrix(y_test,y_pred_LR)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_LR) )

print ('Report : ')

print (classification_report(y_test,y_pred_LR) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_LR ):

    cm = metrics.confusion_matrix( y_test,y_pred_LR )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_LR )



print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_LR):

    if (len(y_test.shape) != len(y_pred_LR.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_LR.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_LR)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
    conf_mat = create_conf_mat(y_test,y_pred_LR)

    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

    plt.xlabel('Predicted Values')

    plt.ylabel('Actual Values')

    plt.title('Actual vs. Predicted Confusion Matrix')

    plt.show()
#Get predicted probabilites

target_probailities_log = LR.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)

print(roc_auc_score(y_test,y_pred_nb))

Model.append('Naive Bayes')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
results=confusion_matrix(y_test,y_pred_nb)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_nb) )

print ('Report : ')

print (classification_report(y_test,y_pred_nb) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_nb ):

    cm = metrics.confusion_matrix( y_test,y_pred_nb )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_nb )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_nb):

    if (len(y_test.shape) != len(y_pred_nb.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_nb.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_nb)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_nb)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = nb.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()

Model,ROC_AUC_Accuracy
params = {

    

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':range(1,10),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



dt = DecisionTreeClassifier()



gs = GridSearchCV(estimator=dt,n_jobs=-1,cv=3,param_grid=params,scoring='recall')

gs.fit(train_out,y)
dt = DecisionTreeClassifier(**gs.best_params_)

dt.fit(x_train,y_train)

y_pred_dt = dt.predict(x_test)

print(roc_auc_score(y_test,y_pred_dt))

Model.append('Decision Tree')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_dt):

    cm = metrics.confusion_matrix( y_test,y_pred_dt )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
results=confusion_matrix(y_test,y_pred_dt)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_dt) )

print ('Report : ')

print (classification_report(y_test,y_pred_dt) )
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_dt )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_dt):

    if (len(y_test.shape) != len(y_pred_dt.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_dt.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_dt)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_dt)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = dt.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()

Model,ROC_AUC_Accuracy
params = {

    

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

    'max_depth':range(2,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



rf = RandomForestClassifier()



rs = RandomizedSearchCV(estimator=rf,param_distributions=params,cv=3,scoring='recall',n_jobs=-1)

rs.fit(train_out,y)
rf = RandomForestClassifier(**rs.best_params_)

rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print(roc_auc_score(y_test,y_pred_rf))

Model.append('Random Forest')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
results=confusion_matrix(y_test,y_pred_rf)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_rf) )

print ('Report : ')

print (classification_report(y_test,y_pred_rf) )

#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_rf ):

    cm = metrics.confusion_matrix( y_test,y_pred_rf )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_rf )

print("confusion matrix = \n",mat_pruned)

def create_conf_mat(y_test,y_pred_rf):

    if (len(y_test.shape) != len(y_pred_rf.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_rf.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_rf)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_rf)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = rf.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
Model,ROC_AUC_Accuracy
final_result = pd.DataFrame({'Model - VIF Data':Model,'Accuracy':ROC_AUC_Accuracy})

final_result
LR_Bag = BaggingClassifier(base_estimator=LR,n_estimators=100,n_jobs=-1,random_state=1)

nb_Bag = BaggingClassifier(base_estimator=nb,n_estimators=100,n_jobs=-1,random_state=1)

dt_Bag = BaggingClassifier(base_estimator=dt,n_estimators=100,n_jobs=-1,random_state=1)
x = np.array(x)
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = LR_Bag

name = 'Bagged-LR'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Bag

name = 'Bagged-NB'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = dt_Bag

name = 'Bagged-DT'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMModel,LGBMClassifier
LR_Boost = AdaBoostClassifier(base_estimator=LR,n_estimators=100,learning_rate=0.01,random_state=1)

nb_Boost = AdaBoostClassifier(base_estimator=nb,n_estimators=100,learning_rate=0.01,random_state=1)

dt_Boost = AdaBoostClassifier(base_estimator=dt,n_estimators=100,learning_rate=0.01,random_state=1)

rf_Boost = AdaBoostClassifier(base_estimator=rf,n_estimators=100,learning_rate=0.01,random_state=1)

gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

lgbm = LGBMClassifier(objective='binary',n_estimators=100,reg_alpha=2,reg_lambda=5,random_state=1,learning_rate=0.01,is_unbalance=True)
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = LR_Boost

name = 'Boosted-LR'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Boost

name = 'Boosted-NB'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = dt_Boost

name = 'Boosted-DT'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = lgbm

name = 'LGBM'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
final_result = pd.DataFrame({'Model - RFE Data':Model,'Accuracy':ROC_AUC_Accuracy})

final_result
from sklearn.linear_model import LassoCV, Lasso

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  

      str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (16.0, 20.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
col_names=['qu_sum', 'qa_sum', 'pprice_sum', 'price_sum', 'total_discount_sum', 'cd_sum', 'odd_sum', 'cdd_sum']
x=df[col_names]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1990)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
LR = LogisticRegression()

LR.fit(x_train,y_train)

y_pred_LR = LR.predict(x_test)

print(classification_report(y_test,y_pred_LR))
roc_auc_score(y_test,y_pred_LR)

Model = ['Logistic Regression']

ROC_AUC_Accuracy = [roc_auc_score(y_test,y_pred_LR)]
results=confusion_matrix(y_test,y_pred_LR)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_LR) )

print ('Report : ')

print (classification_report(y_test,y_pred_LR) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_LR ):

    cm = metrics.confusion_matrix( y_test,y_pred_LR )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_LR )



print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_LR):

    if (len(y_test.shape) != len(y_pred_LR.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_LR.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_LR)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
    conf_mat = create_conf_mat(y_test,y_pred_LR)

    sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

    plt.xlabel('Predicted Values')

    plt.ylabel('Actual Values')

    plt.title('Actual vs. Predicted Confusion Matrix')

    plt.show()
#Get predicted probabilites

target_probailities_log = LR.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)

print(roc_auc_score(y_test,y_pred_nb))

Model.append('Naive Bayes')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_nb))
print(classification_report(y_test,y_pred_nb))
results=confusion_matrix(y_test,y_pred_nb)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_nb) )

print ('Report : ')

print (classification_report(y_test,y_pred_nb) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_nb ):

    cm = metrics.confusion_matrix( y_test,y_pred_nb )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_nb )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_nb):

    if (len(y_test.shape) != len(y_pred_nb.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_nb.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_nb)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_nb)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = nb.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
Model,ROC_AUC_Accuracy
params = {

    

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':range(1,10),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



dt = DecisionTreeClassifier()



rs = RandomizedSearchCV(estimator=dt,n_jobs=-1,cv=3,param_distributions=params,scoring='recall')

rs.fit(x,y)
dt = DecisionTreeClassifier(**rs.best_params_)

dt.fit(x_train,y_train)

y_pred_dt = dt.predict(x_test)

print(roc_auc_score(y_test,y_pred_dt))

Model.append('Decision Tree')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_dt):

    cm = metrics.confusion_matrix( y_test,y_pred_dt )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
results=confusion_matrix(y_test,y_pred_dt)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_dt) )

print ('Report : ')

print (classification_report(y_test,y_pred_dt) )
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_dt )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_dt):

    if (len(y_test.shape) != len(y_pred_dt.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_dt.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_dt)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_dt)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = dt.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)
#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
Model,ROC_AUC_Accuracy
params = {

    

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

    'max_depth':range(2,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



rf = RandomForestClassifier()



rs = RandomizedSearchCV(estimator=rf,param_distributions=params,cv=5,scoring='recall',n_jobs=-1)

rs.fit(x,y)
rf = RandomForestClassifier(**rs.best_params_)

rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

print(roc_auc_score(y_test,y_pred_rf))

Model.append('Random Forest')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
results=confusion_matrix(y_test,y_pred_rf)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_rf) )

print ('Report : ')

print (classification_report(y_test,y_pred_rf) )

#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_rf ):

    cm = metrics.confusion_matrix( y_test,y_pred_rf )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_rf )

print("confusion matrix = \n",mat_pruned)

def create_conf_mat(y_test,y_pred_rf):

    if (len(y_test.shape) != len(y_pred_rf.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_rf.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_rf)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_rf)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = rf.predict_proba(x_test)[:,1]
#Create true and false positive rates

log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,target_probailities_log)

#Plot ROC Curve

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(log_false_positive_rate,log_true_positive_rate)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.show()
Model,ROC_AUC_Accuracy
final_result = pd.DataFrame({'Model':Model,'Accuracy':ROC_AUC_Accuracy})

final_result
LR_Bag = BaggingClassifier(base_estimator=LR,n_estimators=100,n_jobs=-1,random_state=1)

nb_Bag = BaggingClassifier(base_estimator=nb,n_estimators=100,n_jobs=-1,random_state=1)

dt_Bag = BaggingClassifier(base_estimator=dt,n_estimators=100,n_jobs=-1,random_state=1)
x = np.array(x)
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = LR_Bag

name = 'Bagged-LR'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Bag

name = 'Bagged-NB'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = dt_Bag

name = 'Bagged-DT'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMModel,LGBMClassifier
LR_Boost = AdaBoostClassifier(base_estimator=LR,n_estimators=100,learning_rate=0.01,random_state=1)

nb_Boost = AdaBoostClassifier(base_estimator=nb,n_estimators=100,learning_rate=0.01,random_state=1)

dt_Boost = AdaBoostClassifier(base_estimator=dt,n_estimators=100,learning_rate=0.01,random_state=1)

rf_Boost = AdaBoostClassifier(base_estimator=rf,n_estimators=100,learning_rate=0.01,random_state=1)

gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

lgbm = LGBMClassifier(objective='binary',n_estimators=100,reg_alpha=2,reg_lambda=5,random_state=1,learning_rate=0.01,is_unbalance=True)
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = LR_Boost

name = 'Boosted-LR'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = nb_Boost

name = 'Boosted-NB'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = dt_Boost

name = 'Boosted-DT'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
kf = KFold(n_splits=5,shuffle=True,random_state=1)

model = lgbm

name = 'LGBM'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
final_result = pd.DataFrame({'Model - RFE Data':Model,'Accuracy':ROC_AUC_Accuracy})

final_result