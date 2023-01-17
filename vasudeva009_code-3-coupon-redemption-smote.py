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
train.shape, campaign.shape, items.shape, coupons.shape, cust_demo.shape, cust_tran.shape, test.shape
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
train.head()
train.redemption_status.value_counts(normalize=True)*100
value=train['redemption_status'].value_counts().plot(kind='bar')

plt.ylabel('redemption_status')
cust_demo.head()
cust_demo.info()
cust_demo.marital_status.value_counts()
cust_demo.family_size.value_counts()
cust_demo.no_of_children.value_counts()
#The below lines of code is to get rid of the + and keeping 5+ as 5 and 3+ as 3 and converting the columns to int data type.

#type of family size = int64 ... Cant apply astype as we have 5+ as family size

#no of children = int64 ... we need to ignore the NaN values while converting to float

cust_demo['family_size'] = cust_demo.family_size.apply(lambda x: int(re.sub('\+','',x)))

cust_demo['no_of_children'] = cust_demo.no_of_children.apply(lambda x: int(re.sub('\+','',x)) if pd.notna(x) else x)
#Filling NaN values for marital_status



#customers with family size =1 will be single

cust_demo.loc[pd.isnull(cust_demo.marital_status) & (cust_demo.family_size == 1),'marital_status'] = 'Single'



#customers whos family size - no of childrens == 1, will also be single 

#This is applicable where there is only 1 parent --- We treat 1 parent as Single

cust_demo.loc[(cust_demo.family_size - cust_demo.no_of_children == 1) & pd.isnull(cust_demo.marital_status),'marital_status'] = 'Single'



#from the orignal data we have 186 of 196 customers with diff of 2 in their family size and number of childrens as

#Married (see the below cell) and hence where ever the difference is 2 and marital status is NaN and No of Children is 

#NaN we impute the Mariatl Status with Married

cust_demo.loc[(pd.isnull(cust_demo.marital_status)) & ((cust_demo.family_size - cust_demo.no_of_children) == 2)  

              & (pd.notnull(cust_demo.no_of_children)),'marital_status'] = 'Married'



#original data shows customers with fam size == 2, and NaN in no of childrens are majorly Married (see below cell skipping 1 cell)

cust_demo.loc[pd.isnull(cust_demo.marital_status) & (pd.isnull(cust_demo.no_of_children)) 

              & (cust_demo.family_size ==2),'marital_status'] = 'Married'
a = cust_demo.marital_status.groupby((cust_demo.family_size - cust_demo.no_of_children) == 2).value_counts()

print(a[True])
cust_demo.marital_status.isnull().sum()
#FillingNaN values for no of children



#Married people with family_size ==2 will have 0 childrens

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.marital_status == 'Married') & (cust_demo.family_size == 2),'no_of_children'] = 0



#customers with family size 1 will have zero childrens

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.family_size == 1), 'no_of_children'] = 0



#singles with family size == 2, will probably have 1 child

cust_demo.loc[pd.isnull(cust_demo.no_of_children) & (cust_demo.family_size == 2),'no_of_children'] = 1



cust_demo['no_of_children']=cust_demo['no_of_children'].astype(np.int64)
cust_demo.no_of_children.isnull().sum()
cust_demo.info()
#Label Encoding Marital Status --- 0 is Single and 1 is Married

cust_demo["marital_status"] = encoder.fit_transform(cust_demo["marital_status"])
# Label Encoding age_range ... 18-25 is 0, 26-35 is 1, 36-45 is 2, 46-55 is 3, 56-70 is 4 and 70+ is 5

cust_demo["age_range"] = encoder.fit_transform(cust_demo["age_range"])
cust_demo.head()
campaign.head()
campaign.info()
campaign.campaign_type.value_counts()
#Label Encoding Campaign type

campaign["campaign_type"] = encoder.fit_transform(campaign.campaign_type)
#Converting the date columns to date time

campaign['start_date'] = pd.to_datetime(campaign['start_date'], format = '%d/%m/%y')

campaign['end_date'] = pd.to_datetime(campaign['end_date'], format = '%d/%m/%y')
#Creating a new column campaign_duration

campaign["campaign_duration"] = campaign["end_date"] - campaign["start_date"]

campaign["campaign_duration"] = campaign["campaign_duration"].apply(lambda x: x.days) 
campaign.head()
cust_tran.head()
cust_tran.info()
#Converting the date column into date time

#Reset the index of the DataFrame, and use the default one instead.

#If the DataFrame has a MultiIndex, this method can remove one or more levels.

cust_tran['date'] = pd.to_datetime(cust_tran['date'])

cust_tran = cust_tran.sort_values('date').reset_index(drop=True)
cust_tran.head()
#Creating 3 new columns from the date column

cust_tran['day'] = cust_tran["date"].apply(lambda x: x.day)

cust_tran['dow'] = cust_tran["date"].apply(lambda x: x.weekday())

cust_tran['month'] = cust_tran["date"].apply(lambda x: x.month)
cust_tran.head()
#Given selling_price and other_discount are for the entire transaction. Hence getting the Actual value of the transaction.

cust_tran.selling_price = cust_tran.selling_price/cust_tran.quantity

cust_tran.other_discount = cust_tran.other_discount/cust_tran.quantity

cust_tran.selling_price = cust_tran.selling_price - cust_tran.other_discount
#Inserting a new column to know if the coupon was used or not

cust_tran['coupon_used'] = cust_tran.coupon_discount.apply(lambda x: 1 if x !=0 else 0)
cust_tran.head()
items.head()
items.brand_type.value_counts()
items.category.value_counts()
#Label Encoding the brand_type and category columns

items.brand_type = encoder.fit_transform(items["brand_type"])

items.category = encoder.fit_transform(items["category"])
items.head()
coupons.head()
Image("../input/Schema.png")
coupons_items = pd.merge(coupons, items, on="item_id", how="left")
coupons_items.head()
cust_tran.head()
# Aggregate transactions by item_id by mean for a particular customer

transactions1 = pd.pivot_table(cust_tran, index = "item_id", 

               values=['customer_id','quantity','selling_price', 'other_discount','coupon_discount','coupon_used'],

               aggfunc={'customer_id':lambda x: len(set(x)),

                        'quantity':np.mean,

                        'selling_price':np.mean,

                        'other_discount':np.mean,

                        'coupon_discount':np.mean,

                        'coupon_used': np.sum

                        } )

transactions1.reset_index(inplace=True)

transactions1.rename(columns={'customer_id': 'no_of_customers'}, inplace=True)
transactions1.head()
# Aggregate transactions by item_id by sum for a particular customer

transactions2 = pd.pivot_table(cust_tran, index = "item_id", 

               values=['customer_id','quantity','selling_price', 'other_discount','coupon_discount'],

               aggfunc={'customer_id':len,

                        'quantity':np.sum,

                        'selling_price':np.sum,

                        'other_discount':np.sum,

                        'coupon_discount':np.sum,

                        } )

transactions2.reset_index(inplace=True)

transactions2.rename(columns={'customer_id': 't_counts', 'quantity':'qu_sum',

                             'selling_price':'price_sum', 'other_discount':'od_sum',

                             'coupon_discount':'cd_sum'}, inplace=True)
transactions2.head()
transactions1 = pd.merge(transactions1, transactions2, on='item_id',how='left' )
transactions1['total_discount_mean'] = transactions1['coupon_discount'] + transactions1['other_discount']

transactions1['total_discount_sum'] = transactions1['od_sum'] + transactions1['cd_sum']

transactions1.head()
item_coupon_trans = pd.merge(coupons_items, transactions1, on='item_id', how='left')
item_coupon_trans.head()
item_coupon_trans.columns
coupon = pd.pivot_table(item_coupon_trans, index ="coupon_id",

                         values=[ 'item_id', 'brand', 'brand_type', 'category',

       'coupon_discount', 'coupon_used', 'no_of_customers', 'other_discount',

       'quantity', 'selling_price', 'cd_sum', 't_counts', 'od_sum', 'qu_sum',

       'price_sum', 'total_discount_mean', 'total_discount_sum'],

              aggfunc={'item_id':lambda x: len(set(x)),

                       'brand':lambda x: mode(x)[0][0],

                       'brand_type':lambda x: mode(x)[0][0],

                       'category':lambda x: mode(x)[0][0],

                       'coupon_discount':np.mean,

                       'no_of_customers':np.mean,

                       'other_discount':np.mean,

                       'quantity':np.mean,

                       'selling_price':np.mean,

                      'coupon_used': np.sum,

                       'cd_sum': np.sum,

                       't_counts': np.sum,

                       'od_sum': np.sum,

                       'qu_sum': np.sum,

                       'price_sum': np.sum,

                       'total_discount_mean': np.mean,

                       'total_discount_sum': np.sum

                      })

coupon.reset_index(inplace=True)
coupon.rename(columns={'item_id':'item_counts'}, inplace=True)
coupon.head()
# Aggregate transactions by customer_id

transactions3 = pd.pivot_table(cust_tran, index = "customer_id", 

               values=['item_id','quantity','selling_price', 'other_discount','coupon_discount','coupon_used','day','dow','month'],

               aggfunc={'item_id':lambda x: len(set(x)),

                        'quantity':np.mean,

                        'selling_price':np.mean,

                        'other_discount':np.mean,

                        'coupon_discount':np.mean,

                        'coupon_used': np.sum,

                        'day':lambda x: mode(x)[0][0],

                        'dow':lambda x: mode(x)[0][0],

                        'month':lambda x: mode(x)[0][0]}

              )

transactions3.reset_index(inplace=True)

transactions3.rename(columns={'item_id': 'no_of_items'}, inplace=True)

transactions3.head()
# Aggregate transactions by customer_id by sum

transactions4 = pd.pivot_table(cust_tran, index = "customer_id", 

               values=['item_id','quantity','selling_price', 'other_discount','coupon_discount'],

               aggfunc={'item_id':len,

                        'quantity':np.sum,

                        'selling_price':np.sum,

                        'other_discount':np.sum,

                        'coupon_discount':np.sum}

              )

transactions4.reset_index(inplace=True)

transactions4.rename(columns={'item_id': 'customer_id_count','quantity':'qa_sum','selling_price':'pprice_sum',

                             'other_discount':'odd_sum','coupon_discount':'cdd_sum'  }, inplace=True)

transactions4.head()
transactions = pd.merge(transactions3, transactions4, on='customer_id', how='left')

transactions.head()
def merge_all(df): 

    df=  pd.merge(df, coupon, on="coupon_id", how="left")

    df = pd.merge(df, campaign, on="campaign_id", how="left")

    df = pd.merge(df, cust_demo, on="customer_id", how="left")

    df = pd.merge(df, transactions, on='customer_id', how='left')

    return df
train = merge_all(train)

test = merge_all(test)
train.shape, test.shape
## To save the final file after merging the data

##train.to_csv('FinalData.csv')
train.isnull().sum()
test.isnull().sum()
def deal_na(df):

    for col in cust_demo.columns.tolist()[1:]:

        df[col].fillna(mode(df[col]).mode[0], inplace=True)

    return df



train = deal_na(train)

test = deal_na(test)
train.isnull().sum()
test.isnull().sum()
test_id = test['id']

target = train['redemption_status']

train.drop(['id','campaign_id','start_date','end_date', 'redemption_status'], axis=1, inplace=True)

test.drop(['id','campaign_id','start_date','end_date'], axis=1, inplace=True)
train.head()
train.columns
train.shape
target
value=target.value_counts().plot(kind='bar')

plt.ylabel('redemption_status')
df = pd.read_csv('../input/final_train.csv')

df.head()
df.drop(columns=['Unnamed: 0'],inplace=True)

df.head()
df.shape
from imblearn.over_sampling import SMOTE



# Separate input features and target

y = df.redemption_status

x = df.drop('redemption_status', axis=1)



# Standardizig the Data

col_names = ['cd_sum','coupon_discount_x', 'coupon_used_x', 'item_counts', 'no_of_customers',

       'od_sum', 'other_discount_x', 'price_sum', 'qu_sum', 'quantity_x',

       'selling_price_x', 't_counts', 'total_discount_mean',

       'total_discount_sum', 'campaign_type', 'campaign_duration',

        'family_size', 'no_of_children',

       'income_bracket', 'coupon_discount_y', 'coupon_used_y',

       'no_of_items', 'other_discount_y', 'quantity_y',

       'selling_price_y', 'cdd_sum', 'customer_id_count', 'odd_sum', 'qa_sum',

       'pprice_sum']

features = x[col_names]

scaler = StandardScaler().fit(features.values)

features = scaler.transform(features.values)

x[col_names] = features





# setting up testing and training sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1990)



sm = SMOTE(random_state=1990, ratio=1.0)

x_train, y_train = sm.fit_sample(x_train, y_train)
x.head()
x.columns
x.shape
x_train.shape,y_train.shape,x_test.shape,y_test.shape
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
params = {

    

    'n_neighbors': range(1,5),

    'weights': ['uniform','distance'],

    'algorithm': ['ball_tree','kd_tree'],

    'p': [1,2,3]

}



knn = KNeighborsClassifier()



rs = RandomizedSearchCV(estimator=knn,n_jobs=-1,cv=3,param_distributions=params,scoring='recall')

rs.fit(x,y)
knn = KNeighborsClassifier(**rs.best_params_)

knn.fit(x_train,y_train)

y_pred_knn = knn.predict(x_test)

print(roc_auc_score(y_test,y_pred_knn))

Model.append('k-Nearest-Neighbours')

ROC_AUC_Accuracy.append(roc_auc_score(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))
results=confusion_matrix(y_test,y_pred_knn)

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred_knn) )

print ('Report : ')

print (classification_report(y_test,y_pred_knn) )
#Function to visulise confusion matrix

def draw_cm( y_test,y_pred_knn ):

    cm = metrics.confusion_matrix( y_test,y_pred_knn )

    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels = ["0", "1"] , yticklabels = ["0", "1"] , cmap="Greens")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
#Confusion matrix

from sklearn.metrics import classification_report,confusion_matrix

mat_pruned = confusion_matrix(y_test,y_pred_knn )

print("confusion matrix = \n",mat_pruned)
def create_conf_mat(y_test,y_pred_knn):

    if (len(y_test.shape) != len(y_pred_knn.shape) == 1):

        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')

    elif (y_test.shape != y_pred_knn.shape):

        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')

    else:

        # Set Metrics

        test_crosstb_comp = pd.crosstab(index = y_test,

                                       columns = y_pred_knn)

        # Changed for Future deprecation of as_matrix

        test_crosstb = test_crosstb_comp.values

        return test_crosstb
conf_mat = create_conf_mat(y_test,y_pred_knn)

sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Actual vs. Predicted Confusion Matrix')

plt.show()
#Get predicted probabilites

target_probailities_log = knn.predict_proba(x_test)[:,1]
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
LR_Bag = BaggingClassifier(base_estimator=LR,n_estimators=100,n_jobs=-1,random_state=1)

knn_Bag = BaggingClassifier(base_estimator=knn,n_estimators=100,n_jobs=-1,random_state=1)

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

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = nb_Bag

name = 'Bagged-NB'

#for model,name in zip([LR_Bag,knn_Bag,nb_Bag,dt_Bag],['Bagged-LR','Bagged-kNN','Bagged-NB','Bagged-DT']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

knn_Boost = AdaBoostClassifier(base_estimator=knn,n_estimators=100,learning_rate=0.01,random_state=1)

nb_Boost = AdaBoostClassifier(base_estimator=nb,n_estimators=100,learning_rate=0.01,random_state=1)

dt_Boost = AdaBoostClassifier(base_estimator=dt,n_estimators=100,learning_rate=0.01,random_state=1)

rf_Boost = AdaBoostClassifier(base_estimator=rf,n_estimators=100,learning_rate=0.01,random_state=1)

gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

lgbm = LGBMClassifier(objective='binary',n_estimators=100,reg_alpha=2,reg_lambda=5,random_state=1,learning_rate=0.01,is_unbalance=True)
kf = KFold(n_splits=3,shuffle=True,random_state=1)

model = LR_Boost

name = 'Boosted-LR'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = nb_Boost

name = 'Boosted-NB'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = dt_Boost

name = 'Boosted-DT'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = rf_Boost

name = 'Boosted-Random Forest'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = gb_Boost

name = 'Gradient Boost'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

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

model = lgbm

name = 'LGBM'

#for model,name in zip([LR_Boost,nb_Boost,dt_Boost,rf_Boost,lgbm],['Boosted-LR','Boosted-NB','Boosted-DT','Boosted - Random Forest','LGBM']):

roc_acc = []

for train,test in kf.split(x,y):

    x_train = x[train,:]

    x_test = x[test,:]

    y_train = y[train]

    y_test = y[test]

    sm = SMOTE(random_state=1990, ratio=1.0)

    x_train, y_train = sm.fit_sample(x_train, y_train)

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    score = roc_auc_score(y_test,y_pred)

    fpr,tpr,_ = roc_curve(y_test,y_pred)

    roc_acc.append(auc(fpr,tpr))

Model.append(name)

ROC_AUC_Accuracy.append(np.mean(roc_acc))

print('The AUC Score for')

print('%s is %0.02f with variacne of (+/-) %0.5f'%(name,np.mean(roc_acc),np.var(roc_acc,ddof=1)))
final_result = pd.DataFrame({'Model - SMOTE Data':Model,'Accuracy':ROC_AUC_Accuracy})

final_result