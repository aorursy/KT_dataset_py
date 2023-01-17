# import all libraries required 
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import sklearn.preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import re
# read churn dataset
tchurn = pd.read_csv("../input/telecom_churn_data.csv")
#lets take a look on data 
tchurn.head()
print("There are total %d columns." %tchurn.shape[1])
print("There are total %d observations." %tchurn.shape[0])
#Lets look more into attributes and stats of dataset
tchurn.info()
tchurn.describe()
#lets get all the column names
for col in tchurn.columns:
    print(col)
#As we can see last 4 columns of above dataset has month name as part of their name lets make it 
#similar to other column standard
tchurn = tchurn.rename(columns={'jun_vbc_3g': 'vbc_3g_6', 'jul_vbc_3g': 'vbc_3g_7', 'aug_vbc_3g': 'vbc_3g_8', 'sep_vbc_3g': 'vbc_3g_9'})
#lets get all the column names
for col in tchurn.columns:
    print(col)
#lets get the all null values of all columns in percentage, it would be better to look in terms of percentage
print("Total Null Values in percentage:\n")
(100*(tchurn.isnull().sum())/len(tchurn.index))
#Lets look into columns which are important to identify high value customers
#check if they have null values
tchurn[['total_rech_amt_7','total_rech_amt_6','av_rech_amt_data_6','av_rech_amt_data_7','total_rech_data_6','total_rech_data_7']].isnull().sum()
#lets impute missing values with '0' to extract high value customers for these columns 
tchurn[['av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8','av_rech_amt_data_9','total_rech_data_6','total_rech_data_7','total_rech_data_8','total_rech_data_9']]=tchurn[['av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8','av_rech_amt_data_9','total_rech_data_6','total_rech_data_7','total_rech_data_8','total_rech_data_9']].fillna(0, axis=1)
#Lets impute all these columns with '0' as they look important for model building
col4 = ['max_rech_data_6','max_rech_data_7','max_rech_data_8','count_rech_2g_6','count_rech_2g_7','count_rech_2g_8','count_rech_3g_6','count_rech_3g_7','count_rech_3g_8','arpu_3g_6','arpu_3g_7','arpu_3g_8','arpu_2g_6','arpu_2g_7','arpu_2g_8','night_pck_user_6','night_pck_user_7','night_pck_user_8','fb_user_6','fb_user_7','fb_user_8']
tchurn[col4]=tchurn[col4].replace(np.nan, 0)
#lets check for null values
tchurn.isnull().sum()
#lets sum up all types of data recharge in the month
tchurn['total_rech_num_data_6'] = (tchurn['count_rech_2g_6']+tchurn['count_rech_3g_6']).astype(int)
tchurn['total_rech_num_data_7'] = (tchurn['count_rech_2g_7']+tchurn['count_rech_3g_7']).astype(int)
tchurn['total_rech_num_data_8'] = (tchurn['count_rech_2g_8']+tchurn['count_rech_3g_8']).astype(int)
#lets calculate total amount spent on recharging data(mobile internet) in the month
#multiply amount with number of times it was recharged for data 
tchurn['total_rech_amt_data_6'] = tchurn['total_rech_num_data_6']*tchurn['av_rech_amt_data_6']
tchurn['total_rech_amt_data_7'] = tchurn['total_rech_num_data_7']*tchurn['av_rech_amt_data_7']
tchurn['total_rech_amt_data_8'] = tchurn['total_rech_num_data_8']*tchurn['av_rech_amt_data_8']
#lets calculate total monthly recharge for data and call, so sum amounts spents on call and data recharge for the month.
tchurn['total_month_rech_6'] = tchurn['total_rech_amt_6']+tchurn['total_rech_amt_data_6']
tchurn['total_month_rech_7'] = tchurn['total_rech_amt_7']+tchurn['total_rech_amt_data_7']
tchurn['total_month_rech_8'] = tchurn['total_rech_amt_8']+tchurn['total_rech_amt_data_8']
#lets extract high value customers based on the average recharge amount in the first two months(6,7) (the good phase).
hv_cust=tchurn[tchurn[['total_month_rech_6','total_month_rech_7']].mean(axis=1)> tchurn[['total_month_rech_6','total_month_rech_7']].mean(axis=1).quantile(0.7)]
#lets get the number of features and observations in new dataset high value customers
#hv_cust.info()
print("There are total %d features." %hv_cust.shape[1])
print("There are total %d observations." %hv_cust.shape[0])
#lets get the all null values of all columns in percentage, it would be better to look in terms of percentage
print("Total Null Values in percentage:\n")
(100*(hv_cust.isnull().sum())/len(hv_cust.index))
#lets define a function  to find all the columns where more than percentahe of values are null.
#Looking at the above statistics there are many columns where values are null for more than 49% 
def nullvalue(cutoff):
    null = (100*(hv_cust.isnull().sum())/len(hv_cust.index))
    print("{} features have more than {}% null values".format(len(null.loc[null > cutoff]),cutoff))
    return null.loc[null > cutoff]
nullvalue(49)
col1 = ['vol_3g_mb_9', 'vol_2g_mb_9','total_ic_mou_9','total_og_mou_9']
hv_cust['churn']=hv_cust[col1].apply(lambda x: 1 if ((x['vol_3g_mb_9']==0) & (x['vol_2g_mb_9']==0.0) & (x['total_ic_mou_9']==0)  & (x['total_og_mou_9']==0)) else 0, axis=1)
print("Total number of customers churned is:",len(hv_cust[hv_cust['churn']==1]))
print("Total number of customers non-churned is:",len(hv_cust[hv_cust['churn']==0]))
#Lets take a look on stats
hv_cust.shape
#After tagging churners, remove all the attributes corresponding to the churn phase
#(all attributes having ‘ _9’, etc. in their names).
import re
#filter all columns where last char in column name is _9
col2 = hv_cust.filter(regex=('_9')).columns
#drop these columns as mentioned
hv_cust.drop(col2,axis=1,inplace=True)
#lets get the number of features and observations in new dataset high value customers
#hv_cust.info()
print("Total features.",hv_cust.shape[1])
print("Total observations.",hv_cust.shape[0])
#Lets look into few features. Circle id and mobile number can be dropped from the list.
# circle id has only one value so drop it. mobile number has not much importance in our analysis
hv_cust.circle_id.value_counts()
hv_cust.drop(['circle_id','mobile_number'],axis=1,inplace=True)
#lets look into all date columns and convert them into correct format
#filter column names where they have date in their name
col3 = hv_cust.filter(regex=('date')).columns
col3
# lets Convert dtype of date columns to datetime
hv_cust['last_date_of_month_6'] = pd.to_datetime(hv_cust['last_date_of_month_6'], format='%m/%d/%Y')
hv_cust['last_date_of_month_7'] = pd.to_datetime(hv_cust['last_date_of_month_7'], format='%m/%d/%Y')
hv_cust['last_date_of_month_8'] = pd.to_datetime(hv_cust['last_date_of_month_8'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_6'] = pd.to_datetime(hv_cust['date_of_last_rech_6'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_7'] = pd.to_datetime(hv_cust['date_of_last_rech_7'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_8'] = pd.to_datetime(hv_cust['date_of_last_rech_8'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_data_6'] = pd.to_datetime(hv_cust['date_of_last_rech_data_6'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_data_7'] = pd.to_datetime(hv_cust['date_of_last_rech_data_7'], format='%m/%d/%Y')
hv_cust['date_of_last_rech_data_8'] = pd.to_datetime(hv_cust['date_of_last_rech_data_8'], format='%m/%d/%Y')
#lets get columns which have more than 0% missing values
nullvalue(0)
#Lets look into columns which have only values as 0 as we looked into stats thru describe
#looks like all 3 columns have only 0 and null values.
print(hv_cust['loc_og_t2o_mou'].unique())
print(hv_cust['std_og_t2o_mou'].unique())
print(hv_cust['loc_ic_t2o_mou'].unique())
#lets drop above 3 columns from dataset
hv_cust.drop(['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou'],inplace=True,axis=1)
#Lets look into columns which have only values as 0. 
#looks like all 6 columns have only 0 and null values.
print(hv_cust['std_og_t2c_mou_6'].unique())
print(hv_cust['std_og_t2c_mou_7'].unique())
print(hv_cust['std_og_t2c_mou_8'].unique())
print(hv_cust['std_ic_t2o_mou_6'].unique())
print(hv_cust['std_ic_t2o_mou_7'].unique())
print(hv_cust['std_ic_t2o_mou_8'].unique())
#lets drop above 6 columns from dataset
hv_cust.drop(['std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8','std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8'],inplace=True,axis=1)
#lets get columns which have more than 3% missing values
nullvalue(3)
#Lets drop these columns 3 date columns which have more than 40% values as null.
#they don't see to much imporatnt as we already have date columns 
hv_cust.drop(['date_of_last_rech_data_6','date_of_last_rech_data_7','date_of_last_rech_data_8'],inplace=True,axis=1)
missing3 = list(nullvalue(3).index)
missing3
#Lets impute all these columns with '0' as they look important for model building
hv_cust[missing3]=hv_cust[missing3].replace(np.nan, 0)
#Lets look into date columns for unique values.
hv_cust['date_of_last_rech_6'].unique()
hv_cust['date_of_last_rech_7'].unique()
hv_cust['date_of_last_rech_8'].unique()
# they all have dates for only one month in all their rows. that means rechrge was done in that particular month.
# we will just impute a particular date of that month for all those null value rows.
#Filling null values with the previous ones
hv_cust['date_of_last_rech_6'].fillna(method ='pad',inplace=True) 
hv_cust['date_of_last_rech_7'].fillna(method ='pad',inplace=True) 
hv_cust['date_of_last_rech_8'].fillna(method ='pad',inplace=True) 
#lets get columns which have more than 1% missing values
nullvalue(0)
#Lets look into date columns for unique values.
print(hv_cust['last_date_of_month_7'].unique())
hv_cust['last_date_of_month_8'].unique()
# they all have same dates for the month in all their rows and null values for few.
# we will just impute a particular same date of that month for all those null value rows.
#Filling null values with the previous ones in the dataset
hv_cust['last_date_of_month_7'].fillna(method ='pad',inplace=True) 
hv_cust['last_date_of_month_8'].fillna(method ='pad',inplace=True) 
#now we have 54 columns which have null values which have almost same percentage of null values
#also we can notice that all these columns belong to 6th and 7th month so lets impute these columsn with 0s.
#lets get all the columns with null values as they are important for our analysis.
missing0 = list(nullvalue(0).index)
#Lets impute all these columns with '0' as they look important for model building
hv_cust[missing0]=hv_cust[missing0].replace(np.nan, 0)
#Lets look for null values one last time
nullvalue(0)
#we have taken care of all null values.
print("Total features.",hv_cust.shape[1])
print("Total observations.",hv_cust.shape[0])
#lets list all the columns currently present in dataframe
hv_cust.columns.values
# Lets see distribution of same fields in each month using box plot.
# Quantitative Variables
import seaborn as sns
plt.figure(figsize=(15,8),facecolor='b')
sns.set_style("dark")
# subplot 1
plt.subplot(2, 3, 1)
ax = sns.boxplot(hv_cust['roam_og_mou_6'])
ax.set_title('Outgoing roaming Usage mon-6 - Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 2)
ax = sns.boxplot(hv_cust['roam_og_mou_7'])
ax.set_title('Outgoing roaming Usage mon-7- Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 3)
ax = sns.boxplot(hv_cust['roam_og_mou_8'])
ax.set_title('Outgoing roaming Usage mon-8- Box Plot',fontsize=14,color='w')
plt.show()

# Observation: 
# Distribution of roaming usage shows august month usage has reduced for sure. 
# but it should have been increased if customer is happy.
# Lets see distribution of same fields in each motnh using box plot.
# Quantitative Variables

plt.figure(figsize=(15,8),facecolor='b')
sns.set_style("dark")
# subplot 1
plt.subplot(2, 3, 1)
ax = sns.boxplot(hv_cust['total_og_mou_6'])
ax.set_title('total Outgoing Usage mon-6 - Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 2)
ax = sns.boxplot(hv_cust['total_og_mou_7'])
ax.set_title('total Outgoing Usage mon-7- Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 3)
ax = sns.boxplot(hv_cust['total_og_mou_8'])
ax.set_title('total Outgoing Usage mon-8- Box Plot',fontsize=14,color='w')
plt.show()

# Observation: 
# Distribution of total outgoing usage shows august month usage has reduced for sure. 
# but it should have been increased or constant if customer is happy but it doesn't look that way.
# Lets see distribution of same fields in each motnh using box plot.
# Quantitative Variables

plt.figure(figsize=(15,8),facecolor='b')
sns.set_style("dark")
# subplot 1
plt.subplot(2, 3, 1)
ax = sns.boxplot(hv_cust['total_ic_mou_6'])
ax.set_title('total incomig Usage mon-6 - Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 2)
ax = sns.boxplot(hv_cust['total_ic_mou_7'])
ax.set_title('total incoming Usage mon-7- Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 3)
ax = sns.boxplot(hv_cust['total_ic_mou_8'])
ax.set_title('total incoming Usage mon-8- Box Plot',fontsize=14,color='w')
plt.show()

# Observation: 
# Distribution of total incoming usage shows august month usage has got better or constant for sure. 
# Lets see distribution of same fields in each motnh using box plot.
# Quantitative Variables

plt.figure(figsize=(15,8),facecolor='b')
sns.set_style("dark")
# subplot 1
plt.subplot(2, 3, 1)
ax = sns.boxplot(hv_cust['last_day_rch_amt_6'])
ax.set_title('Last Recharge amount mon-6 - Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 2)
ax = sns.boxplot(hv_cust['last_day_rch_amt_7'])
ax.set_title('Last Recharge amount mon-7 - BOx Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 3)
ax = sns.boxplot(hv_cust['last_day_rch_amt_8'])
ax.set_title('Last Recharge amount mon-8 - BOx Plot',fontsize=14,color='w')
plt.show()

# Observation: 
# Distribution of recharge amount in august shows customer has reduced recharge amount for sure. 
# but it should have been increased or constant if customer is happy but it doesn't look that way.
# Lets see distribution of same fields in each motnh using box plot.
# Quantitative Variables

plt.figure(figsize=(15,8),facecolor='b')
sns.set_style("dark")
# subplot 1
plt.subplot(2, 3, 1)
ax = sns.boxplot(hv_cust['total_month_rech_6'])
ax.set_title('Total monthly recharge-6 - Box Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 2)
ax = sns.boxplot(hv_cust['total_month_rech_7'])
ax.set_title('Total monthly recharge-7 - BOx Plot',fontsize=14,color='w')
# subplot 2
plt.subplot(2, 3, 3)
ax = sns.boxplot(hv_cust['total_month_rech_8'])
ax.set_title('Total monthly recharge-8 - BOx Plot',fontsize=14,color='w')
plt.show()

# Observation: 
# Distribution of total monthly recharge amount in august shows customer has reduced recharge amount for sure. 
# but it should have been increased or constant if customer is happy but it doesn't look that way.
# Lets see distribution of same fields in each motnh using box plot.
# Quantitative Variables

plt.figure(figsize=(8,4),facecolor='b')
sns.set_style("dark")
ax = sns.boxplot(hv_cust['aon'])
ax.set_title('Age on Netwrok - Box Plot',fontsize=14,color='w')
plt.show()
#sum of total isd MOU per month churn vs Non-Churn
hv_cust.groupby(['churn'])['isd_og_mou_6','isd_og_mou_7','isd_og_mou_8'].sum()
#mean of total 3G usage per month churn vs Non-Churn
hv_cust.groupby(['churn'])['vol_3g_mb_6','vol_3g_mb_7','vol_3g_mb_8'].mean()
#mean of total 2G usage per month churn vs Non-Churn
hv_cust.groupby(['churn'])['vol_2g_mb_6','vol_2g_mb_7','vol_2g_mb_8'].mean()
#mean of total std MOU per month churn vs Non-Churn
hv_cust.groupby(['churn'])['std_og_mou_6','std_og_mou_7','std_og_mou_8'].mean()
#sum of total special MOU per month churn vs Non-Churn
hv_cust.groupby(['churn'])['spl_og_mou_6','spl_og_mou_7','spl_og_mou_8'].sum()
#mean of total incoming MOU per month churn vs Non-Churn
hv_cust.groupby(['churn'])['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8'].mean()
#mean of total outgoing MOU per month churn vs Non-Churn
hv_cust.groupby(['churn'])['total_og_mou_6','total_og_mou_7','total_og_mou_8'].mean()
#mean of total monthly recharge per month churn vs Non-Churn
hv_cust.groupby(['churn'])['total_rech_amt_6','total_rech_amt_7','total_rech_amt_8'].mean()
#mean of outgoing in roaming usage per month churn vs Non-Churn
hv_cust.groupby(['churn'])['roam_og_mou_6','roam_og_mou_7','roam_og_mou_8'].mean()
#mean of maximum recharge amount per month churn vs Non-Churn
hv_cust.groupby(['churn'])['max_rech_amt_6','max_rech_amt_7','max_rech_amt_8'].mean()
#mean of count of total data recharge per month churn vs Non-Churn
hv_cust.groupby(['churn'])['total_rech_num_data_6','total_rech_num_data_7','total_rech_num_data_8'].mean()
#mean of last recharge amount churn vs Non-Churn
hv_cust.groupby(['churn'])['last_day_rch_amt_6','last_day_rch_amt_7','last_day_rch_amt_8'].mean()
#mean of local outgoing on same network usage per month churn vs Non-Churn
hv_cust.groupby(['churn'])['loc_og_t2t_mou_6','loc_og_t2t_mou_7','loc_og_t2t_mou_8'].mean()
#mean of age on network churn vs Non-Churn
hv_cust.groupby(['churn'])['aon'].mean()
#lets copy the dataframe to another before we do other activities
hv_custcopy = hv_cust
print(hv_custcopy.info())
hv_custcopy.head()
hv_custcopy.describe()
#lets remove aon column 
hv_custcopy.drop(['aon'], axis=1, inplace=True)
#lets remove datetime columns from dataset else it will give error further
datecols = list(hv_custcopy.select_dtypes(include=['datetime']).columns)
print(datecols)
hv_custcopy.drop(datecols, axis=1, inplace=True)
#lets import train test split 
from sklearn.model_selection import train_test_split
X = hv_custcopy.drop(['churn'], axis=1)
y = hv_custcopy['churn']    
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)
#perform minmax scaling before PCA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# fit transform the scaler on train
X_train = scaler.fit_transform(X_train)
# transform test using the already fit scaler
X_test = scaler.transform(X_test)
#lets print the stats before sampling
print("counts of label '1':",sum(y_train==1))
print("counts of label '0':",sum(y_train==0))
#perform oversampling using smote
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
X_train_smo, y_train_smo = sm.fit_sample(X_train, y_train)
#lets print stats after smote
print("counts of label '1':",sum(y_train_smo==1))
print("counts of label '0':",sum(y_train_smo==0))
#lets perform PCA on sampled data. import PCA
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)
#lets fit PCA on the train dataset
pca.fit(X_train_smo)
pca.explained_variance_ratio_[:50]
#lets draw screeplot in between cumulative variance and number of components
%matplotlib inline
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
#lets perform incremental PCA for efficiency 
from sklearn.decomposition import IncrementalPCA
pca_again = IncrementalPCA(n_components=35)
#fit
X_train_pca = pca_again.fit_transform(X_train_smo)
X_train_pca.shape
#lets create correlation matrix for the principal components
corrmat = np.corrcoef(X_train_pca.transpose())
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
#correlations are close to 0
#Applying selected components to the test data - 35 components
X_test_pca = pca_again.transform(X_test)
X_test_pca.shape
#import library and fit train model on train data
#class_weight="balanced":it basically means replicating the smaller class until you have as many samples as in the larger one, 
#but in an implicit way.Though we have already used smote but here we can use this too.
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
learner_pca2 = LogisticRegression(class_weight='balanced')
learner_pca2.fit(X_train_pca,y_train_smo)
#Predict on training set
dtrain_predictions = learner_pca2.predict(X_train_pca)
dtrain_predprob = learner_pca2.predict_proba(X_train_pca)[:,1]
#lets print some scores
print ("Accuracy :",metrics.roc_auc_score(y_train_smo, dtrain_predictions))
print ("Recall/Sensitivity :",metrics.recall_score(y_train_smo, dtrain_predictions))
print ("AUC Score (Train):",metrics.roc_auc_score(y_train_smo, dtrain_predprob))
#lets predict on test dataset.
#print all scores
pred_probs_test = learner_pca2.predict(X_test_pca)
confusion = metrics.confusion_matrix(y_test, pred_probs_test)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",(metrics.roc_auc_score(y_test, pred_probs_test)))
print('precision score:',(metrics.precision_score(y_test, pred_probs_test)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
print("Accuracy :",(metrics.accuracy_score(y_test,pred_probs_test)))
#lets check with probability cutoff 0.5
y_train_pred = learner_pca2.predict_proba(X_train_pca)[:,1]
y_train_pred_final = pd.DataFrame({'Churn':y_train_smo, 'Churn_Prob':y_train_pred})
y_train_pred_final['Churn_Prob'] = y_train_pred
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
#lets define function for ROC curve
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Churn, y_train_pred_final.Churn_Prob, drop_intermediate = False )
#lets draw roc curve
draw_roc(y_train_pred_final.Churn, y_train_pred_final.Churn_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
#lets plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
#apply cutoff probability
y_train_pred_final['final_predicted'] = y_train_pred_final.Churn_Prob.map( lambda x: 1 if x > 0.45 else 0)
y_train_pred_final.head()
#lets predict on train dataset with optimal cutoff probability
y_train_pred = learner_pca2.predict_proba(X_train_pca)[:,1]
y_train_pred_final = pd.DataFrame({'Churn':y_train_smo, 'Churn_Prob':y_train_pred})
y_train_pred_final['Churn_Prob'] = y_train_pred
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_train_pred_final.head()
#lets find out all scores of train dataset
#print all scores
confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",(metrics.roc_auc_score(y_train_pred_final.Churn, y_train_pred_final.predicted)))
print('precision score:',(metrics.precision_score(y_train_pred_final.Churn, y_train_pred_final.predicted)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
#lets predict on test datset with optimal cutoff obtained earlier
y_test_pred = learner_pca2.predict_proba(X_test_pca)[:,1]
y_test_pred_final = pd.DataFrame({'Churn':y_test, 'Churn_Prob':y_test_pred})
y_test_pred_final['Churn_Prob'] = y_test_pred
y_test_pred_final['predicted'] = y_test_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_test_pred_final.head()
#lets find out all scores of test dataset
#print all scores
confusion = metrics.confusion_matrix(y_test_pred_final.Churn, y_test_pred_final.predicted)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",metrics.roc_auc_score(y_test_pred_final.Churn, y_test_pred_final.predicted))
print('precision score :',(metrics.precision_score(y_test_pred_final.Churn, y_test_pred_final.predicted)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
# GridSearchCV to find optimal n_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(10, 30, 5)}

# instantiate the model
rf = RandomForestClassifier()

# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",
                 return_train_score=True)
rf.fit(X_train_pca, y_train_smo)
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with max_depth
plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal n_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'n_estimators': range(50, 150, 25)}

# instantiate the model (note we are specifying a max_depth)
rf = RandomForestClassifier(max_depth=20)

# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
rf.fit(X_train_pca, y_train_smo)
# scores of GridSearch CV
scores = rf.cv_results_
pd.DataFrame(scores).head()
# plotting accuracies with n_estimators
plt.figure()
plt.plot(scores["param_n_estimators"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_n_estimators"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal max_features
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'max_features': [4, 8, 14, 20, 24]}

# instantiate the model
rf = RandomForestClassifier(max_depth=20,n_estimators=80)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
rf.fit(X_train_pca, y_train_smo)
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
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal min_samples_leaf
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(100, 400, 50)}

# instantiate the model
rf = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
rf.fit(X_train_pca, y_train_smo)
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
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# GridSearchCV to find optimal min_samples_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(50, 300, 50)}

# instantiate the model
rf = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5,min_samples_leaf=100)


# fit tree on training data
rf = GridSearchCV(rf, parameters, 
                    cv=n_folds, 
                   scoring="accuracy",return_train_score=True)
rf.fit(X_train_pca, y_train_smo)
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
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=20,
                             min_samples_leaf=100, 
                             min_samples_split=100,
                             max_features=5,
                             n_estimators=80,
                             random_state=10)
# fit
rf_pca=rfc.fit(X_train_pca,y_train_smo)
#Predict on training set
rtrain_predictions = rf_pca.predict(X_train_pca)
rtrain_predprob = rf_pca.predict_proba(X_train_pca)[:,1]
#lets print some scores
print ("Accuracy :",metrics.roc_auc_score(y_train_smo, rtrain_predictions))
print ("Recall/Sensitivity :",metrics.recall_score(y_train_smo, rtrain_predictions))
print ("AUC Score (Train):",metrics.roc_auc_score(y_train_smo, rtrain_predprob))
#lets predict on test dataset
pred_probs_test = rf_pca.predict(X_test_pca)
confusion = metrics.confusion_matrix(y_test, pred_probs_test)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",(metrics.roc_auc_score(y_test, pred_probs_test)))
print('precision score:',(metrics.precision_score(y_test, pred_probs_test)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
print("Accuracy :",(metrics.accuracy_score(y_test,pred_probs_test)))
#lets check with probability cutoff 0.5
y_train_predrf = rf_pca.predict_proba(X_train_pca)[:,1]
y_train_predrf_final = pd.DataFrame({'Churn':y_train_smo, 'Churn_Prob':y_train_predrf})
y_train_predrf_final['Churn_Prob'] = y_train_predrf
y_train_predrf_final['predicted'] = y_train_predrf_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_predrf_final.head()
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_predrf_final[i]= y_train_predrf_final.Churn_Prob.map(lambda x: 1 if x > i else 0)
y_train_predrf_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_predrf_final.Churn, y_train_predrf_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
#lets plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
#apply cutoff probability
y_train_predrf_final['final_predicted'] = y_train_predrf_final.Churn_Prob.map( lambda x: 1 if x > 0.45 else 0)
#lets predict on train dataset with optimal cutoff probability
y_train_predrf = rf_pca.predict_proba(X_train_pca)[:,1]
y_train_predrf_final = pd.DataFrame({'Churn':y_train_smo, 'Churn_Prob':y_train_predrf})
y_train_predrf_final['Churn_Prob'] = y_train_predrf
y_train_predrf_final['predicted'] = y_train_predrf_final.Churn_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_train_predrf_final.head()
#lets find out all scores of train dataset
confusion = metrics.confusion_matrix(y_train_predrf_final.Churn, y_train_predrf_final.predicted)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",(metrics.roc_auc_score(y_train_predrf_final.Churn, y_train_predrf_final.predicted)))
print('precision score:',(metrics.precision_score(y_train_predrf_final.Churn, y_train_predrf_final.predicted)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
#lets predict on test datset with optimal cutoff obtained earlier
y_test_predrf = rf_pca.predict_proba(X_test_pca)[:,1]
y_test_predrf_final = pd.DataFrame({'Churn':y_test, 'Churn_Prob':y_test_predrf})
y_test_predrf_final['Churn_Prob'] = y_test_predrf
y_test_predrf_final['predicted'] = y_test_predrf_final.Churn_Prob.map(lambda x: 1 if x > 0.45 else 0)
y_test_predrf_final.head()
#lets find out all scores of test dataset
confusion = metrics.confusion_matrix(y_test_predrf_final.Churn, y_test_predrf_final.predicted)
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
print("Roc_auc_score :",metrics.roc_auc_score(y_test_predrf_final.Churn, y_test_predrf_final.predicted))
print('precision score :',(metrics.precision_score(y_test_predrf_final.Churn, y_test_predrf_final.predicted)))
print('Sensitivity/Recall :',(TP / float(TP+FN)))
print('Specificity:',(TN / float(TN+FP)))
print('False Positive Rate:',(FP/ float(TN+FP)))
print('Positive predictive value:',(TP / float(TP+FP)))
print('Negative Predictive value:',(TN / float(TN+ FN)))
# model with the best hyperparameters
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=10,
                             min_samples_leaf=100, 
                             min_samples_split=100,
                             max_features=5,
                             n_estimators=80)
# fit
rfc.fit(X_train_smo,y_train_smo)
plt.figure(figsize=(15,10))
impo_features = pd.Series(rfc.feature_importances_, index=X.columns)
impo_features.nlargest((25)).sort_values().plot(kind='barh', align='center')
plt.show()