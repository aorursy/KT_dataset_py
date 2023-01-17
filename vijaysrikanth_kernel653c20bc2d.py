# Import required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline    

pd.options.display.float_format = '{:.2f}'.format

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")
#Load the data into a dataframe

telecom = pd.read_csv("../input/telecom-churn-data-set-for-the-south-asian-market/telecom_churn_data.csv", low_memory=False)
telecom.head()
telecom.shape
telecom.info()
#mobile_number is unique

print(telecom.mobile_number.is_unique)

telecom.mobile_number.nunique()
# Columns with more than 70% missing values

colmns_missing_data = round(100*(telecom.isnull().sum()/len(telecom.index)), 2)

colmns_missing_data[colmns_missing_data >= 40]
telecom.shape
telecom.total_rech_data_6.fillna(value=0, inplace=True)

telecom.total_rech_data_7.fillna(value=0, inplace=True)

telecom.total_rech_data_8.fillna(value=0, inplace=True)

telecom.total_rech_data_9.fillna(value=0, inplace=True)#

telecom.av_rech_amt_data_6.fillna(value=0, inplace=True)

telecom.av_rech_amt_data_7.fillna(value=0, inplace=True)

telecom.av_rech_amt_data_8.fillna(value=0, inplace=True)

telecom.av_rech_amt_data_9.fillna(value=0, inplace=True)
#Total recharge amounts for months 6 and 7

#Total recharge amount logic = Total data recharge + Total recharge Amount. 

#if any of the data recharge columns are 0 then retain the total recharge amt column as is



telecom['total_rech_amt_6'] = np.where((telecom['total_rech_data_6'] != 0) & (telecom['av_rech_amt_data_6'] != 0),

                                            telecom['total_rech_data_6']*telecom['av_rech_amt_data_6']+telecom['total_rech_amt_6'],

                                            telecom['total_rech_amt_6'])



telecom['total_rech_amt_7'] = np.where((telecom['total_rech_data_7'] != 0) & (telecom['av_rech_amt_data_7'] != 0),

                                            telecom['total_rech_data_7']*telecom['av_rech_amt_data_7']+telecom['total_rech_amt_7'],

                                            telecom['total_rech_amt_7'])
# Filter high-value customers

telecom['av_rech_amt'] = (telecom["total_rech_amt_6"] + 

                          telecom["total_rech_amt_7"]) / 2.0

cutoff = telecom.av_rech_amt.quantile(.70)

print('70 percentile of first two months avg recharge amount: ', cutoff)

telecom_hv = telecom[telecom['av_rech_amt'] >= cutoff]
telecom_hv.shape
# We can drop total_rech_data_* and av_rech_amt_data_*

drop_data_columns = ["total_rech_data_6", "total_rech_data_7", "total_rech_data_8", "total_rech_data_9", 

                'av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8', 'av_rech_amt_data_9']

telecom_hv.drop(drop_data_columns, axis=1, inplace=True)
pd.set_option('display.max_rows', telecom_hv.shape[0]+1)
def conditions(s):

    if ((s['total_ic_mou_9'] <= 0) & (s['total_og_mou_9'] <= 0) & (s['vol_2g_mb_9'] <= 0) & (s['vol_3g_mb_9'] <= 0)):

        return 1

    else:

        return 0
telecom_hv['Churn'] = telecom_hv.apply(conditions, axis=1)
telecom_hv = telecom_hv.loc[:,~telecom_hv.columns.str.endswith('_9')]

telecom_hv = telecom_hv.loc[:,~telecom_hv.columns.str.startswith('sep')]
telecom_hv.shape
churn_rate = (sum(telecom_hv['Churn'])/len(telecom_hv['Churn'].index))*100

churn_rate
imbalance = (sum(telecom_hv['Churn'] != 0)/sum(telecom_hv['Churn'] == 0))*100

imbalance
#Study the dataset

telecom_hv.describe()
nunique = telecom_hv.apply(pd.Series.nunique)

cols_to_drop = nunique[nunique == 1].index

cols_to_drop
telecom_hv.drop(cols_to_drop,axis=1,inplace=True)

telecom_hv.shape
# sum it up to check how many rows have all missing values

print("All null values:", telecom_hv.isnull().all(axis=1).sum())

# drop rows with 55% of missing data

telecom_hv = telecom_hv[(telecom_hv.isnull().sum(axis=1)/telecom_hv.shape[1])*100 < 55]

print("Record Count after Row/Column Data deletion:", telecom_hv.shape[0])
#Create Bar Plot

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.barplot(x = 'Churn', y = 'arpu_6', data = telecom_hv)

plt.subplot(2,3,2)

plt.ylabel('Av Rev. Month 7')

sns.barplot(x = 'Churn', y = 'arpu_7', data = telecom_hv)

plt.subplot(2,3,3)

plt.ylabel('Av Rev. Month 8')

sns.barplot(x = 'Churn', y = 'arpu_8', data = telecom_hv)

#Create Bar Plot

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.barplot(x = 'Churn', y = 'total_og_mou_6', data = telecom_hv)

plt.subplot(2,3,2)

sns.barplot(x = 'Churn', y = 'total_og_mou_7', data = telecom_hv)

plt.subplot(2,3,3)

sns.barplot(x = 'Churn', y = 'total_og_mou_8', data = telecom_hv)

#Create Bar Plot

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.barplot(x = 'Churn', y = 'total_ic_mou_6', data = telecom_hv)

plt.subplot(2,3,2)

sns.barplot(x = 'Churn', y = 'total_ic_mou_7', data = telecom_hv)

plt.subplot(2,3,3)

sns.barplot(x = 'Churn', y = 'total_ic_mou_8', data = telecom_hv)

#Create Bar Plot

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.barplot(x = 'Churn', y = 'onnet_mou_6', data = telecom_hv)

plt.subplot(2,3,2)

sns.barplot(x = 'Churn', y = 'onnet_mou_7', data = telecom_hv)

plt.subplot(2,3,3)

sns.barplot(x = 'Churn', y = 'onnet_mou_8', data = telecom_hv)

#Create Bar Plot

plt.figure(figsize=(20, 12))

plt.subplot(2,3,1)

sns.barplot(x = 'Churn', y = 'offnet_mou_6', data = telecom_hv)

plt.subplot(2,3,2)

sns.barplot(x = 'Churn', y = 'offnet_mou_7', data = telecom_hv)

plt.subplot(2,3,3)

sns.barplot(x = 'Churn', y = 'offnet_mou_8', data = telecom_hv)

telecom_hv.shape
rech_data = telecom_hv.loc[:,telecom_hv.columns.str.contains('rech')]

tot_data = telecom_hv.loc[:,telecom_hv.columns.str.contains('tot')]

amt_data = telecom_hv.loc[:,telecom_hv.columns.str.contains('amt')]

ic_mou_data = telecom_hv.loc[:,(telecom_hv.columns.str.contains('ic') & telecom_hv.columns.str.contains('mou'))]

og_mou_data = telecom_hv.loc[:,(telecom_hv.columns.str.contains('og') & telecom_hv.columns.str.contains('mou'))]

net_mou_data = telecom_hv.loc[:,telecom_hv.columns.str.contains('net_mou')]

data3g = telecom_hv.loc[:,(telecom_hv.columns.str.contains('3g'))]

data2g = telecom_hv.loc[:,(telecom_hv.columns.str.contains('2g'))]
rech_data.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (25,25))

sns.heatmap(rech_data.corr(),annot = True)
tot_data.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (12,12))

sns.heatmap(tot_data.corr(),annot = True)
amt_data.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (10,10))

sns.heatmap(amt_data.corr(),annot = True)
#Create scatter plot to understand distribution of amounts

plt.figure(figsize=(25, 10))

plt.subplot(2,3,1)

sns.scatterplot(x = 'total_rech_amt_6', y = 'total_rech_amt_8', data = telecom_hv, hue = 'Churn')

plt.subplot(2,3,2)

sns.scatterplot(x = 'total_rech_amt_7', y = 'total_rech_amt_8', data = telecom_hv, hue = 'Churn')

plt.subplot(2,3,3)

sns.scatterplot(x = 'av_rech_amt', y = 'total_rech_amt_8', data = telecom_hv, hue = 'Churn')
ic_mou_data.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (36,36))

sns.heatmap(ic_mou_data.corr(),annot = True)
og_mou_data.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (40,40))

sns.heatmap(og_mou_data.corr(),annot = True)
net_mou_data.shape

#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (6,6))

sns.heatmap(net_mou_data.corr(),annot = True)
data3g.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (18,18))

sns.heatmap(data3g.corr(),annot = True)
data2g.shape
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (15,15))

sns.heatmap(data2g.corr(),annot = True)
check_cols = (telecom_hv[telecom_hv == 0].count(axis=0)/len(telecom_hv.index)*100)

check_cols = check_cols[check_cols > 75].index

check_cols
check_cols = check_cols[check_cols != 'Churn']
telecom_n = telecom_hv.select_dtypes(include=np.number)
telecom_n.head()
telecom_n.shape
# Columns with more than 70% missing values

colmns_missing_data = round(100*(telecom_n.isnull().sum()/len(telecom_n.index)), 2)

cols = colmns_missing_data[colmns_missing_data>1]
cols
telecom_cat = pd.DataFrame(telecom_n,columns = ['mobile_number','night_pck_user_6','night_pck_user_7','night_pck_user_8','fb_user_6','fb_user_7','fb_user_8'])

telecom_n.drop(['night_pck_user_6','night_pck_user_7','night_pck_user_8','fb_user_6','fb_user_7','fb_user_8'],axis=1,inplace=True)

telecom_cat.shape

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, verbose=0)

imp.fit(telecom_n)

imputed_df = imp.transform(telecom_n)
# Columns with more than 70% missing values

new_df = pd.DataFrame(imputed_df)

new_df.columns = telecom_n.columns

new_df.head()
telecom_n = pd.merge(new_df, telecom_cat, on='mobile_number', how='inner')

telecom_n.head()
telecom_nn = telecom_hv.select_dtypes(exclude=telecom_n.dtypes)
telecom_nn.head()
telecom_n['gp_avg_arpu'] = (telecom_n['arpu_6'] + telecom_n['arpu_7'])/2
telecom_n['gp_avg_arpu'] = np.where((telecom_n['arpu_8'] > 0) & (telecom_n['gp_avg_arpu'] == 0),telecom_n['arpu_8'],telecom_n['gp_avg_arpu'])                              
telecom_n.drop(['arpu_6','arpu_7'],axis=1,inplace=True)
telecom_n['total_og_mou_gp'] = (telecom_n['total_og_mou_6'] + telecom_n['total_og_mou_7'])/2
telecom_n['total_og_mou_gp'] = np.where((telecom_n['total_og_mou_8'] > 0) & (telecom_n['total_og_mou_gp'] == 0),telecom_n['total_ic_mou_8'],telecom_n['total_og_mou_gp'])                              
telecom_n.drop(['total_og_mou_6','total_og_mou_7'],axis=1,inplace=True)
telecom_n['total_ic_mou_gp'] = (telecom_n['total_ic_mou_6'] + telecom_n['total_ic_mou_7'])/2
telecom_n['total_ic_mou_gp'] = np.where((telecom_n['total_ic_mou_8'] > 0) & (telecom_n['total_ic_mou_gp'] == 0),telecom_n['total_ic_mou_8'],telecom_n['total_ic_mou_gp'])                              
telecom_n.drop(['total_ic_mou_6','total_ic_mou_7'],axis=1,inplace=True)
telecom_n['onnet_mou_gp'] = (telecom_n['onnet_mou_6'] + telecom_n['onnet_mou_7'])/2

telecom_n['offnet_mou_gp'] = (telecom_n['offnet_mou_6'] + telecom_n['offnet_mou_7'])/2
telecom_n.drop(['onnet_mou_6','onnet_mou_7','offnet_mou_6','offnet_mou_7'],axis=1,inplace=True)
telecom_n.fillna(0,inplace=True)

telecom_n.shape
#telecom_n.dtypes

telecom_n['retain_factor_arpu'] = round(telecom_n['arpu_8'] / telecom_n['gp_avg_arpu'],2)

telecom_n['retain_factor_rech'] = round(telecom_n['total_rech_num_8'] / telecom_n['total_rech_num_7'],2)

telecom_n['retain_factor_rech'] = np.where(telecom_n['retain_factor_rech'] > 1,1,telecom_n['retain_factor_rech'])

telecom_n['retain_factor_arpu'] = np.where(telecom_n['retain_factor_arpu'] > 1,1,telecom_n['retain_factor_arpu'])
#Deduce a factor for retaining the customer

telecom_n['retain_factor'] = np.where((telecom_n['retain_factor_arpu'] > 0.5) & (telecom_n['retain_factor_rech'] > 0.5),0,1)

telecom_n.drop(columns = ['retain_factor_rech','retain_factor_arpu'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

# Assign feature variable to X

X = telecom_n.drop(['Churn','mobile_number'],axis=1)

# Assign response variable to y

y = telecom_n['Churn']

y.head()
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



scaler = preprocessing.StandardScaler().fit(X)

XScale = scaler.transform(X)
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(XScale,y, train_size=0.7,test_size=0.3,random_state=100)
X_train.shape
y_train_imb = (y_train != 0).sum()/(y_train == 0).sum()

y_test_imb = (y_test != 0).sum()/(y_test == 0).sum()

print("Imbalance in Train Data:", y_train_imb)

print("Imbalance in Test Data:", y_test_imb)
count_class = pd.value_counts(telecom_n['Churn'], sort=True)

count_class.plot(kind='bar',rot = 0)

plt.title('Churn Distribution')

plt.xlabel('Churn')
### Other Sampling Techniques just for playing around

#from imblearn.combine import SMOTETomek

#from imblearn.under_sampling import NearMiss

#smk = SMOTETomek(random_state = 42)

#X_trainb,y_trainb = smk.fit_sample(X_train,y_train)
### Other Sampling Techniques just for playing around

#from imblearn.over_sampling import RandomOverSampler

#os = RandomOverSampler(sampling_strategy=1)

#X_trainb,y_trainb = os.fit_sample(X_train,y_train)
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state = 2) 

X_trainb,y_trainb = smt.fit_sample(X_train,y_train)
X_trainb.shape
y_trainb.shape
#Improting the PCA module

from sklearn.decomposition import PCA

pca = PCA(svd_solver='randomized', random_state=42)
#Doing the PCA on the train data

pca.fit(X_trainb)
pca.components_
colnames = list(X.columns)

pcs_df = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':colnames})

pcs_df.head(10)
%matplotlib inline

fig = plt.figure(figsize = (20,20))

plt.scatter(pcs_df.PC1, pcs_df.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(pcs_df.Feature):

    plt.annotate(txt, (pcs_df.PC1[i],pcs_df.PC2[i]))

plt.tight_layout()

plt.show()
#Making the screeplot - plotting the cumulative variance against the number of components

%matplotlib inline

fig = plt.figure(figsize = (12,9))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
#Using incremental PCA for efficiency - saves a lot of time on larger datasets

from sklearn.decomposition import IncrementalPCA

pca_final = IncrementalPCA(n_components=50)
df_train_pca = pca_final.fit_transform(X_trainb)

df_train_pca.shape
#creating correlation matrix for the principal components

corrmat = np.corrcoef(df_train_pca.transpose())
#plotting the correlation matrix

%matplotlib inline

plt.figure(figsize = (50,30))

sns.heatmap(corrmat,annot = True)
# 1s -> 0s in diagonals

corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())

print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)

# we see that correlations are indeed very close to 0
#Applying selected components to the test data - 45 components

telecom_test_pca = pca_final.transform(X_test)

telecom_test_pca.shape
#Training the model on the train data

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



learner_pca = LogisticRegression()

model_pca = learner_pca.fit(df_train_pca,y_trainb)
#Making prediction on the test data

pred_probs_test = model_pca.predict_proba(telecom_test_pca)[:,1]

"{:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test))
# Predict Results from PCA Model

ypred_pca = model_pca.predict(telecom_test_pca)
# Confusion matrix 

confusion_PCA = metrics.confusion_matrix(y_test, ypred_pca)

print(confusion_PCA)
from sklearn.metrics import classification_report

print(classification_report(y_test, ypred_pca))
pca_again = PCA(0.90)
df_train_pca2 = pca_again.fit_transform(X_trainb)

df_train_pca2.shape

# we see that PCA selected 38 components
#training the regression model

learner_pca2 = LogisticRegression()

model_pca2 = learner_pca2.fit(df_train_pca2,y_trainb)
df_test_pca2 = pca_again.transform(X_test)

df_test_pca2.shape
#Making prediction on the test data

pred_probs_test2 = model_pca2.predict_proba(df_test_pca2)[:,1]

"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test2))
# Predict Results from PCA Model

ypred_pca2 = model_pca2.predict(df_test_pca2)
# Confusion matrix 

confusion_PCA = metrics.confusion_matrix(y_test, ypred_pca2)

print(confusion_PCA)
from sklearn.metrics import classification_report

print(classification_report(y_test, ypred_pca2))
pca_again = PCA(0.95)
df_train_pca3 = pca_again.fit_transform(X_trainb)

df_train_pca3.shape

# we see that PCA selected 51 components
#training the regression model

learner_pca3 = LogisticRegression()

model_pca3 = learner_pca3.fit(df_train_pca3,y_trainb)
df_test_pca3 = pca_again.transform(X_test)

df_test_pca3.shape
#Making prediction on the test data

pred_probs_test3 = model_pca3.predict_proba(df_test_pca3)[:,1]

"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test3))
# Predict Results from PCA Model

ypred_pca3 = model_pca3.predict(df_test_pca3)
# Confusion matrix 

confusion_PCA = metrics.confusion_matrix(y_test, ypred_pca3)

print(confusion_PCA)
from sklearn.metrics import classification_report

print(classification_report(y_test, ypred_pca3))
# Function to map the colors as a list from the input list of x variables

def pltcolor(lst):

    cols=[]

    for l in lst:

        if l==0:

            cols.append('red')

        elif l==1:

            cols.append('blue')

        else:

            cols.append('green')

    return cols

# Create the colors list using the function above

cols=pltcolor(y_trainb)


%matplotlib inline

fig = plt.figure(figsize = (12,10))

plt.scatter(df_train_pca[:,0], df_train_pca[:,1], s=200,c = cols)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.gray()

plt.show()
pca_last = PCA(n_components=10)

df_train_pca4 = pca_last.fit_transform(X_trainb)

df_test_pca4 = pca_last.transform(X_test)

df_test_pca4.shape
#training the regression model

learner_pca4 = LogisticRegression()

model_pca4 = learner_pca4.fit(df_train_pca4,y_trainb)

#Making prediction on the test data

pred_probs_test4 = model_pca4.predict_proba(df_test_pca4)[:,1]

"{:2.2f}".format(metrics.roc_auc_score(y_test, pred_probs_test4))
# Create a copy

telecom_LR_wPCA = telecom_n.copy()
telecom_LR_wPCA.shape
telecom_LR_wPCA['Churn'].value_counts()
plt.figure(figsize=(8,4))

telecom_LR_wPCA['Churn'].value_counts().plot(kind = 'bar')

plt.ylabel('Count')

plt.xlabel('Churn status')

plt.title('Churn status Distribution',fontsize=14)
# Create correlation matrix and check correlation greater than 0.95 adn drop those columns

corr_matrix = telecom_LR_wPCA.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(to_drop)
# Drop high correlated features

telecom_LR_wPCA.drop(telecom_LR_wPCA[to_drop], axis=1, inplace=True)
telecom_LR_wPCA.shape
#telecom_LR_wPCA.dtypes

telecom_LR_wPCA['retain_factor_arpu'] = round(telecom_LR_wPCA['arpu_8'] / telecom_LR_wPCA['gp_avg_arpu'],2)

telecom_LR_wPCA['retain_factor_rech'] = round(telecom_LR_wPCA['total_rech_num_8'] / telecom_LR_wPCA['total_rech_num_7'],2)

telecom_LR_wPCA['retain_factor_rech'] = np.where(telecom_LR_wPCA['retain_factor_rech'] > 1,1,telecom_LR_wPCA['retain_factor_rech'])

telecom_LR_wPCA['retain_factor_arpu'] = np.where(telecom_LR_wPCA['retain_factor_arpu'] > 1,1,telecom_LR_wPCA['retain_factor_arpu'])
#Deduce a factor for retaining the customer

telecom_LR_wPCA['retain_factor'] = np.where((telecom_LR_wPCA['retain_factor_arpu'] > 0.6) & (telecom_LR_wPCA['retain_factor_rech'] > 0.6),0,1)

telecom_LR_wPCA.drop(columns = ['retain_factor_rech','retain_factor_arpu'], axis=1, inplace=True)
telecom_LR_wPCA.retain_factor.describe()
# Assign feature variable to X

X = telecom_LR_wPCA.drop(['Churn','mobile_number'],axis=1)

X.head()
# Assign the response variable to y

y_LR = telecom_LR_wPCA[['Churn']]

y_LR.head()
# Splitting the data into train and test

X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X, y_LR, train_size=0.7, test_size=0.3, random_state=100)
X_train_LR.shape
smt = SMOTE(random_state = 2) 

X_train_LR,y_train_LR = smt.fit_sample(X_train_LR,y_train_LR)
X_train_LR.shape
data_imbalance = (y_train_LR != 0).sum()/(y_train_LR == 0).sum()

print("Imbalance in Train Data: {}".format(data_imbalance))
#X_train_LR.head()

columns = X.columns

X_train_LR = pd.DataFrame(X_train_LR)

X_train_LR.columns = columns

ycolumns = y_LR.columns

y_train_LR = pd.DataFrame(y_train_LR)

y_train_LR.columns = ycolumns
y_train_LR.shape
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_LR[columns] = scaler.fit_transform(X_train_LR[columns])

X_train_LR.retain_factor.describe()
X_train_LR.retain_factor.describe()
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
from sklearn.feature_selection import RFE

rfe = RFE(logreg, 45)             # running RFE with 38 variables as output

rfe = rfe.fit(X_train_LR, y_train_LR)
rfe.support_
list(zip(X_train_LR.columns, rfe.support_, rfe.ranking_))
col = X_train_LR.columns[rfe.support_]
X_train_sm = sm.add_constant(X_train_LR[col])

logm2 = sm.GLM(y_train_LR,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
###Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Churn':y_train_LR.Churn, 'Churn_Prob':y_train_pred})

y_train_pred_final['MobileNumber'] = y_train_LR.index

y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
from sklearn import metrics
# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )

print(confusion)
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train_LR[col].columns

vif['VIF'] = [variance_inflation_factor(X_train_LR[col].values, i) for i in range(X_train_LR[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#### There are a some variables with very high VIF. It's best to drop these variables as they aren't helping much with prediction and unnecessarily making the model complex.

#### Lets drop all variables that have very high VIF i.e. above 9

col = vif[vif['VIF'] < 9]

col = col.Features
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train_LR[col])

logm3 = sm.GLM(y_train_LR,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Churn_Prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))
vif = pd.DataFrame()

vif['Features'] = X_train_LR[col].columns

vif['VIF'] = [variance_inflation_factor(X_train_LR[col].values, i) for i in range(X_train_LR[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Let's re-run the model using the selected variables

X_train_sm = sm.add_constant(X_train_LR[col])

logm4 = sm.GLM(y_train_LR,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
y_train_pred[:10]
y_train_pred_final['Churn_Prob'] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))
from sklearn.metrics import classification_report

print(classification_report(y_train_pred_final.Churn, y_train_pred_final.predicted))
X_test_LR = X_test_LR[col]

X_test_LR.head()
X_test_sm = sm.add_constant(X_test_LR)
X_test_LR.shape
# Making predictions on the test set

y_test_pred = res.predict(X_test_sm)
y_test_pred.shape
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Assigning CustID to index

y_test_df['MobileNumber'] = y_test_df.index
# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()
# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Churn_Prob'})
# Rearranging the columns

#y_pred_final = y_pred_final.reindex_axis(['CustID','Churn','Churn_Prob'], axis=1)
y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_pred_final.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)
y_pred_final.shape
confusion2 = metrics.confusion_matrix(y_pred_final.Churn, y_pred_final.final_predicted )

confusion2
from sklearn.metrics import classification_report

print(classification_report(y_pred_final.Churn, y_pred_final.final_predicted))
# create a copy first

telecom_wPCA_RF = telecom_LR_wPCA.copy()
# Assign feature variable to X

X_RF = telecom_wPCA_RF.drop(['Churn','mobile_number'],axis=1)

X_RF.head()
# Assign response variable to y

y_RF = telecom_wPCA_RF['Churn']

y_RF.head()
# Splitting the data into train and test

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X_RF, y_RF, train_size=0.7, test_size=0.3, random_state=100)
smt = SMOTE(random_state = 2) 

X_train_RF,y_train_RF = smt.fit_sample(X_train_RF,y_train_RF)
X_train_RF.shape
X_train_RF = pd.DataFrame(X_train_RF)

X_train_RF.columns = X_RF.columns
y_train_RF.shape
from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel
numerics = ['int16','int32','int64','float16','float32','float64']

numerical_vars = list(X_train_RF.select_dtypes(include=numerics).columns)

X_train_RF = X_train_RF[numerical_vars]

X_train_RF.shape
Linear_SVC = LinearSVC(C=0.1, penalty="l1", dual=False).fit(X_train_RF, y_train_RF)

lasso_model = SelectFromModel(Linear_SVC, prefit=False)

lasso_model.fit(scaler.transform(X_train_RF.fillna(0)), y_train_RF)

lasso_model.get_support()
np.sum(lasso_model.estimator_.coef_ == 0)
deleted_vars = X_train_RF.columns[(lasso_model.estimator_.coef_ == 0).ravel().tolist()]

deleted_vars
#perform the same operation in the Test Data set for matching the columns

X_train_RF.drop(columns = deleted_vars,inplace=True,axis=1)

X_test_RF.drop(columns = deleted_vars,inplace=True,axis=1)

X_train_RF.shape
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier



# Running the random forest with default parameters.

rfc_d = RandomForestClassifier()
# fit

rfc_d.fit(X_train_RF,y_train_RF)
# Making predictions

predictions = rfc_d.predict(X_test_RF)
# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
# Let's check the report of our default model

print(classification_report(y_test_RF,predictions))
# Printing confusion matrix

print(confusion_matrix(y_test_RF,predictions))
print(accuracy_score(y_test_RF,predictions))
importances = list(rfc_d.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train_RF.columns, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# GridSearchCV to find optimal min_samples_leaf

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
# GridSearchCV to find optimal max_depth

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'max_depth': range(4, 10, 2)}



# instantiate the model

rf = RandomForestClassifier()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="accuracy", return_train_score=True)

rf.fit(X_train_RF, y_train_RF)
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

parameters = {'n_estimators': range(50, 200, 50)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestClassifier(max_depth=6)



# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="precision", return_train_score=True)

rf.fit(X_train_RF, y_train_RF)
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

parameters = {'max_features': [ 8, 12, 16, 20, 24]}



# instantiate the model

rf = RandomForestClassifier(max_depth = 6)



# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                    

                   scoring="accuracy", return_train_score=True)

rf.fit(X_train_RF, y_train_RF)
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




# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'min_samples_leaf': range(30, 200, 50)}



# instantiate the model

rf = RandomForestClassifier()



# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds,                   

                   scoring="accuracy", return_train_score=True)

rf.fit(X_train_RF, y_train_RF)
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
# model with the best hyperparameters

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(bootstrap=True,

                             max_depth=10,

                             min_samples_leaf=50, 

                             min_samples_split=200,

                             max_features=22,

                             n_estimators=100)
# fit

rfc.fit(X_train_RF,y_train_RF)
# predict

predictions = rfc.predict(X_test_RF)
# evaluation metrics

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_RF,predictions))
print(confusion_matrix(y_test_RF,predictions))
print(accuracy_score(y_test_RF,predictions))
importances = list(rfc.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_train_RF.columns, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
import xgboost as xgb
x_xgboost, y_xgboost = telecom_n.drop(['Churn'],axis=1),telecom_n[['Churn']]
#Create a matrix for identifying important predictors

data_dmatrix = xgb.DMatrix(data=x_xgboost,label=y_xgboost)
#separate the data into train and test

X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(x_xgboost, y_xgboost, test_size=0.3, random_state=123)
#Crate XGBoost classifer model

xg_class = xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
#Train and Predict based on the model

xg_class.fit(X_train_xg,y_train_xg)



preds = xg_class.predict(X_test_xg)
print(accuracy_score(y_test_xg,preds))
print(confusion_matrix(y_test_xg,preds))
params = {"objective":"reg:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,

                    num_boost_round=50,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)
cv_results.head()
#Perform KFold cross validation to obtain a better meausre of Accuracy

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(xg_class, x_xgboost, y_xgboost, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(classification_report(y_test_xg,preds))
xg_class1 = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
import matplotlib.pyplot as plt



xgb.plot_tree(xg_class1,num_trees=0)

plt.figure(figsize=(50,50))

plt.show()
xgb.plot_importance(xg_class1)

plt.rcParams['figure.figsize'] = [100, 100]

plt.show()