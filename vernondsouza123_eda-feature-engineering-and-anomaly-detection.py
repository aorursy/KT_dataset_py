import numpy as np 



import pandas as pd 



import os



import seaborn as sns



from scipy.stats import chi2_contingency

from scipy.stats import chisquare

import scipy.stats as ss

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform



import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing 

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.model_selection import KFold



import datetime



from lightgbm import LGBMClassifier





## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
df_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")



# Check for Missing  data

total = df_transaction.isnull().sum()

print("NaN values in Transaction database",total)



# Unique Values

UniqueID = df_transaction['TransactionID'].nunique()

print("No of unique Transaction ID's",UniqueID)



# Columns in dataframe

print("Columns in dataframe",df_transaction.columns)



# Columns datatypes

string_columns = df_transaction.select_dtypes('object').columns.tolist()

print("Object Columns",string_columns)



numeric_columns = df_transaction.select_dtypes(include=np.number).columns.tolist()

print("Numerical columns ",numeric_columns)
# Memory Reduction

df_transaction = reduce_mem_usage(df_transaction, verbose=True)

print(df_transaction.head(10))
# Check for Missing Data



df_transaction = df_transaction.replace(np.nan, 'NaN', regex=True)



lst = string_columns

data = df_transaction[df_transaction.columns.intersection(lst)]



# Check for M variables since they have a high number of NaN vales

M = data.filter(regex='^M',axis=1)

M.head()
columns = M.columns



#df_trans = df_transaction.copy()



for col in columns:   

    

        

    df_trans = df_transaction[df_transaction.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

    

    

    

del df_trans 

# Columns ProductCD, card4, card6

columns = ['ProductCD', 'card4', 'card6']



for col in columns:

    

    df_trans = df_transaction[df_transaction.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1
# Columns other than M

columns = ['P_emaildomain', 'R_emaildomain']





for col in columns:

    

    df_trans = df_transaction[df_transaction.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    
# Cramers V for categorical correlations

def cramers_v(x, y):

    x = np.array(x)

    y = np.array(y)

    confusion_matrix = pd.crosstab(x, y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))





cramersv = pd.DataFrame(index=data.columns,columns=data.columns)

columns = data.columns



for i in range(0,len(columns)):

    for j in range(0,len(columns)):

        #print(data[columns[i]].tolist())

        u = cramers_v(data[columns[i]].tolist(),data[columns[j]].tolist())

        cramersv.loc[columns[i],columns[j]] = u

        

cramersv.fillna(value=np.nan,inplace=True)
plt.figure(figsize=(15,15))

sns.heatmap(cramersv, annot=True)

plt.show()
# Reduce Memory

del cramersv,data
df_transaction.drop(columns=M.columns,inplace=True)

print(df_transaction.head(10))

del M
print(df_transaction.columns)
# Too many V variables. We can do a correlation analysis and remove columns that are highly correlated

V = df_transaction.filter(regex='^V',axis=1)



print(len(V))
# Create correlation matrix



V = V.applymap(float)

corr_matrix = V.corr().abs()





# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



print(to_drop)

HighCorrVDrop = to_drop



del corr_matrix, upper
# Let's look at other columns

df_transaction.drop(columns = V.columns,inplace=True)

del V
# Let's look at transaction amt



df_transaction['TransactionAmt'] = df_transaction['TransactionAmt'].apply(pd.to_numeric)

f, ax = plt.subplots(1, 1,figsize=(15,15))



sns.distplot(ax= ax,a = df_transaction[df_transaction['isFraud']==0]['TransactionAmt'], color="skyblue", label="is Not Fraud")





sns.distplot(ax=ax,a = df_transaction[df_transaction['isFraud']==1]['TransactionAmt'], color="red", label="is Fraud")

plt.legend(labels=['is Not Fraud', 'is Fraud'])

ax.set_xlabel("Transaction Amt")



plt.show()



# Data heavily skewed. Apply log of the data



f, ax = plt.subplots(1, 1,figsize=(15,15))



sns.distplot(ax= ax,a = np.log(df_transaction[df_transaction['isFraud']==0]['TransactionAmt']), color="skyblue", label="is Not Fraud")





sns.distplot(ax=ax,a = np.log(df_transaction[df_transaction['isFraud']==1]['TransactionAmt']), color="red", label="is Fraud")

plt.legend(labels=['is Not Fraud', 'is Fraud'])

ax.set_xlabel("Log of Transaction Amt")

plt.show()
# Find which of the columns have the highest frquency among fake transactions



df_trans = df_transaction[df_transaction['isFraud']==1]

df_tran = df_transaction.copy()

C = df_transaction.filter(regex='^C',axis=1)







for col in C.columns:

   

    index_list = df_trans[col].value_counts(ascending=False)[:15].index.to_list()

    df_tran.loc[df_tran[(~df_tran[col].isin(index_list))].index, col] = "Others"

    

    

del df_trans
columns = C.columns





for col in columns:

    

    df_trans = df_tran[df_tran.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

del C
# Find which of the columns have the highest frquency among fake transactions

# Due to high presence of many features. Let's pick only the top features by frequency for plotting



df_trans = df_transaction[df_transaction['isFraud']==1]

df_tran = df_transaction.copy()

D = df_transaction.filter(regex='^D',axis=1)







for col in D.columns:

   

    index_list = df_trans[col].value_counts(ascending=False)[:15].index.to_list()

    df_tran.loc[df_tran[(~df_tran[col].isin(index_list))].index, col] = "Others"

    

    

del df_trans
columns = D.columns





for col in columns:

    

    df_trans = df_tran[df_tran.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

del D

# Find which of the columns have the highest frquency among fake transactions

# Due to high presence of many features. Let's pick only the top features by frquency for plotting



df_trans = df_transaction[df_transaction['isFraud']==1]

df_tran = df_transaction.copy()

D = df_transaction[['addr1','addr2','dist1','dist2']]







for col in D.columns:

   

    index_list = df_trans[col].value_counts(ascending=False)[:15].index.to_list()

    df_tran.loc[df_tran[(~df_tran[col].isin(index_list))].index, col] = "Others"

    

    

del df_trans
columns = D.columns



for col in columns:

    

    df_trans = df_tran[df_tran.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

del D
# Find which of the columns have the highest frquency among fake transactions

# Due to high presence of many features. Let's pick only the top features by frquency for plotting



df_trans = df_transaction[df_transaction['isFraud']==1]

df_tran = df_transaction.copy()

D = df_transaction[['card1','card2','card3','card5']]







for col in D.columns:

   

    index_list = df_trans[col].value_counts(ascending=False)[:15].index.to_list()

    df_tran.loc[df_tran[(~df_tran[col].isin(index_list))].index, col] = "Others"

    

    

del df_trans
columns = D.columns



for col in columns:

    

    df_trans = df_tran[df_tran.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

del D
START_DATE = '2017-12-01'

startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

df_transaction["date"] = df_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))



df_transaction['Weekdays'] = df_transaction['date'].dt.dayofweek

df_transaction['Hours'] = df_transaction['date'].dt.hour

df_transaction['Days'] = df_transaction['date'].dt.day



columns = ['Weekdays','Hours','Days']



for col in columns:

    

    df_trans = df_transaction[df_transaction.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_trans)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

truncated_df = df_transaction[['TransactionID','isFraud']]

del df_transaction
df_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
# Name of columns and null values among them



print(df_identity.columns)



print("NaN values : ",df_identity.isna().sum())
# Let's look at the data in the ID subset folder



ID = df_identity.filter(regex='^id',axis=1)

ID.head()
print(ID.iloc[:,0:10])
print(ID.iloc[:,10:20])
print(ID.iloc[:,20:30])
print(ID.iloc[:,30:40])
# Merge the targets and identity database for plotting purpose



df_identity = pd.merge(truncated_df,df_identity, how='left',on='TransactionID')

df_identity.reset_index(inplace=True)
columns = ['id_34','id_35','id_36','id_37','id_38']



for col in columns:

    

    df_iden = df_identity[df_identity.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_iden)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    
columns = ['id_12','id_15','id_16','id_28','id_29']



for col in columns:

    

    df_iden = df_identity[df_identity.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_iden)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

 

    
# ID columns with high number of NaN values



columns =['id_07', 'id_08', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27']





for col in columns:

    

    df_iden = df_identity[df_identity.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_iden)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1
# Further analysis of id_23



  

plt.figure(figsize=(14,10))

    

g1 = sns.countplot(x='id_23',hue='isFraud',data=df_identity)

    

g1.set_title("Frequeny of " + "id_23" + " by Fraud Values", fontsize=19)

 

g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

plt.show()



del g1
# Find which of the columns have the highest frquency among fake transactions

# Due to high presence of many features. Let's pick only the top features by frquency for plotting



df_ident =df_identity[df_identity['isFraud']==1]

df_iden = df_identity.copy()

D = df_identity[['id_30','id_31','id_33','DeviceInfo']]







for col in D.columns:

   

    index_list = df_ident[col].value_counts(ascending=False)[:15].index.to_list()

    df_iden.loc[df_iden[(~df_iden[col].isin(index_list))].index, col] = "Others"

    

    

del df_ident
columns = D.columns





for col in columns:

    

    df_ident = df_iden[df_iden.isFraud==1][col]  

    

    

    plt.figure(figsize=(14,10))

    

    g1 = sns.countplot(x=df_ident)

    

    g1.set_title("Frequeny of " + str(col)+ " by Fraud Values", fontsize=19)

 

    g1.set_xticklabels(g1.get_xticklabels(),rotation=45)

    plt.show()

  

    del g1

    

del df_iden



  

df_trans = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

df_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
# Data Merging

df = pd.merge(df_trans,df_identity, how='left',on='TransactionID')

df.reset_index(inplace=True)

del df_trans,df_identity
df.drop(columns=['index'],inplace=True)
def pick_high_freq_only(df,df2 = pd.DataFrame(),second=False):

    

    # Find which of the columns have the highest frquency among fake transactions

    

    # Device Info

    d = df[df['isFraud']==1]

    dnf = df[df['isFraud']==0]

    columns  = ['DeviceInfo']





    for col in columns:

   

        index_list1 = d[col].value_counts(ascending=False)[:100].index.to_list()

        index_list2 = dnf[col].value_counts(ascending=False)[:100].index.to_list()

    

        index_list = index_list1 + index_list2

        df.loc[df[(~df[col].isin(index_list))].index, col] = "Others"

    

        if second == True:

            df2.loc[df2[(~df2[col].isin(index_list))].index, col] = "Others"

    

    

    del d,dnf

    

    # Email Domains

    d = df[df['isFraud']==1]

    dnf = df[df['isFraud']==0]

    columns  = ['P_emaildomain', 'R_emaildomain']





    for col in columns:

   

        index_list1 = d[col].value_counts(ascending=False)[:20].index.to_list()

        index_list2 = dnf[col].value_counts(ascending=False)[:20].index.to_list()

    

        index_list = index_list1 + index_list2

        df.loc[df[(~df[col].isin(index_list))].index, col] = "Others"

        

        if second == True:

            df2.loc[df2[(~df2[col].isin(index_list))].index, col] = "Others"

            

    

    

    

    del d,dnf

    

    # Selected ID's

    d = df[df['isFraud']==1]

    dnf = df[df['isFraud']==0]

    columns = ['id_30','id_31','id_33']



    for col in columns:

   

        index_list1 = d[col].value_counts(ascending=False)[:100].index.to_list()

        index_list2 = dnf[col].value_counts(ascending=False)[:100].index.to_list()

    

        index_list = index_list1 + index_list2

        df.loc[df[(~df[col].isin(index_list))].index, col] = "Others"

        

        if second == True:

            df2.loc[df2[(~df2[col].isin(index_list))].index, col] = "Others"

    

    

    del d,dnf

    

    return df,df2









df,_ = pick_high_freq_only(df)

    

#print(df)  
# Seggregate into features and target



Features = df

Features = Features.drop(columns=['isFraud','TransactionID'])

Target = df['isFraud'].astype(float)

Target = Target.values

Target = Target.reshape((len(Target), 1))



del df
def FETransactionDT(Features):

    

    #print(Features)

    START_DATE = '2017-12-01'

    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

    Features["date"] = Features['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))



    Features['Weekdays'] = Features['date'].dt.dayofweek

    Features['Hours'] = Features['date'].dt.hour

    Features['Days'] = Features['date'].dt.day



    Features['Weekdays'] = pd.to_numeric(Features['Weekdays'])

    Features['Hours'] = pd.to_numeric(Features['Hours'])

    Features['Days'] = pd.to_numeric(Features['Days'])



    Features.drop(columns=["date"],inplace=True)

    

    return Features

    

Features = FETransactionDT(Features)

def HighNaNcol(Features):

    

    HighNaN = []

    for col in Features.columns:

    

        no = Features[col].isna().sum()

    

        if no > 0.95 * len(Features):

            HighNaN.append(col)

            

    print(HighNaN)

    return HighNaN



#HighNaN = HighNaNcol(Features)

# Features to drop 

columns_to_drop = HighCorrVDrop





#Features.drop(columns=HighNaN,inplace=True)

Features.drop(columns = columns_to_drop,inplace=True)
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=test_size,random_state=32)
def ScaleColumns(X_train,X_test):

    for col in X_train.columns:

        if X_train[col].dtype=='object' or X_test[col].dtype=='object': 

            print(col)

            lbl = preprocessing.LabelEncoder()

        

            X_train[col] = X_train[col].astype(str)

            X_test[col] = X_test[col].astype(str)

        

            X_train[col] = lbl.fit_transform(X_train[col])

            X_test[col] = lbl.transform(X_test[col])  

        

        if col == 'TransactionAmt':

            

            print(col)

            SS = preprocessing.StandardScaler()

        

            X_train_arr = X_train[col].astype(float).values

            X_test_arr = X_test[col].astype(float).values   

        

            

            X_train_arr = SS.fit_transform(X_train_arr.reshape(-1,1))

            X_test_arr = SS.transform(X_test_arr.reshape(-1,1))

            

            X_train[col]  = X_train_arr 

            X_test[col]   = X_test_arr

            

    return X_train, X_test

            

X_train, X_test = ScaleColumns(X_train,X_test)


#fit_params={"early_stopping_rounds":50, 

           # "eval_metric" : 'auc', 

           #"eval_set" : [(X_test,y_test)],

           #'eval_names': ['valid'],

           # 'verbose': 100,

           #'categorical_feature': 'auto'}



#param_test ={  'n_estimators': [400, 700, 1000],

  #'colsample_bytree': [0.7, 0.8],

  # 'max_depth': [15,20,25],

   #'num_leaves': [50, 100, 200],

  # 'reg_alpha': [1.1, 1.2, 1.3],

 # 'reg_lambda': [1.1, 1.2, 1.3],

 # 'min_split_gain': [0.3, 0.4],

 #'subsample': [0.7, 0.8, 0.9],

  # 'subsample_freq': [20]}
#clf = LGBMClassifier(random_state=314, silent=True, metric='None', n_jobs=2)

#model = RandomizedSearchCV(

  # estimator=clf, param_distributions=param_test, 

   # scoring='roc_auc',

   # cv=3,

    #refit=True,

   # random_state=314,

   # verbose=True)
#model.fit(X_train, y_train, **fit_params)

#print('Best score reached: {} with params: {} '.format(model.best_score_, model.best_params_))


model = LGBMClassifier(subsample_freq= 20,subsample= 0.7, reg_lambda= 1.2, reg_alpha = 1.1, 

                       num_leaves = 200, n_estimators = 700, 

                       min_split_gain = 0.3, max_depth = 25, colsample_bytre = 0.8,random_state=314,n_jobs=2)



model.fit(X_train, y_train)
# make predictions for test data

y_pred = model.predict_proba(X_test)
# evaluate predictions

score = roc_auc_score(y_test, y_pred[:,1])

print("ROC_AUC score: %.2f%%" % (score * 100.0))
df_train_trans = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

df_train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")



# Data Merging

df_train = pd.merge(df_train_trans,df_train_identity , how='left',on='TransactionID')

df_train.reset_index(inplace=True)

del df_train_trans,df_train_identity
df_test_trans = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

df_test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")



# Data Merging

df_test = pd.merge(df_test_trans,df_test_identity , how='left',on='TransactionID')

df_test.reset_index(inplace=True)

del df_test_trans,df_test_identity
df_train.drop(columns=['index'],inplace=True)

df_test.drop(columns=['index'],inplace=True)
df_test.columns = df_test.columns.str.replace("-", "_")

print(df_test.columns)
df_train,df_test = pick_high_freq_only(df_train,df_test,second=True)
# Seggregate into features and target



Features_train = df_train

Features_train = Features_train.drop(columns=['isFraud'])

Target_train = df_train['isFraud']

Target_train = Target_train.values

Target_train = Target_train.reshape((len(Target_train), 1))



del df_train
Features_test = df_test



del df_test
Features_train = FETransactionDT(Features_train)

Features_test = FETransactionDT(Features_test)
HighNaN = HighNaNcol(Features_train)
# Features to drop 

columns_to_drop = HighCorrVDrop



#Features_train.drop(columns=HighNaN,inplace=True)

Features_train.drop(columns = columns_to_drop,inplace=True)



#Features_test.drop(columns=HighNaN,inplace=True)

Features_test.drop(columns = columns_to_drop,inplace=True)
Features_train, Features_test = ScaleColumns(Features_train, Features_test)
Features_train = Features_train.drop(columns=['TransactionID'])

Features_test_without_ID = Features_test.drop(columns=['TransactionID'])


model = LGBMClassifier(subsample_freq= 20,subsample= 0.7, reg_lambda= 1.2, reg_alpha = 1.1, 

                       num_leaves = 200, n_estimators = 700, 

                       min_split_gain = 0.3, max_depth = 25, colsample_bytre = 0.8,random_state=314,n_jobs=2)



model.fit(Features_train, Target_train)
# make predictions for test data

y_pred = model.predict_proba(Features_test_without_ID)

print(y_pred)
submission = pd.DataFrame()

submission['TransactionID'] = Features_test['TransactionID']

submission['isFraud'] = y_pred[:,1]



print(submission)

submission.to_csv('submission.csv', index=False)
def plotImp(model, X , num = 20):

    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})

    plt.figure(figsize=(40, 20))

    sns.set(font_scale = 5)

    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 

                                                        ascending=False)[0:num])

    plt.title('LightGBM Feature imprtances')

    plt.tight_layout()

    

    plt.show()

    

plotImp(model, Features_train , num =20)