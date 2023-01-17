#for data visualization
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
%matplotlib inline
import missingno as mssno
seed=45

#for ML
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
#Reading the files
train=pd.read_csv("../input/insurance/train.csv",sep=',')
test=pd.read_csv("../input/insurance/test.csv",sep=',')
#Displaying Files
train.head()
#Displaying Files
test.head()
#Number of rows and columns
train.shape
#Description
train.describe()
#Data Types
train.info()
#Finding the nul values as null value is filled with -1
train1 = train.replace(-1, np.NaN)
test1 = test.replace(-1, np.NaN)
train1.isnull().sum()


#Heat map for Null Values
plt.figure(figsize=(16,10))
sns.heatmap(train1.head(100).isnull() == True, cmap='Set1')
#missing values visualization
#for train dataset
mssno.bar(train1,color='g',figsize=(16,5),fontsize=12)
#missing values test dataset
mssno.bar(test1,color='r',figsize=(16,5),fontsize=12)
#Categorizing the data as binary,categorical,interval and ordinal
binary = [c for c in train.columns if c.endswith("bin")]
categorical = [c for c in train.columns if c.endswith("cat")]
interval= [c for c in train.columns if (train[c].dtype == float)]
ordinal=[c for c in train.columns if not(c.endswith("bin")) and not(c.endswith("cat"))and (train[c].dtype != float)
         and(c!= "target") and (c!= "id")]
#Binary values visualization
plt.figure(figsize=(17,24))
for i, c in enumerate(binary):
    ax = plt.subplot(6,3,i+1)
    sns.countplot(train1[c],palette='rainbow')
#categorical varible visualization
plt.figure(figsize=(17,24))
for i, c in enumerate(categorical):
    ax = plt.subplot(6,3,i+1)
    sns.countplot(train1[c],palette='hls')
#Heatmap for interval variable
intercor = train[interval].corr()
plt.figure(figsize=(14,9))
sns.heatmap(intercor,annot=True)
plt.tight_layout()
interval



s = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
s = train.sample(frac=0.1)
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()

ordinal=[c for c in train.columns if not(c.endswith("bin")) and not(c.endswith("cat"))and (train[c].dtype != float)
         and(c!= "target") and (c!= "id")]
                                        
ordicor = train[ordinal].corr()
plt.figure(figsize=(17,10))
sns.heatmap(ordicor,annot=True)
plt.tight_layout()
ordinal
s = train.sample(frac=0.1)
sns.lmplot(x='ps_ind_14', y='ps_ind_15', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
s = train.sample(frac=0.1)
sns.lmplot(x='ps_ind_01', y='ps_ind_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})
plt.show()
#handling imbalanced classes(undersampling)
desired_apriori=0.10

# Get the indices per target value
idx_0 = train[train.target == 0].index
idx_1 = train[train.target == 1].index

# Get original number of records per target value
nb_0 = len(train.loc[idx_0])
nb_1 = len(train.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)
undersampled_nb_0 = int(undersampling_rate*nb_0)
print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train = train.loc[idx_list].reset_index(drop=True)
def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=104:
            df[i] = df[i].astype('category')
category_type(train)
category_type(test)
def OHE(df1,df2,column):
    cat_col = column
    #cat_col = df.select_dtypes(include =['category']).columns
    len_df1 = df1.shape[0]
    
    df = pd.concat([df1,df2],ignore_index=True)
    c2,c3 = [],{}
    
    print('Categorical feature',len(column))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2
categorical1 = [c for c in train.columns if c.endswith("cat")]

train1,test1 = OHE(train,test,categorical1)
mem = train1.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = test1.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))
def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['int64']).columns)
    for col in float_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)

change_datatype(train1)
change_datatype(test1)
def change_datatype_float(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
change_datatype_float(train1)
change_datatype_float(test1)
mem = train1.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
print('\n')
#--- memory consumed by test dataframe ---
mem = test1.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))
X = train1.drop(['target','id'],axis=1)
y = train1['target'].astype('category')
x_test = test1.drop(['target','id'],axis=1)
del train1,test1



kf = StratifiedKFold(n_splits=5,random_state=seed,shuffle=True)
pred_test_full=0
cv_score=[]
i=1
for train_index,test_index in kf.split(X,y):    
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    lr = LogisticRegression(class_weight='balanced',C=0.003)
    lr.fit(xtr, ytr)
    pred_test = lr.predict_proba(xvl)[:,1]
    score = roc_auc_score(yvl,pred_test)
    print('roc_auc_score',score)
    cv_score.append(score)
    pred_test_full += lr.predict_proba(x_test)[:,1]
    i+=1
print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))
print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))
proba = lr.predict_proba(xvl)[:,1]
fpr,tpr, threshold = roc_curve(yvl,proba)
auc_val = auc(fpr,tpr)

plt.figure(figsize=(16,10))
plt.title('Reciever Operating Charactaristics')
plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc_val)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
y_pred = pred_test_full/10
submit = pd.DataFrame({'id':test['id'],'target':y_pred}) 
submit.to_csv('lr_porto.csv',index=False) 
submit.head(10)
