# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import IPython

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('max_columns', None) #to display all the columns

X = pd.read_csv("../input/summeranalytics2020/train.csv")

X_test = pd.read_csv("../input/summeranalytics2020/test.csv")

X.head()
y = X['Attrition']

X.drop(['Attrition'], axis =1, inplace=True)
X.shape
from sklearn.model_selection import train_test_split

X.drop(columns='Id',inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=1)

#check the unique values in object columns to see whether to do hotencoding or labelencoding on them

obj_dict_train={obj_col: list(X_train[obj_col].unique()) for obj_col in X_train.select_dtypes(include='object')}

obj_dict_train
obj_dict_valid={obj_col: list(X_valid[obj_col].unique()) for obj_col in X_valid.select_dtypes(include='object')}

obj_dict_valid
label_col = ['BusinessTravel', 'OverTime']     # these two have ordinal categorical data

hotencode_col = list(set(X.select_dtypes(include='object').columns)-set(label_col))   
X_train = pd.concat([X_train, pd.get_dummies(X_train[hotencode_col])], axis=1)

X_valid = pd.concat([X_valid, pd.get_dummies(X_valid[hotencode_col])], axis=1)

X_train.drop(hotencode_col, axis=1, inplace=True)

X_valid.drop(hotencode_col,axis=1,inplace=True)



X_test = pd.concat([X_test, pd.get_dummies(X_test[hotencode_col])], axis=1)

X_test.drop(hotencode_col, axis=1, inplace=True)
#for ordinally encoding BussinessTravel column in train set

cat1 = pd.Categorical(X_train.BusinessTravel, categories=['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], ordered=True)

labels1, unique1 = pd.factorize(cat1, sort=True)

X_train['BusinessTravel'] = labels1



#for ordinally encoding OverTime column in train set

cat2 = pd.Categorical(X_train.OverTime, categories=['No','Yes'], ordered=True)

labels2, unique2 = pd.factorize(cat2, sort=True)

X_train['OverTime'] = labels2





#for ordinally encoding BussinessTravel column in validation set

cat_1 = pd.Categorical(X_valid.BusinessTravel, categories=['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], ordered=True)

labels_1, unique_1 = pd.factorize(cat_1, sort=True)

X_valid['BusinessTravel'] = labels_1



#for ordinally encoding OverTime column in validation set

cat_2 = pd.Categorical(X_valid.OverTime, categories=['No','Yes'], ordered=True)

labels_2, unique_2 = pd.factorize(cat_2, sort=True)

X_valid['OverTime'] = labels_2





#for ordinally encoding BussinessTravel column in test set

cat_test1 = pd.Categorical(X_test.BusinessTravel, categories=['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], ordered=True)

labels_test1, unique_test1 = pd.factorize(cat_test1, sort=True)

X_test['BusinessTravel'] = labels_test1



#for ordinally encoding OverTime column in test set

cat_test2 = pd.Categorical(X_test.OverTime, categories=['No','Yes'], ordered=True)

labels_test2, unique_test2 = pd.factorize(cat_test2, sort=True)

X_test['OverTime'] = labels_test2
# dropping useless columns

X_train.drop(columns =['Behaviour','Gender_Female'], inplace=True)

X_valid.drop(columns =['Behaviour','Gender_Female'], inplace=True)

X_test.drop(columns =['Behaviour','Gender_Female'], inplace=True)

# As all the columns now have numerical dataype we see the number of unique values to find 

#the columns with numerical data

X_train.nunique()
num_cols = [col for col in X_train.columns if X_train[col].nunique()>5]
from sklearn.preprocessing import StandardScaler

num_scaler = StandardScaler(copy=False)

scaled_train = pd.DataFrame(num_scaler.fit_transform(X_train[num_cols]))

scaled_valid = pd.DataFrame(num_scaler.transform(X_valid[num_cols]))

scaled_test = pd.DataFrame(num_scaler.transform(X_test[num_cols]))

scaled_test.columns  = num_cols

scaled_train.columns = num_cols

scaled_valid.columns = num_cols

scaled_train.index=X_train.index

scaled_valid.index=X_valid.index

scaled_test.index=X_test.index
X_train.drop(columns = num_cols, inplace=True)

X_valid.drop(columns = num_cols, inplace=True)

X_test.drop(columns = num_cols, inplace =True)

X_train = pd.concat([X_train, scaled_train], axis =1)

X_valid = pd.concat([X_valid, scaled_valid], axis =1)

X_test = pd.concat([X_test, scaled_test], axis =1)
%matplotlib inline

matplotlib.style.use('ggplot')

sns.set_style('white')
#preparing dataset for plotting

data_count = pd.concat([X_train.drop(columns = num_cols), y], axis=1)
fig, axes = plt.subplots(round((len(data_count.columns)-1) / 3), 3, figsize=(12, 25))



for i, ax in enumerate(fig.axes):

    if i <= len(data_count.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=data_count.columns[i], hue=data_count.Attrition, alpha=0.7, data=data_count, ax=ax)



fig.tight_layout()
X_train[num_cols].hist(figsize=(14,10),xlabelsize=8, ylabelsize=8);
for col in num_cols:

    plt.figure(figsize=(15,3))

    sns.scatterplot(x=X[col],y=y)

    plt.show()
data_pair = pd.concat([X_train[num_cols],y_train], axis=1)
corr_matrix = data_pair.corr()

cmap = sns.diverging_palette(300, 0, as_cmap=True)

plt.figure(figsize=(11,11))

sns.heatmap(corr_matrix, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
rows = round(len(num_cols)*(len(num_cols)-1)/6)

fig, axes = plt.subplots(rows, 3, figsize=(16, 60))

i=0

j=i+1

for __, ax in enumerate(fig.axes):

    if i==len(num_cols): break

    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

    sns.scatterplot(data=data_pair, x=data_pair.columns[i], y=data_pair.columns[j], hue='Attrition', alpha=0.7, ax=ax)

    j+=1

    if j==len(num_cols): 

        i+=1

        j=i+1

fig.tight_layout()
X_train['stock0'] = (X_train.StockOptionLevel==0).astype(int)

X_valid['stock0'] = (X_valid.StockOptionLevel==0).astype(int)

X_test['stock0'] = (X_test.StockOptionLevel==0).astype(int)



X_train['jobsatisfied'] = (X_train.JobSatisfaction==4).astype(int)

X_valid['jobsatisfied'] = (X_valid.JobSatisfaction==4).astype(int)

X_test['jobsatisfied'] = (X_test.JobSatisfaction==4).astype(int)



X_train['well_communicate'] = (X_train.CommunicationSkill==5).astype(int)

X_valid['well_communicate'] = (X_valid.CommunicationSkill==5).astype(int)

X_test['well_communicate'] = (X_test.CommunicationSkill==5).astype(int)



X_train['less_mi'] = (X_train.MonthlyIncome<0.2).astype(int)

X_valid['less_mi'] = (X_valid.MonthlyIncome<0.2).astype(int)

X_test['less_mi'] = (X_test.MonthlyIncome<0.2).astype(int)
X_train['mi_age']=X_train.MonthlyIncome*X_train.Age

X_valid['mi_age']=X_valid.MonthlyIncome*X_valid.Age

X_test['mi_age']=X_test.MonthlyIncome*X_test.Age



X_train['twy_age']=X_train.TotalWorkingYears*X_train.Age

X_valid['twy_age']=X_valid.TotalWorkingYears*X_valid.Age

X_test['twy_age']=X_test.TotalWorkingYears*X_test.Age



X_train['yac_age']=np.square(X_train.YearsAtCompany)/0.7+np.square(X_train.Age)

X_valid['yac_age']=np.square(X_valid.YearsAtCompany)/0.7+np.square(X_valid.Age)

X_test['yac_age']=np.square(X_test.YearsAtCompany)/0.7+np.square(X_test.Age)



X_train['mi_dfh']=np.square(X_train.MonthlyIncome)/0.2-np.square(X_train.DistanceFromHome)

X_valid['mi_dfh']=np.square(X_valid.MonthlyIncome)/0.2-np.square(X_valid.DistanceFromHome)

X_test['mi_dfh']=np.square(X_test.MonthlyIncome)/0.2-np.square(X_test.DistanceFromHome)



X_train['yac_dfh']=np.square(X_train.YearsAtCompany)/0.5-np.square(X_train.DistanceFromHome)

X_valid['yac_dfh']=np.square(X_valid.YearsAtCompany)/0.5-np.square(X_valid.DistanceFromHome)

X_test['yac_dfh']=np.square(X_test.YearsAtCompany)/0.5-np.square(X_test.DistanceFromHome)



X_train['mi_twc']=X_train.MonthlyIncome*X_train.TotalWorkingYears

X_valid['mi_twc']=X_valid.MonthlyIncome*X_valid.TotalWorkingYears

X_test['mi_twc']=X_test.MonthlyIncome*X_test.TotalWorkingYears



X_train['mi_yac']=X_train.MonthlyIncome*X_train.YearsAtCompany

X_valid['mi_yac']=X_valid.MonthlyIncome*X_valid.YearsAtCompany

X_test['mi_yac']=X_test.MonthlyIncome*X_test.YearsAtCompany



X_train['twy_ywcm']=np.square(X_train.TotalWorkingYears)/0.6-np.square(X_train.YearsWithCurrManager)/8

X_valid['twy_ywcm']=np.square(X_valid.TotalWorkingYears)/0.6-np.square(X_valid.YearsWithCurrManager)/8

X_test['twy_ywcm']=np.square(X_test.TotalWorkingYears)/0.6-np.square(X_test.YearsWithCurrManager)/8
# scaling the newly formed features

new_cols =['mi_age','twy_age','yac_age','mi_dfh','yac_dfh','mi_twc','mi_yac','twy_ywcm']

scaled_train1 = pd.DataFrame(num_scaler.fit_transform(X_train[new_cols]))

scaled_valid1 = pd.DataFrame(num_scaler.transform(X_valid[new_cols]))

scaled_test1 = pd.DataFrame(num_scaler.transform(X_test[new_cols]))

scaled_test1.columns = new_cols

scaled_train1.columns = new_cols

scaled_valid1.columns = new_cols

scaled_train1.index=X_train.index

scaled_valid1.index=X_valid.index

scaled_test1.index=X_test.index



X_train.drop(columns = new_cols, inplace=True)

X_valid.drop(columns = new_cols, inplace=True)

X_test.drop(columns = new_cols, inplace =True)

X_train = pd.concat([X_train, scaled_train1], axis =1)

X_valid = pd.concat([X_valid, scaled_valid1], axis =1)

X_test = pd.concat([X_test, scaled_test1], axis =1)
num_cols2 = [col for col in X_train.columns if X_train[col].nunique()>10]
# using boxplot to check for outliers and skewedness

plt.figure(figsize=(15,5))

ax = sns.boxplot(data=X_train[num_cols2])

plt.xticks(rotation=90)

plt.show()
X_train['twy_ywcm'] = np.log(2+X_train.twy_ywcm/2)

X_train['mi_yac'] = np.log(3+X_train.mi_yac)

X_train['mi_twc'] = np.log(3+X_train.mi_twc)

X_train['yac_dfh'] = np.log(2+X_train.yac_dfh)

X_train['mi_dfh'] = np.log(2+X_train.mi_dfh)

X_train['yac_age'] = np.log(2+X_train.yac_age)

X_train['twy_age'] = np.sqrt(3+X_train.twy_age)

X_train['mi_age'] = np.sqrt(3+X_train.mi_age)

X_train['YearsSinceLastPromotion'] = np.log(1.5+X_train.YearsSinceLastPromotion/2)

X_train['MonthlyIncome'] = np.log(2+X_train.MonthlyIncome)

X_train['TotalWorkingYears'] = np.log(2+X_train.TotalWorkingYears)

X_train['YearsAtCompany'] = np.sqrt(3+X_train.YearsAtCompany/5)
# again plot the boxplot to see the difference

plt.figure(figsize=(15,5))

ax = sns.boxplot(data=X_train[num_cols2])

plt.xticks(rotation=90)

plt.show()
# creating same features for validation set

X_valid['twy_ywcm'] = np.log(2+X_valid.twy_ywcm/2)

X_valid['mi_yac'] = np.log(3+X_valid.mi_yac)

X_valid['mi_twc'] = np.log(3+X_valid.mi_twc)

X_valid['yac_dfh'] = np.log(2+X_valid.yac_dfh)

X_valid['mi_dfh'] = np.log(2+X_valid.mi_dfh)

X_valid['yac_age'] = np.log(2+X_valid.yac_age)

X_valid['twy_age'] = np.sqrt(3+X_valid.twy_age)

X_valid['mi_age'] = np.sqrt(3+X_valid.mi_age)

X_valid['YearsSinceLastPromotion'] = np.log(1.5+X_valid.YearsSinceLastPromotion/2)

X_valid['MonthlyIncome'] = np.log(2+X_valid.MonthlyIncome)

X_valid['TotalWorkingYears'] = np.log(2+X_valid.TotalWorkingYears)

X_valid['YearsAtCompany'] = np.sqrt(3+X_valid.YearsAtCompany/5)
# creating same features for test set

X_test['twy_ywcm'] = np.log(2+X_test.twy_ywcm/2)

X_test['mi_yac'] = np.log(3+X_test.mi_yac)

X_test['mi_twc'] = np.log(3+X_test.mi_twc)

X_test['yac_dfh'] = np.log(2+X_test.yac_dfh)

X_test['mi_dfh'] = np.log(2+X_test.mi_dfh)

X_test['yac_age'] = np.log(2+X_test.yac_age)

X_test['twy_age'] = np.sqrt(3+X_test.twy_age)

X_test['mi_age'] = np.sqrt(3+X_test.mi_age)

X_test['YearsSinceLastPromotion'] = np.log(1.5+X_test.YearsSinceLastPromotion/2)

X_test['MonthlyIncome'] = np.log(2+X_test.MonthlyIncome)

X_test['TotalWorkingYears'] = np.log(2+X_test.TotalWorkingYears)

X_test['YearsAtCompany'] = np.sqrt(3+X_test.YearsAtCompany/5)
X_train_final = pd.concat([X_train,X_valid],axis=0)

y_train_final = pd.concat([y_train,y_valid],axis=0)

X_train_final.reset_index(drop=True)

y_train_final.reset_index(drop=True)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,accuracy_score
num_iters=np.arange(200,501,50)

C = np.arange(0.1,3,0.1, dtype=np.float32)

arg=[]

for i in C:

    for j in num_iters: arg.append((j,i))
# model for checking the validation score

def model(arg):

    num_iters = arg[0]

    C=arg[1]

    lr=LogisticRegression(solver='liblinear', max_iter=num_iters,C=C, verbose=0, random_state=7)

    lr.fit(X_train,y_train)

    predictions = lr.predict(X_valid)

    return roc_auc_score(y_valid,predictions)
for params in arg:

    score = model(params)

    print("C={}, iterations={} : {}".format(params[1],params[0],score))



lr_final = LogisticRegression( solver='liblinear',C=1.8, max_iter=300,random_state=7)

lr_final.fit(X_train,y_train)

pred_lr = lr_final.predict_proba(X_valid)[:,1]

roc_auc_score(y_valid,pred_lr)
feature_weights = np.array(lr_final.coef_).reshape(-1)
feature_val = pd.DataFrame({'features':X_train.columns, 'weights':feature_weights})

feature_val.sort_values(by='weights', ascending=False, inplace=True)

plt.figure(figsize=(14,16))

sns.barplot(y='features',x='weights',data=feature_val,color='blue')

plt.show()
#Columns which have absolute value of weights less than 0.1

less_weight_cols= feature_val[abs(feature_val.weights)<0.1].features.to_list()

less_weight_cols
#---------------

X_train_final.drop(columns=less_weight_cols,inplace=True)



X_test.drop(columns=less_weight_cols,inplace=True)
lr_final.fit(X_train_final,y_train_final)

lr_final
final_pred = lr_final.predict_proba(X_test.drop(columns='Id'))[:,1]

output = pd.concat([X_test.Id,pd.Series(final_pred)], axis=1)

output.columns=['Id','Attrition']

output.set_index('Id')

output.to_csv("submission.csv",index=False)