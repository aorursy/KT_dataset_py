#Impoting all the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.decomposition import FactorAnalysis

from sklearn import preprocessing

from sklearn.utils import resample

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

import os

df1=pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/train_data.csv")

df2=pd.read_csv("../input/av-healthcare-analytics-ii/healthcare/test_data.csv")
le = preprocessing.LabelEncoder()

df1.drop(['case_id'],axis=1,inplace=True)

df2.drop(['case_id'],axis=1,inplace=True)
df1.describe()
df1.isna().sum(),df2.isna().sum()
cor=df1.corr()

plt.figure(figsize=(15,10))

sns.heatmap(cor,vmax=.3, square=True,annot=True)

df1['Hospital_code'].nunique()
df1['Hospital_code'].isna().sum()
df1['Hospital_type_code'].value_counts()
dummy_variable_1 = pd.get_dummies(df1["Hospital_type_code"])

df1 = pd.concat([df1, dummy_variable_1], axis=1)
type_list=dummy_variable_1.columns.tolist()
for type in type_list:

         df1.rename(columns={ type :'Hospital_type_'+ type},inplace=True)
dummy_variable_1 = pd.get_dummies(df2["Hospital_type_code"])

df2 = pd.concat([df2, dummy_variable_1], axis=1)

type_list=dummy_variable_1.columns.tolist()

for type in type_list:

         df2.rename(columns={ type :'Hospital_type_'+ type},inplace=True)

df1['Hospital_region_code'].value_counts()
dummy_variable_2 = pd.get_dummies(df1["Hospital_region_code"])

df1 = pd.concat([df1, dummy_variable_2], axis=1)

region_list=dummy_variable_2.columns.tolist()

for region in region_list:

         df1.rename(columns={ region :'Hospital_region_'+ region},inplace=True)

dummy_variable_2 = pd.get_dummies(df2["Hospital_region_code"])

df2 = pd.concat([df2, dummy_variable_2], axis=1)

region_list=dummy_variable_2.columns.tolist()

for region in region_list:

         df2.rename(columns={ region :'Hospital_region_'+ region},inplace=True)

df1['Department'].value_counts()
dummy_variable_3 = pd.get_dummies(df1["Department"])

df1 = pd.concat([df1, dummy_variable_3], axis=1)

department_list=dummy_variable_3.columns.tolist()

for dep in department_list:

         df1.rename(columns={ dep : dep + '_Department'},inplace=True)

dummy_variable_3 = pd.get_dummies(df2["Department"])

df2 = pd.concat([df2, dummy_variable_3], axis=1)

department_list=dummy_variable_3.columns.tolist()

for dep in department_list:

         df2.rename(columns={ dep : dep + '_Department'},inplace=True)

df1['Ward_Type'].value_counts()
dummy_variable_4 = pd.get_dummies(df1["Ward_Type"])

df1 = pd.concat([df1, dummy_variable_4], axis=1)

ward_list=dummy_variable_4.columns.tolist()

for ward in ward_list:

         df1.rename(columns={ ward : 'Ward_Type_'+ ward},inplace=True)

dummy_variable_4 = pd.get_dummies(df2["Ward_Type"])

df2 = pd.concat([df2, dummy_variable_4], axis=1)

ward_list=dummy_variable_4.columns.tolist()

for ward in ward_list:

         df2.rename(columns={ ward : 'Ward_Type_'+ ward},inplace=True)

df1['Ward_Facility_Code'].value_counts()
dummy_variable_5 = pd.get_dummies(df1["Ward_Facility_Code"])

df1 = pd.concat([df1, dummy_variable_5], axis=1)

fac_list=dummy_variable_5.columns.tolist()

for fac in fac_list:

         df1.rename(columns={ fac : 'Ward_Facility_'+ fac},inplace=True)

dummy_variable_5 = pd.get_dummies(df2["Ward_Facility_Code"])

df2 = pd.concat([df2, dummy_variable_5], axis=1)

fac_list=dummy_variable_5.columns.tolist()

for fac in fac_list:

         df2.rename(columns={ fac : 'Ward_Facility_'+ fac},inplace=True)

df1['Type of Admission'].value_counts()
dummy_variable_6 = pd.get_dummies(df1["Type of Admission"])

df1 = pd.concat([df1, dummy_variable_6], axis=1)

ad_list=dummy_variable_6.columns.tolist()

for ad in ad_list:

         df1.rename(columns={ad : ad+'_type'},inplace=True)

dummy_variable_6 = pd.get_dummies(df2["Type of Admission"])

df2 = pd.concat([df2, dummy_variable_6], axis=1)

ad_list=dummy_variable_6.columns.tolist()

for ad in ad_list:

         df2.rename(columns={ad : ad+'_type'},inplace=True)

df1['Severity of Illness'].value_counts()
dummy_variable_7 = pd.get_dummies(df1["Severity of Illness"])

df1 = pd.concat([df1, dummy_variable_7], axis=1)

sol_list=dummy_variable_7.columns.tolist()

for ill in sol_list:

         df1.rename(columns={ill : ill+'_type'},inplace=True)

dummy_variable_7 = pd.get_dummies(df2["Severity of Illness"])

df2 = pd.concat([df2, dummy_variable_7], axis=1)

sol_list=dummy_variable_7.columns.tolist()

for ill in sol_list:

         df2.rename(columns={ill : ill+'_type'},inplace=True)

age_list=df1['Age'].value_counts().index.tolist()

age_list
onehotencoder = OneHotEncoder()

X = onehotencoder.fit_transform(df1.Age.values.reshape(-1,1)).toarray()
dfOneHot = pd.DataFrame(X, columns = ["Age_"+ i for i in age_list] )
df1 = pd.concat([df1, dfOneHot], axis=1)

#df1= df1.drop(['Age'], axis=1)

X = onehotencoder.fit_transform(df2.Age.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Age_"+ i for i in age_list] )

df2 = pd.concat([df2, dfOneHot], axis=1)

#df2= df2.drop(['Age'], axis=1)

df1['Stay']=le.fit_transform(df1['Stay'])
#Replacing the missing bed grade values with mode

Mode=df1['Bed Grade'].mode()

Mode
df1["Bed Grade"].replace(np.nan, 2.0, inplace=True)
Mode2=df2['Bed Grade'].mode()

Mode2
df2["Bed Grade"].replace(np.nan, 2.0, inplace=True)
#Replacing the missing city_code_patient values with mean

Avg=df1['City_Code_Patient'].mean(axis=0)
df1["City_Code_Patient"].replace(np.nan, Avg, inplace=True)
Avg2=df2['City_Code_Patient'].mean(axis=0)
df2["City_Code_Patient"].replace(np.nan, Avg2, inplace=True)
corr=df1.corr()

corr.head()
#Count

Grt=df1.groupby(['Hospital_type_code'],as_index=False)['Hospital_code'].count()

Grt.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grt, x='Hospital_type_code', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dft=df1[['Hospital_type_code','Stay','Hospital_code']]

typ=dft.groupby(['Hospital_type_code','Stay'],as_index=False).count()

typ.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Hospital_type_code', y='Count', hue='Stay', data=typ,palette='magma')

plt.title('Type vs Stay')

plt.show()

#Count

Grr=df1.groupby(['Hospital_region_code'],as_index=False)['Hospital_code'].count()

Grr.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grr, x='Hospital_region_code', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfr=df1[['Hospital_region_code','Stay','Hospital_code']]

reg=dfr.groupby(['Hospital_region_code','Stay'],as_index=False).count()

reg.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Hospital_region_code', y='Count', hue='Stay', data=reg,palette='magma')

plt.title('Region vs Stay')

plt.show()

#Count

plt.figure(figsize=(12,8))

Grd=df1.groupby(['Department'],as_index=False)['Hospital_code'].count()

Grd.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grd, x='Department', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfd=df1[['Department','Stay','Hospital_code']]

dep=dfd.groupby(['Department','Stay'],as_index=False).count()

dep.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Department', y='Count', hue='Stay', data=dep,palette='magma')

plt.title('Department vs Stay')

plt.show()

#Count

Grw=df1.groupby(['Ward_Type'],as_index=False)['Hospital_code'].count()

Grw.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grw, x='Ward_Type', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfw=df1[['Ward_Type','Stay','Hospital_code']]

ward=dfw.groupby(['Ward_Type','Stay'],as_index=False).count()

ward.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Ward_Type', y='Count', hue='Stay', data=ward,palette='magma')

plt.title('Ward_Type vs Stay')

plt.show()

#Count

Grf=df1.groupby(['Ward_Facility_Code'],as_index=False)['Hospital_code'].count()

Grf.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grf, x='Ward_Facility_Code', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dff=df1[['Ward_Facility_Code','Stay','Hospital_code']]

fac=dff.groupby(['Ward_Facility_Code','Stay'],as_index=False).count()

fac.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Ward_Facility_Code', y='Count', hue='Stay', data=fac,palette='magma')

plt.title('Ward_Facility_Code vs Stay')

plt.show()

#Count

Gra=df1.groupby(['Type of Admission'],as_index=False)['Hospital_code'].count()

Gra.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Gra, x='Type of Admission', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfa=df1[['Type of Admission','Stay','Hospital_code']]

ad=dfa.groupby(['Type of Admission','Stay'],as_index=False).count()

ad.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Type of Admission', y='Count', hue='Stay', data=ad,palette='magma')

plt.title('Admission Type vs Stay')

plt.show()
#Count

Grs=df1.groupby(['Severity of Illness'],as_index=False)['Hospital_code'].count()

Grs.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grs, x='Severity of Illness', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfs=df1[['Severity of Illness','Stay','Hospital_code']]

se=dfs.groupby(['Severity of Illness','Stay'],as_index=False).count()

se.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Severity of Illness', y='Count', hue='Stay', data=se,palette='magma')

plt.title('Illness vs Stay')

plt.show()
#Count

Grag=df1.groupby(['Age'],as_index=False)['Hospital_code'].count()

Grag.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grag, x='Age', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfag=df1[['Age','Stay','Hospital_code']]

age=dfag.groupby(['Age','Stay'],as_index=False).count()

age.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Age', y='Count', hue='Stay', data=age,palette='magma')

plt.title('Age vs Stay')

plt.show()
#Count

Grab=df1.groupby(['Bed Grade'],as_index=False)['Hospital_code'].count()

Grab.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(data=Grab, x='Bed Grade', y = "Count")

plt.show()
#Relationship with target

plt.figure(figsize=(16,8))

dfab=df1[['Bed Grade','Stay','Hospital_code']]

bed=dfab.groupby(['Bed Grade','Stay'],as_index=False).count()

bed.rename(columns={'Hospital_code':'Count'},inplace=True)

sns.barplot(x='Bed Grade', y='Count', hue='Stay', data=bed,palette='magma')

plt.title('Bed Gray vs Stay')

plt.show()
c=df1[['Hospital_code', 'Hospital_type_code', 'City_Code_Hospital',

       'Hospital_region_code', 'Available Extra Rooms in Hospital',

       'Department', 'Ward_Type', 'Ward_Facility_Code', 'Bed Grade',

       'patientid', 'City_Code_Patient', 'Type of Admission',

       'Severity of Illness', 'Visitors with Patient', 'Age',

       'Admission_Deposit', 'Stay']].corr()

plt.figure(figsize=(15,10))

sns.heatmap(c,vmax=.3, square=True,annot=True)



#Dropping the unnecessary attributes

df1.drop(['City_Code_Patient','City_Code_Hospital','patientid','Hospital_type_code','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age'],axis=1,inplace=True)

df2.drop(['City_Code_Patient','City_Code_Hospital','patientid','Hospital_type_code','Hospital_region_code','Department','Ward_Type','Ward_Facility_Code','Type of Admission','Severity of Illness','Age'],axis=1,inplace=True)
df1.shape
df2.shape
#Importing the necessary libraries

from sklearn.metrics import classification_report, confusion_matrix

import itertools

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import RFE

import statsmodels.api as sm

from sklearn.model_selection import cross_val_score

from sklearn.metrics import jaccard_score

from sklearn.metrics import log_loss

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_confusion_matrix
scaler=preprocessing.MinMaxScaler()

scaled=pd.DataFrame(scaler.fit_transform(df1),columns=df1.columns)
X=df1[['Hospital_code','Admission_Deposit','Bed Grade','Available Extra Rooms in Hospital', 'Visitors with Patient', 'Hospital_type_a', 'Hospital_type_b','Hospital_type_c', 'Hospital_type_d', 'Hospital_type_e','Hospital_type_f', 'Hospital_type_g', 'Hospital_region_X','Hospital_region_Y', 'Hospital_region_Z','TB & Chest disease_Department', 'anesthesia_Department','gynecology_Department', 'radiotherapy_Department','surgery_Department', 'Ward_Type_P', 'Ward_Type_Q', 'Ward_Type_R','Ward_Type_S', 'Ward_Type_T', 'Ward_Type_U', 'Ward_Facility_A','Ward_Facility_B', 'Ward_Facility_C', 'Ward_Facility_D','Ward_Facility_E', 'Ward_Facility_F', 'Emergency_type', 'Trauma_type','Urgent_type', 'Extreme_type', 'Minor_type', 'Moderate_type','Age_41-50', 'Age_31-40', 'Age_51-60', 'Age_21-30', 'Age_71-80','Age_61-70', 'Age_11-20', 'Age_81-90', 'Age_0-10', 'Age_91-100']]

Y= df1['Stay']

X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1,test_size=0.2)

rfclassifier=RandomForestClassifier(criterion= 'entropy', max_depth= 15, n_estimators=60,random_state=0)

rfclassifier.fit(X_train,y_train)

rfpred=rfclassifier.predict(X_test)
print (classification_report(y_test, rfpred))
fig, ax = plt.subplots(figsize=(20, 20))

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(rfclassifier,X_test,y_test,ax=ax)
import xgboost as xgb
xgbcl=xgb.XGBClassifier(max_depth=10, objective='multi:softmax', n_estimators=40,random_state=42)

xgbcl.fit(X_train,y_train)

xgbpred=xgbcl.predict(X_test)
print (classification_report(y_test, xgbpred))
fig, ax = plt.subplots(figsize=(20, 20))

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(xgbcl,X_test,y_test,ax=ax)
from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer

from scipy.stats import chi2
from catboost import CatBoostClassifier
cat = CatBoostClassifier(verbose=0, n_estimators=100)

cat.fit(X_train,y_train)

catpred=cat.predict(X_test)

print (classification_report(y_test, catpred))
fig, ax = plt.subplots(figsize=(20, 20))

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cat,X_test,y_test,ax=ax)
predictions=cat.predict(df2)
Pred=np.ravel(predictions)
df2['Length of Stay']=le.inverse_transform(Pred)
df2[['Length of Stay']].head(15)