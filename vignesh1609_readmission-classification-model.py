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
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, precision_score, recall_score
df=pd.read_csv('/kaggle/input/diabetes/diabetic_data.csv')
df.info()
df.head()
#removing id columns

df.drop(['encounter_id','patient_nbr'],axis=1,inplace=True)
list_unique_columns=[]

for i in df.columns:

    if len(df[i].value_counts())==1:

        list_unique_columns.append(i)

for i in list_unique_columns:

    df.drop([i],axis=1,inplace=True)    
#replace ? with nan

df=df.replace('?',np.nan)  
#missing value function

def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return tt    



missing_data(df)['Percent'].sort_values(ascending=False)
#weight column since it has more missing value 

df['weight'].value_counts()
df['has_weight']=df['weight'].notnull().astype('int')

df.drop(['weight'],axis=1,inplace=True)
df['medical_specialty'].isnull().sum()

df['medical_specialty']=df['medical_specialty'].fillna('UNK') #filling null with unk

print(df['medical_specialty'].nunique()) #more categories
print(df['medical_specialty'].value_counts())
top_10=['UNK','InternalMedicine',

        'Emergency/Trauma','Family/GeneralPractice','Cardiology','Surgery-General',

        'Nephrology','Orthopedics','Orthopedics-Reconstructive','Radiologist']



df.loc[~df['medical_specialty'].isin(top_10),'medical_specialty']='Other'
print(df['payer_code'].isnull().sum())



print(df['payer_code'].value_counts())



df['payer_code']=df['payer_code'].fillna('UNK') #filling null with unk
df['race'].isnull().sum()

df['race'].value_counts()



df['race']=df['race'].fillna('UNK') #filling null with unk
#Generating output variable

#we need to check whether a patient admitted within 30 days or not

df['target']=(df['readmitted']=='<30').astype('int')



#dropping readmitted column

df.drop(['readmitted'],axis=1,inplace=True)
print(df['age'].value_counts())



cleanup_age = {"age":     {"[0-10)": 0, "[10-20)": 10,"[20-30)": 20,"[30-40)": 30,"[40-50)": 40,"[50-60)": 50,

    "[60-70)": 60,"[70-80)": 70,"[80-90)": 80,"[90-100)": 90}}



df.replace(cleanup_age, inplace=True)
#analyzing gender column

df['gender'].value_counts()

#removing invalid/unknown entries for gender

df=df[df['gender']!='Unknown/Invalid']
#Distribution of Readmission

sns.countplot(df['target']).set_title('Distrinution of Readmission')
#checking for balance data

print(sum(df['target'].values)/len(df['target'].values)) 
#time in hospital vs readmitted

fig = plt.figure(figsize=(13,7),)

ax=sns.kdeplot(df.loc[(df['target'] == 0),'time_in_hospital'] , color='b',shade=True,label='Not Readmitted')

ax=sns.kdeplot(df.loc[(df['target'] == 1),'time_in_hospital'] , color='r',shade=True, label='Readmitted')

ax.set(xlabel='Time in Hospital', ylabel='Frequency')

plt.title('Time in Hospital VS. Readmission')
#age vs readmission



fig = plt.figure(figsize=(15,10))

sns.countplot(y= df['age'], hue = df['target']).set_title('Age of Patient VS. Readmission')
#race vs readmission



fig = plt.figure(figsize=(8,8))

sns.countplot(y = df['race'], hue = df['target'])
#Number of medication used VS. Readmission

fig = plt.figure(figsize=(18,18))

sns.countplot(y = df['num_medications'], hue = df['target'])

fig = plt.figure(figsize=(8,8))

sns.barplot(x = df['target'], y = df['num_medications']).set_title("Number of medication used VS. Readmission")
#Gender and Readmission

#Male = 1

#Female = 0



fig = plt.figure(figsize=(8,8))

sns.countplot(df['gender'], hue = df['target']).set_title("Gender of Patient VS. Readmission")
#change of medication vs readmission



fig = plt.figure(figsize=(8,8))

sns.countplot(df['change'], hue = df['target']).set_title('Change of Medication VS. Readmission')
#diabetic medication vs readmission



fig = plt.figure(figsize=(8,8))

sns.countplot(df['diabetesMed'], hue = df['target']).set_title('Diabetes Medication prescribed VS Readmission')
#max_glue_serum vs target

fig = plt.figure(figsize=(8,8))

sns.countplot(y = df['max_glu_serum'], hue = df['target']).set_title('Glucose test serum test result VS. Readmission')
#a1c test result vs target

fig = plt.figure(figsize=(8,10))

sns.countplot(y= df['A1Cresult'], hue = df['target']).set_title('A1C test result VS. Readmission')
#no of lab procedure vs target

fig = plt.figure(figsize=(15,6),)

ax=sns.kdeplot(df.loc[(df['target'] == 0),'num_lab_procedures'] , color='b',shade=True,label='Not readmitted')

ax=sns.kdeplot(df.loc[(df['target'] == 1),'num_lab_procedures'] , color='r',shade=True, label='readmitted')

ax.set(xlabel='Number of lab procedure', ylabel='Frequency')

plt.title('Number of lab procedure VS. Readmission')
#admission type vs readmission

#1-	Emergency

#2-Urgent

#3-Elective

#4-Newborn

#5-Not Available

#6-NULL

#7-Trauma Center

#8-Not Mapped



fig = plt.figure(figsize=(8,10))

sns.countplot(y= df['admission_type_id'], hue = df['target']).set_title('admission_type_id VS. Readmission')
#discharge_disposition_id VS. Readmission

fig = plt.figure(figsize=(8,10))

sns.countplot(y= df['discharge_disposition_id'], hue = df['target']).set_title('discharge_disposition_id VS. Readmission')
categorical_feature=df.select_dtypes(include='object')

cat=categorical_feature.columns

print(cat)
cat=['race', 'gender', 'payer_code', 'medical_specialty', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',

       'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',

       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',

       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',

       'insulin', 'glyburide-metformin', 'glipizide-metformin',

       'glimepiride-pioglitazone', 'metformin-rosiglitazone',

       'metformin-pioglitazone', 'change', 'diabetesMed']
#taking copy of dataframe

df_copy=df.copy()
cols_cat_num=['admission_type_id','discharge_disposition_id','admission_source_id']

df[cols_cat_num]=df[cols_cat_num].astype('str')

df_cat=pd.get_dummies(df[cat+cols_cat_num],drop_first=True)
print(df_cat.columns)
#dropping encoded columns

for i in cat:

    df_copy.drop([i],axis=1,inplace=True)
#concating encoded columns and other columns

df_copy=pd.concat([df_copy,df_cat],axis=1)
df_copy['diag_3'].isnull().sum()

df_copy.dropna(inplace=True)

print(df_copy['diag_1'].nunique())

print(df_copy['diag_2'].nunique())

print(df_copy['diag_3'].nunique())

diag_cols = ['diag_1','diag_2','diag_3']

for col in diag_cols:

    df_copy[col] = df_copy[col].str.replace('E','-')

    df_copy[col] = df_copy[col].str.replace('V','-')

    condition = df_copy[col].str.contains('250')

    df_copy.loc[condition,col] = '250'



df_copy[diag_cols] = df_copy[diag_cols].astype(float)
# diagnosis grouping

for col in diag_cols:

    df_copy['temp']=np.nan

    

    condition = df_copy[col]==250

    #condition = df_copy['diag_1']==250

    df_copy.loc[condition,'temp']='Diabetes'

    

    condition = (df_copy[col]>=390) & (df_copy[col]<=458) | (df_copy[col]==785)

    df_copy.loc[condition,'temp']='Circulatory'

    

    condition = (df_copy[col]>=460) & (df_copy[col]<=519) | (df_copy[col]==786)

    df_copy.loc[condition,'temp']='Respiratory'

    

    condition = (df_copy[col]>=520) & (df_copy[col]<=579) | (df_copy[col]==787)

    df_copy.loc[condition,'temp']='Digestive'

    

    condition = (df_copy[col]>=580) & (df_copy[col]<=629) | (df_copy[col]==788)

    df_copy.loc[condition,'temp']='Genitourinary'

    

    condition = (df_copy[col]>=800) & (df_copy[col]<=999)

    df_copy.loc[condition,'temp']='Injury'

    

    condition = (df_copy[col]>=710) & (df_copy[col]<=739)

    df_copy.loc[condition,'temp']='Muscoloskeletal'

    

    condition = (df_copy[col]>=140) & (df_copy[col]<=239)

    df_copy.loc[condition,'temp']='Neoplasms'

    

    condition = df_copy[col]==0

    df_copy.loc[condition,col]='?'

    df_copy['temp']=df_copy['temp'].fillna('Others')

    condition = df_copy['temp']=='0'

    df_copy.loc[condition,'temp']=np.nan

    df_copy[col]=df_copy['temp']

    df_copy.drop('temp',axis=1,inplace=True)
#ENCODING DIAG_ COLUMNS

df_cat_diag=pd.get_dummies(df_copy[diag_cols],drop_first=True)



#dropping encoded columns

for i in diag_cols:

    df_copy.drop([i],axis=1,inplace=True)



df_copy=pd.concat([df_copy,df_cat_diag],axis=1)
X=df_copy.drop(['target'],axis=1)

y=df_copy['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

logit = LogisticRegression(fit_intercept=True, penalty='l2')

logit.fit(X_train, y_train)

logit_pred = logit.predict(X_test)

pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(logit_pred, name = 'Predict'), margins = True)

print("Accuracy is {0:.2f}".format(accuracy_score(y_test, logit_pred)))

print("Precision is {0:.2f}".format(precision_score(y_test, logit_pred)))

print("Recall is {0:.2f}".format(recall_score(y_test, logit_pred)))



accuracy_logit = accuracy_score(y_test, logit_pred)

precision_logit = precision_score(y_test, logit_pred)

recall_logit = recall_score(y_test, logit_pred)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=28, criterion = "entropy", min_samples_split=10)

dtree.fit(X_train, y_train)

dtree_pred = dtree.predict(X_test)

pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(dtree_pred, name = 'Predict'), margins = True)



print("Accuracy is {0:.2f}".format(accuracy_score(y_test, dtree_pred)))

print("Precision is {0:.2f}".format(precision_score(y_test, dtree_pred)))

print("Recall is {0:.2f}".format(recall_score(y_test, dtree_pred)))



accuracy_dtree = accuracy_score(y_test, dtree_pred)

precision_dtree = precision_score(y_test, dtree_pred)

recall_dtree = recall_score(y_test, dtree_pred)
# Create list of top most features based on importance

feature_names = X_train.columns

feature_imports = dtree.feature_importances_

most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")

most_imp_features.sort_values(by="Importance", inplace=True)

print(most_imp_features)

plt.figure(figsize=(10,6))

plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)

plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)

plt.xlabel('Importance')

plt.title('Most important features - Decision Tree')

plt.show()

from sklearn.ensemble import RandomForestClassifier

rm = RandomForestClassifier(n_estimators = 10, max_depth=25, criterion = "gini", min_samples_split=10)

rm.fit(X_train, y_train)



rm_prd = rm.predict(X_test)

pd.crosstab(pd.Series(y_test, name = 'Actual'), pd.Series(rm_prd, name = 'Predict'), margins = True)



print("Accuracy is {0:.2f}".format(accuracy_score(y_test, rm_prd)))

print("Precision is {0:.2f}".format(precision_score(y_test, rm_prd)))

print("Recall is {0:.2f}".format(recall_score(y_test, rm_prd)))



accuracy_rm = accuracy_score(y_test, rm_prd)

precision_rm = precision_score(y_test, rm_prd)

recall_rm = recall_score(y_test, rm_prd)
# Create list of top most features based on importance

feature_names = X_train.columns

feature_imports = rm.feature_importances_

most_imp_features = pd.DataFrame([f for f in zip(feature_names,feature_imports)], columns=["Feature", "Importance"]).nlargest(10, "Importance")

most_imp_features.sort_values(by="Importance", inplace=True)

plt.figure(figsize=(10,6))

plt.barh(range(len(most_imp_features)), most_imp_features.Importance, align='center', alpha=0.8)

plt.yticks(range(len(most_imp_features)), most_imp_features.Feature, fontsize=14)

plt.xlabel('Importance')

plt.title('Most important features - Random Forest ')

plt.show()
plt.figure(figsize=(14, 7))

ax = plt.subplot(111)



models = ['Logistic Regression', 'Decision Tree', 'Random Forests']

values = [accuracy_logit, accuracy_dtree, accuracy_rm]

model = np.arange(len(models))



plt.bar(model, values, align='center', width = 0.15, alpha=0.7, color = 'red', label= 'accuracy')

plt.xticks(model, models)
ax = plt.subplot(111)



models = ['Logistic Regression', 'Decision Tree', 'Random Forests']

values = [precision_logit, precision_dtree, precision_rm]

model = np.arange(len(models))



plt.bar(model+0.15, values, align='center', width = 0.15, alpha=0.7, color = 'blue', label = 'precision')

plt.xticks(model, models)
ax = plt.subplot(111)



models = ['Logistic Regression', 'Decision Tree', 'Random Forests' ]

values = [recall_logit, recall_dtree, recall_rm, ]

model = np.arange(len(models))



plt.bar(model+0.3, values, align='center', width = 0.15, alpha=0.7, color = 'green', label = 'recall')

plt.xticks(model, models)







plt.ylabel('Performance Metrics for Different models')

plt.title('Model')