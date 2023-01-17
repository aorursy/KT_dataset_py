# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Ldata = pd.read_csv('/kaggle/input/lead-scoring-x-online-education/Leads X Education.csv')
import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

from pandas_profiling import ProfileReport
Ldata.head()
print("Number of data points in data set", Ldata.shape)
Ldata = Ldata.replace("Select",np.nan)
round(Ldata.isnull().sum()/len(Ldata),4)*100
cols_with_null = Ldata.columns[Ldata.isnull().any()]



for col in cols_with_null:

    if Ldata[col].isnull().sum()*100/Ldata.shape[0]>40:

        Ldata.drop(col,1,inplace = True)

        

Ldata.shape 
Ldata = Ldata[Ldata.isnull().sum(axis=1)<=5]
Ldata.shape
Ldata.head()
Ldata['Lead Origin'].value_counts(dropna= False)
Ldata['Lead Source'].value_counts(dropna= False)
Ldata['Lead Source'].fillna("Google",inplace =True)

Ldata['Lead Source']=Ldata['Lead Source'].replace('google','Google')

Ldata['Lead Source']=Ldata['Lead Source'].replace(['Click2call','Social Media','Press_Release','Live Chat','youtubechannel','WeLearn','testone','welearnblog_Home','NC_EDM','Pay per Click Ads','blog'],'Other_Lead_Sources')

Ldata['Lead Source']=Ldata['Lead Source'].str.upper()
Ldata['Lead Source'].value_counts(dropna= False)
Ldata['Page Views Per Visit'] = Ldata['Page Views Per Visit'].fillna(Ldata['Page Views Per Visit'].mean())
Ldata['Specialization'].value_counts(dropna =False)
Ldata['Specialization'].replace(np.NaN,'Finance Management',inplace=True)
Ldata['Specialization'].value_counts(dropna =False)
Ldata['What matters most to you in choosing a course'].value_counts(dropna=False)
Ldata['What matters most to you in choosing a course'].replace(np.NaN,'Better Career Prospects',inplace=True)
Ldata['What matters most to you in choosing a course'].value_counts(dropna=False)
Ldata['What is your current occupation'].value_counts(dropna=False)
Ldata['What is your current occupation'].replace(np.NaN,'Unemployed',inplace= True)
Ldata['What is your current occupation'].value_counts(dropna=False)
Ldata['Tags'].value_counts(dropna= False)
Ldata['Tags'] = Ldata['Tags'].str.upper()
Ldata['Last Activity'] = Ldata['Last Activity'].str.upper()
Ldata['Last Activity'].value_counts(dropna= False)
Ldata['Last Activity'].fillna('EMAIL OPENED',inplace =True)
Ldata['Country'].fillna('India',inplace =True)
Ldata['Country'].value_counts(dropna= False)
Ldata['Specialization'].value_counts(dropna= False)
Ldata = Ldata.drop(['Do Not Email','Do Not Call','Search','Magazine','Newspaper Article','X Education Forums','Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque'],axis = 1 )
Ldata.shape
Ldata['TotalVisits'] = Ldata['TotalVisits'].replace(np.NaN,Ldata['TotalVisits'].median())

Ldata['TotalVisits'].describe()
categorical_data = Ldata.select_dtypes(exclude=[np.number])

categorical_data.head()
threshold = 60 # Anything that occurs less than this will be replaced with "Other"

for col in categorical_data.columns:

    value_counts = categorical_data[col].value_counts() # Specific column 

    to_remove = value_counts[value_counts <= threshold].index

    categorical_data[col].replace(to_remove, "Other", inplace=True)
numeric_data = Ldata.select_dtypes(include=[np.number])

numeric_data.head()
Ldata_new = pd.concat([categorical_data,numeric_data],axis=1,join = 'inner')
Ldata_new.head()
Ldata_new['Country'].value_counts(dropna= False)
sns.countplot(Ldata_new['Country'])

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(Ldata_new['Last Activity'])

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(Ldata_new['What is your current occupation'])

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(Ldata_new['Specialization'])

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(15,5))

sns.countplot(Ldata_new['Tags'])

plt.xticks(rotation=45)

plt.show()
Ldata_new.drop(['Prospect ID','Tags','Country','City'],axis=1 ,inplace=True)
sns.boxplot(Ldata_new['TotalVisits'])

plt.show()
quant = Ldata_new['TotalVisits'].quantile([0.05,0.95]).values

Ldata_new['TotalVisits'][Ldata_new['TotalVisits'] <= quant[0]] = quant[0]

Ldata_new['TotalVisits'][Ldata_new['TotalVisits'] >= quant[1]] = quant[1]
sns.boxplot(Ldata_new['TotalVisits'])

plt.show()
sns.boxplot(x=Ldata_new['Total Time Spent on Website'])

plt.show()
sns.boxplot(Ldata_new['Page Views Per Visit'])

plt.show()
quant = Ldata_new['Page Views Per Visit'].quantile([0.05,0.95]).values

Ldata_new['Page Views Per Visit'][Ldata_new['Page Views Per Visit'] <= quant[0]] = quant[0]

Ldata_new['Page Views Per Visit'][Ldata_new['Page Views Per Visit'] >= quant[1]] = quant[1]
sns.boxplot(Ldata_new['Page Views Per Visit'])

plt.show()
profile = ProfileReport(Ldata, title="Pandas Profiling Report")

profile.to_notebook_iframe()
Ldata_new.isnull().sum()/len(Ldata_new)*100
sns.countplot(x=Ldata_new['Lead Origin'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['Lead Source'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['Last Activity'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['Specialization'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['What is your current occupation'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['What matters most to you in choosing a course'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['A free copy of Mastering The Interview'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.countplot(x=Ldata_new['Last Notable Activity'],hue=Ldata['Converted'])

plt.xticks(rotation=90)

plt.show()
sns.scatterplot(x='Total Time Spent on Website',y='TotalVisits', hue= 'Converted', data= Ldata_new)

plt.show()
sns.scatterplot(x='Total Time Spent on Website',y='Page Views Per Visit', hue= 'Converted', data= Ldata_new)

plt.show()
sns.scatterplot(x='TotalVisits',y='Page Views Per Visit', hue= 'Converted', data= Ldata_new)

plt.show()
plt.figure(figsize = (10,7))

sns.heatmap(Ldata_new.corr(),annot=True,cmap='YlGnBu')

plt.show()
Ldata_new.columns
dummy = pd.get_dummies(Ldata_new[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

       'What is your current occupation','What matters most to you in choosing a course','Last Notable Activity']],drop_first=True)

dummy.head()
Ldata_new = pd.concat([Ldata_new,dummy],axis=1)
Ldata_new.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization',

       'What is your current occupation','What matters most to you in choosing a course','Last Notable Activity'],axis=1,inplace=True)
Ldata_new.head()
Ldata_new['A free copy of Mastering The Interview'] = Ldata_new['A free copy of Mastering The Interview'].map({'Yes':1,'No':0})
Ldata_new
y = Ldata_new['Converted']

X = Ldata_new.drop(['Converted','Lead Number'],axis=1)
from sklearn.model_selection import train_test_split

from sklearn import metrics



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()
sns.distplot(X_train['Total Time Spent on Website'],color = 'blue',label = 'Total Time')

sns.distplot(X_train['TotalVisits'],color = 'green', label = 'Total Visits')

sns.distplot(X_train['Page Views Per Visit'],color = 'red',label = 'Page Views')

plt.legend()

plt.show()
from sklearn.linear_model import LogisticRegression 

lr = LogisticRegression()



from sklearn.feature_selection import RFE

rfe = RFE(lr,20)

rfe = rfe.fit(X_train,y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
n_col = X_train.columns[rfe.support_]

X_train.columns[~rfe.support_]
import statsmodels

import statsmodels.api as sm





X_train_sm = sm.add_constant(X_train[n_col])

logm = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm.fit()

res.summary()
col1 = n_col.drop('What matters most to you in choosing a course_Other',1)

col1
X_train_sm = sm.add_constant(X_train[col1])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col2 = col1.drop('Specialization_Hospitality Management',1)

col2
X_train_sm = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col3 = col2.drop('What is your current occupation_Working Professional',1)

col3
X_train_sm = sm.add_constant(X_train[col3])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col4 = col3.drop('Lead Origin_Other',1)

col4
X_train_sm = sm.add_constant(X_train[col4])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
col5 = col4.drop('Lead Source_Other')

col5
X_train_sm = sm.add_constant(X_train[col5])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred.values.reshape(-1)})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head(10)
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train[col5].columns

vif['VIF'] = [variance_inflation_factor(X_train[n_col].values, i) for i in range(X_train[col5].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col6 = col5.drop('Last Notable Activity_SMS Sent')

col6
X_train_sm = sm.add_constant(X_train[col6])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
vif = pd.DataFrame()

vif['Features'] = X_train[col6].columns

vif['VIF'] = [variance_inflation_factor(X_train[n_col].values, i) for i in range(X_train[col6].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col7 = col6.drop('What is your current occupation_Student')

col7
X_train_sm = sm.add_constant(X_train[col7])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred.values.reshape(-1)})

y_train_pred_final['Lead Number'] = y_train.index

y_train_pred_final.head(10)
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)
numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.35 else 0)



y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))



y_train_pred_final.head()
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)



confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )

confusion2



TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
round(TP / float(TP+FN),2)
round(TN / float(TN+FP),2)
X_test[['Total Time Spent on Website']] = scaler.fit_transform(X_test[['Total Time Spent on Website']])

X_test.head()
X_test = X_test[col7]
X_test.head()
X_test_sm = sm.add_constant(X_test)

y_test_pred= res.predict(X_test_sm)

y_test_pred[:10]
y_pred_1 = pd.DataFrame(y_test_pred)

y_test_df = pd.DataFrame(y_test)

y_test_df['Lead Number'] = y_test_df.index
y_pred_1.reset_index(drop=True,inplace=True)

y_test_df.reset_index(drop=True,inplace=True)

y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)

y_pred_final.head()
y_pred_final = y_pred_final.rename(columns={0 : 'Conversion_Prob'})

y_pred_final.head()
y_pred_final['final_predicted'] = y_pred_final['Conversion_Prob'].map(lambda x: 1 if x>0.3  else 0)

y_pred_final.head()
round(metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted),2)
confusion3 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted)

confusion3
TP = confusion3[1,1]

TN = confusion3[0,0]

FP = confusion3[0,1]

FN = confusion3[1,0]
round(TP/float(FN+TP),2)
round(TN/float(TN+FP),2)
feat_importance = res.params[1:]

feat_importance = 100.0*(feat_importance/feat_importance.max())



sorted_idx = np.argsort(feat_importance,kind= 'quicksort',order = 'list of str')

sorted_idx
plt.figure(figsize = (15,10))

pos= np.arange(sorted_idx.shape[0])



featfig = plt.figure(figsize = (15,10))

featax = featfig.add_subplot(1,1,1)

featax.barh(pos,feat_importance[sorted_idx])

featax.set_yticks(pos)

featax.set_yticklabels(np.array(X_train[col7].columns)[sorted_idx])

featax.set_xlabel('Relative Feature Importance')



plt.tight_layout()

plt.show()