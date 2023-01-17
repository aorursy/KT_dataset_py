# Supress Warnings

import warnings

warnings.filterwarnings('ignore')



# Importing libraries



import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from matplotlib.pyplot import xticks

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



import statsmodels.api as sm



from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn import metrics
leads_df = pd.read_csv('../input/leadscore/Leads.csv')

leads_df.head()
leads_df.shape
sum(leads_df.duplicated(subset = 'Prospect ID'))
sum(leads_df.duplicated(subset = 'Lead Number'))
# looking at summary statistics of numerical columns

leads_df.describe()
# verifying outliers

leads_df['Total Time Spent on Website'].quantile([0,0.25,0.5,0.75,0.95,0.99,1])
leads_df['Page Views Per Visit'].quantile([0,0.25,0.5,0.75,0.95,0.99,1])
# data type of variables

leads_df.info()
# Null values

round((leads_df.isnull().sum()/len(leads_df))*100,1)
# Specialization

leads_df.Specialization.value_counts()
# How did you hear about X Education

leads_df['How did you hear about X Education'].value_counts()
# Lead Profile

leads_df['Lead Profile'].value_counts()
# City

leads_df['City'].value_counts()
leads_df.loc[leads_df['Specialization'].str.lower() == 'select', 'Specialization'] = np.nan

leads_df.loc[leads_df['How did you hear about X Education'].str.lower() == 'select', 'How did you hear about X Education'] = np.nan

leads_df.loc[leads_df['Lead Profile'].str.lower() == 'select', 'Lead Profile'] = np.nan

leads_df.loc[leads_df['City'].str.lower() == 'select', 'City'] = np.nan



leads_df.head()
leads_df = leads_df.drop(['Tags','Lead Quality','Asymmetrique Activity Index','Asymmetrique Profile Index',

                          'Asymmetrique Activity Score','Asymmetrique Profile Score','Lead Profile'], axis=1)
leads_df.info()
# Null values

round((leads_df.isnull().sum()/len(leads_df))*100,1)
# creating a copy of original dataframe

leads_df_prep = leads_df
leads_df_prep = leads_df_prep.drop(leads_df_prep.loc[:,list(round(100*(leads_df_prep.isnull().sum()/len(leads_df_prep.index)), 2)>45)].columns, 1)
# checking dataframe

leads_df_prep.head()
# Null values

round((leads_df_prep.isnull().sum()/len(leads_df_prep))*100,1)
leads_df_prep['Lead Source'].value_counts()
# replacing nulls with mode

leads_df_prep.loc[leads_df_prep['Lead Source'].isnull(), 'Lead Source'] = 'Google'



# checking for null values after replacement, which should be zero

leads_df_prep['Lead Source'].isnull().sum()
# rechecking counts

leads_df_prep['Lead Source'].value_counts()
# This has to be computed using either mean or median. 

# determining the mean and median values for converted and not not converted leads



# mean

print(leads_df_prep.groupby('Converted').TotalVisits.mean())



# median

print(leads_df_prep.groupby('Converted').TotalVisits.median())
# replacing nulls with median

leads_df_prep.loc[leads_df_prep['TotalVisits'].isnull(), 'TotalVisits'] = leads_df_prep['TotalVisits'].median()
# null values should now be zero

leads_df_prep['TotalVisits'].isnull().sum()
# mean

print(leads_df_prep.groupby('Converted').TotalVisits.mean())



# median

print(leads_df_prep.groupby('Converted').TotalVisits.median())
# This has to be computed using either mean or median. 

# determining the mean and median values for converted and not not converted leads



# mean

print(leads_df_prep.groupby('Converted')['Page Views Per Visit'].mean())



# median

print(leads_df_prep.groupby('Converted')['Page Views Per Visit'].median())
# replacing nulls with median

leads_df_prep.loc[leads_df_prep['Page Views Per Visit'].isnull(), 'Page Views Per Visit'] = leads_df_prep['Page Views Per Visit'].median()
# null values should now be zero

leads_df_prep['Page Views Per Visit'].isnull().sum()
# mean

print(leads_df_prep.groupby('Converted')['Page Views Per Visit'].mean())



# median

print(leads_df_prep.groupby('Converted')['Page Views Per Visit'].median())
leads_df_prep['Last Activity'].value_counts()
leads_df_prep.loc[leads_df_prep['Last Activity'].isnull(), 'Last Activity'] = 'Others'
leads_df_prep['Last Activity'].value_counts()
# detrmining leads where Country is missing, however, City is provided



leads_df_prep[leads_df_prep['Country'].isnull() & ~leads_df_prep['City'].isnull()]['City'].value_counts()
# determining countries that fall wihtin "Tier II Cities"



leads_df_prep[leads_df_prep['City'] == 'Tier II Cities']['Country'].value_counts()
# determining overall distribution



leads_df_prep['Country'].value_counts()
# replacing missing values

leads_df_prep.loc[leads_df_prep['Country'].isnull(), 'Country'] = 'India'
# verifying results

leads_df_prep['Country'].value_counts()
leads_df_prep['Specialization'].value_counts(normalize=True) * 100
sns.countplot(y='Specialization',hue='Converted', data=leads_df_prep)

plt.show()
leads_df_prep[leads_df_prep['Specialization'].isnull()]['What is your current occupation'].value_counts(normalize=True)
# replacing missing values with "Others"

leads_df_prep.loc[leads_df_prep['Specialization'].isnull(), 'Specialization'] = 'Others'
# verifying dataframe

leads_df_prep['Specialization'].value_counts(normalize=True) * 100
leads_df_prep['What is your current occupation'].value_counts(normalize=True) * 100
leads_df_prep[leads_df_prep['What is your current occupation'].isnull()]['Specialization'].value_counts(normalize=True)
# replacing missing values with "Others"

leads_df_prep.loc[leads_df_prep['What is your current occupation'].isnull(), 'What is your current occupation'] = 'Other'
leads_df_prep['What is your current occupation'].value_counts(normalize=True) * 100
leads_df_prep['What matters most to you in choosing a course'].value_counts(normalize=True) * 100
leads_df_prep = leads_df_prep.drop('What matters most to you in choosing a course', axis=1)
leads_df_prep.head()
leads_df_prep['City'].value_counts(normalize=True)
# replacing with "Other Cities"

leads_df_prep.loc[leads_df_prep['City'].isnull(), 'City'] = 'Other Cities'
leads_df_prep['City'].value_counts(normalize=True)
leads_df_prep.isnull().sum()/len(leads_df_prep)
# retained rows and columns

leads_df_prep.shape
# determining columns with only one category



columns_with_single_category = []



for col in list(leads_df_prep.columns):

    if leads_df_prep[col].nunique() == 1:

        columns_with_single_category.append(col)



columns_with_single_category
# dropping columns



leads_df_prep = leads_df_prep.drop(columns_with_single_category, axis=1)



leads_df_prep.head()
leads_df_prep['Lead Origin'].value_counts(normalize = True)
print(leads_df_prep['Lead Source'].value_counts(normalize = True))

print(leads_df_prep['Lead Source'].nunique())
# making "google" and "Google" consistent



leads_df_prep['Lead Source'] = leads_df_prep['Lead Source'].replace('google','Google')
# creating "Social Media" as a more generic category



leads_df_prep['Lead Source'] = leads_df_prep['Lead Source'].replace(

    ['Facebook','bing','Live Chat','blog','youtubechannel','NC_EDM'],'Social Media')
# creating "Other Educational Sites" as a more generic category



leads_df_prep['Lead Source'] = leads_df_prep['Lead Source'].replace(

    ['Welingak Website','WeLearn','welearnblog_Home'],'Other Educational Sites')
# creating "Reference and Referral Sites" as a more generic category



leads_df_prep['Lead Source'] = leads_df_prep['Lead Source'].replace(

    ['Reference','Referral Sites'],'Reference and Referral Sites')
# combining categories with low distribution of leads into "Others"



leads_df_prep['Lead Source'] = leads_df_prep['Lead Source'].replace(['Click2call','Press_Release',

                                                     'Pay per Click Ads','testone'] ,'Others')
# determing final categories



leads_df_prep['Lead Source'].value_counts(normalize = True)
print(leads_df_prep['Last Activity'].value_counts(normalize = True))

print(leads_df_prep['Last Activity'].nunique())
# creating "Unreachable" category



leads_df_prep['Last Activity'] = leads_df_prep['Last Activity'].replace(['Email Bounced','Unsubscribed'],'Unreachable')
# combining categories with low distribution into "Others"

leads_df_prep['Last Activity'] = leads_df_prep['Last Activity'].replace([

                                                        'Had a Phone Conversation', 

                                                        'Approached upfront',

                                                        'View in browser link Clicked',       

                                                        'Email Marked Spam',                  

                                                        'Email Received','Resubscribed to emails',

                                                         'Visited Booth in Tradeshow'],'Others')
print(leads_df_prep['Last Activity'].value_counts(normalize = True))

print(leads_df_prep['Last Activity'].nunique())
print(leads_df_prep['Country'].value_counts(normalize = True))

print(leads_df_prep['Country'].nunique())
leads_df_prep = leads_df_prep.drop('Country', axis=1)



leads_df_prep.shape
leads_df_prep.Specialization.value_counts(normalize=True)
# determining distribution against target variable

sns.countplot(y='Specialization', hue='Converted', data=leads_df_prep)

plt.show()
#combining Management Specializations because they show similar trends



leads_df_prep['Specialization'] = leads_df_prep['Specialization'].replace(['Finance Management','Human Resource Management',

                                                           'Marketing Management','Operations Management',

                                                           'IT Projects Management','Supply Chain Management',

                                                    'Healthcare Management','Hospitality Management',

                                                           'Retail Management'] ,'Management_Cross Industry')  
# creating "E-Commerce" as a generic segment



leads_df_prep['Specialization'] = leads_df_prep['Specialization'].replace(['E-Business'],'E-COMMERCE')
leads_df_prep['Specialization'] = leads_df_prep['Specialization'].replace(['Others','Rural and Agribusiness'

                                                                          ,'Services Excellence'],'Other Specializations')
# determining distribution against target variable

leads_df_prep.Specialization.value_counts(normalize=True)
leads_df_prep['What is your current occupation'].value_counts(normalize=True)
leads_df_prep.City.value_counts(normalize=True)
leads_df_prep['City'] = leads_df_prep['City'].replace('Thane & Outskirts','Other Cities of Maharashtra') 
leads_df_prep.City.value_counts(normalize=True)
leads_df_prep['Last Notable Activity'].value_counts(normalize = True)
leads_df_prep = leads_df_prep.drop('Last Notable Activity', axis=1) 
leads_df_prep['Search'].value_counts(normalize=True)
leads_df_prep = leads_df_prep.drop('Search', axis=1)
leads_df_prep['Through Recommendations'].value_counts(normalize=True)
leads_df_prep = leads_df_prep.drop('Through Recommendations', axis=1)
leads_df_prep['Digital Advertisement'].value_counts(normalize=True)
leads_df_prep = leads_df_prep.drop('Digital Advertisement', axis=1)
print(leads_df_prep['Newspaper Article'].value_counts(normalize=True))

print(leads_df_prep['X Education Forums'].value_counts(normalize=True))

print(leads_df_prep['X Education Forums'].value_counts(normalize=True))

print(leads_df_prep['Do Not Call'].value_counts(normalize=True))
leads_df_prep = leads_df_prep.drop(['Newspaper Article', 'X Education Forums', 'Newspaper','Do Not Call'], axis=1)
leads_df_prep.info()
leads_df_prep.shape
leads_df_prep.describe()
plt.figure(figsize=(14,3))

plt.subplot(1,3,1)

sns.boxplot(x='TotalVisits', data=leads_df_prep)



plt.subplot(1,3,2)

sns.boxplot(x='Total Time Spent on Website', data=leads_df_prep)



plt.subplot(1,3,3)

sns.boxplot(x='Page Views Per Visit', data=leads_df_prep)
# checking quantile range

leads_df_prep['TotalVisits'].quantile([0,0.25,0.5,0.75,0.9,0.90,0.95,0.99,1])
# determining 95th quantile



q95 = leads_df_prep['TotalVisits'].quantile(0.95)



# replacing leads with TotalVisits more than the 95th percentile, with 95th percentile

leads_df_prep['TotalVisits'][leads_df_prep['TotalVisits']>q95] = q95
leads_df_prep['Page Views Per Visit'].quantile([0,0.25,0.5,0.75,0.9,0.90,0.95,0.99,1])
# determining 95th quantile



q95 = leads_df_prep['Page Views Per Visit'].quantile(0.95)



# replacing leads with TotalVisits more than the 95th percentile, with 95th percentile

leads_df_prep['Page Views Per Visit'][leads_df_prep['Page Views Per Visit']>q95] = q95
plt.figure(figsize=(14,3))

plt.subplot(1,3,1)

sns.boxplot(x='TotalVisits', data=leads_df_prep)



plt.subplot(1,3,2)

sns.boxplot(x='Total Time Spent on Website', data=leads_df_prep)



plt.subplot(1,3,3)

sns.boxplot(x='Page Views Per Visit', data=leads_df_prep)
# Overall Conversion Rate



round((sum(leads_df_prep['Converted'])/len(leads_df_prep))*100,2)
# Lead Origin



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(y='Lead Origin', hue='Converted', data=leads_df_prep)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('Lead Origin').Converted.sum()/leads_df_prep.groupby('Lead Origin')['Lead Number'].count()

df = df.reset_index()

df.columns = ['Lead Origin','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='Lead Origin', x='Conversion Ratio', data=df, color='salmon')



plt.tight_layout()

plt.show()
# Lead Source



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.countplot(y='Lead Source', hue='Converted', data=leads_df_prep, order=leads_df_prep['Lead Source'].value_counts().index)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('Lead Source').Converted.sum()/leads_df_prep.groupby('Lead Source')['Lead Number'].count()

df = df.reset_index()

df.columns = ['Lead Source','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='Lead Source', x='Conversion Ratio', data=df, color='salmon', order=leads_df_prep['Lead Source'].value_counts().index)



plt.tight_layout()

plt.show()
# Do not Email



plt.figure(figsize=(15,5))



plt.subplot(2,2,1)

sns.countplot(y='Do Not Email', hue='Converted', data=leads_df_prep)



plt.subplot(2,2,2)

df = leads_df_prep.groupby('Do Not Email').Converted.sum()/leads_df_prep.groupby('Do Not Email')['Lead Number'].count()

df = df.reset_index()

df.columns = ['Do Not Email','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='Do Not Email', x='Conversion Ratio', data=df, color='salmon')
# Last Activity

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.countplot(y='Last Activity', hue='Converted', data=leads_df_prep, order=leads_df_prep['Last Activity'].value_counts().index)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('Last Activity').Converted.sum()/leads_df_prep.groupby('Last Activity')['Lead Number'].count()

df = df.reset_index()

df.columns = ['Last Activity','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='Last Activity', x='Conversion Ratio', data=df, color='salmon', order=leads_df_prep['Last Activity'].value_counts().index)



plt.tight_layout()
# Specialization

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.countplot(y='Specialization', hue='Converted', data=leads_df_prep, order=leads_df_prep['Specialization'].value_counts().index)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('Specialization').Converted.sum()/leads_df_prep.groupby('Specialization')['Lead Number'].count()

df = df.reset_index()

df.columns = ['Specialization','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='Specialization', x='Conversion Ratio', data=df, color='salmon', order=leads_df_prep['Specialization'].value_counts().index)



plt.tight_layout()
# What is your current occupation

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.countplot(y='What is your current occupation', hue='Converted', data=leads_df_prep, order=leads_df_prep['What is your current occupation'].value_counts().index)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('What is your current occupation').Converted.sum()/leads_df_prep.groupby('What is your current occupation')['Lead Number'].count()

df = df.reset_index()

df.columns = ['What is your current occupation','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='What is your current occupation', x='Conversion Ratio', data=df, color='salmon', order=leads_df_prep['What is your current occupation'].value_counts().index)



plt.tight_layout()
# City

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.countplot(y='City', hue='Converted', data=leads_df_prep, order=leads_df_prep['City'].value_counts().index)



plt.subplot(1,2,2)

df = leads_df_prep.groupby('City').Converted.sum()/leads_df_prep.groupby('City')['Lead Number'].count()

df = df.reset_index()

df.columns = ['City','Conversion Ratio']

#df = df.sort_values(by = 'Conversion Ratio', ascending=False)

sns.barplot(y='City', x='Conversion Ratio', data=df, color='salmon', order=leads_df_prep['City'].value_counts().index)



plt.tight_layout()
plt.figure(figsize=(14,5))

plt.subplot(1,3,1)

sns.boxplot(y='TotalVisits', x='Converted', data=leads_df_prep)



plt.subplot(1,3,2)

sns.boxplot(y='Total Time Spent on Website', x='Converted', data=leads_df_prep)



plt.subplot(1,3,3)

sns.boxplot(y='Page Views Per Visit', x='Converted', data=leads_df_prep)



plt.tight_layout()
sns.heatmap(leads_df_prep.corr(), annot=True)
var_list = ['Do Not Email', 'A free copy of Mastering The Interview']
# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Capitalize all binary variables

leads_df_prep['Do Not Email'] = leads_df_prep['Do Not Email'].str.capitalize()

leads_df_prep['A free copy of Mastering The Interview'] = leads_df_prep['A free copy of Mastering The Interview'].str.capitalize()



# applying the function to the variables

leads_df_prep[var_list] = leads_df_prep[var_list].apply(binary_map)
leads_df_prep[var_list].head()
dummies_list = ['Lead Origin','Lead Source','Last Activity','Specialization','What is your current occupation','City']
# creating dummies

dummy_set = pd.get_dummies(leads_df_prep[dummies_list], drop_first=True)
# concatenating with dataframe



leads_df_prep = pd.concat([leads_df_prep,dummy_set], axis=1)

leads_df_prep.head()
# dropping all catagorical variables, since we have already created dummies 

leads_df_prep = leads_df_prep.drop(dummies_list, axis=1)
leads_df_prep.head()
# Putting feature variable to X

X = leads_df_prep.drop(['Converted','Prospect ID','Lead Number'], axis=1)



X.head()
y = leads_df_prep['Converted']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
# Standardizing following variables (others are already in Binary form)

scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])



X_train.head()
logreg = LogisticRegression()
rfe = RFE(logreg, 25) 

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# Selected Columns

col = X_train.columns[rfe.support_]

len(col)
# excluded columns

X_train.columns[~rfe.support_]
# fitting the model using RFE variables



X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
X_train['What is your current occupation_Housewife'].value_counts(normalize=True)
X_train['Lead Source_Others'].value_counts(normalize=True)
X_train['Lead Origin_Lead Import'].value_counts(normalize=True)
X_train['Specialization_International Business'].value_counts(normalize=True)
col = col.drop(['What is your current occupation_Housewife', 'Lead Source_Others','Lead Origin_Lead Import',

'Specialization_International Business'],1)

len(col)
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col = col.drop(['What is your current occupation_Unemployed'],1)

col
# re-checking VIF

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# dropping Column with high p-value

col = col.drop(['Last Activity_Olark Chat Conversation'],1)

col
# re-checking VIF

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# dropping column with high p-value

col = col.drop(['What is your current occupation_Student'],1)

len(col)
# re-checking VIF

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# Dropping "_OTHER" Categories and "A free copy of Mastering The Interview"

col = col.drop(['What is your current occupation_Other','Specialization_Other Specializations','Last Activity_Others'

               ,'A free copy of Mastering The Interview'],1)

col
X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()
# re-checking VIF

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(X_train_sm).values.reshape(-1)
# verifying first 10 probabilities in the array

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converting_Probabilities':y_train_pred})

y_train_pred_final['Lead ID'] = y_train.index

y_train_pred_final.head()
# Creating columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Converting_Probabilities.map(lambda x: 1 if x > i else 0)

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

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
# Predicting Conversion (0/1) based on the predicted probabilities, using 0.35 as the cut-off.



y_train_pred_final['Conversion_predicted'] = y_train_pred_final.Converting_Probabilities.map(lambda x: 1 if x > 0.35 else 0)



y_train_pred_final.head()
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Conversion_predicted )

confusion
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
round(((TP+TN)/(TP+TN+FP+FN))*100,1)
round((TP / float(TP+FN))*100,1)
round((TN / float(TN+FP))*100,1)
round((TP / float(TP+FP))*100,1)
round((TP / float(TP+FN))*100,1)
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
fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted, 

                                         y_train_pred_final.Converting_Probabilities, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converting_Probabilities)
from sklearn.metrics import precision_recall_curve
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converting_Probabilities)



# plot

plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_test = X_test[col]

X_test.head()
# add constant

X_test_sm = sm.add_constant(X_test)



# making predictions

y_test_pred = res.predict(X_test_sm)
# creating dataframe with actual and predicted values from test set

y_test_pred_final = pd.DataFrame({'Converted':y_test.values, 'Converting_Probabilities':y_test_pred})

y_test_pred_final['Lead ID'] = y_test.index

y_test_pred_final.head()
y_test_pred_final['Conversion_predicted'] = y_test_pred_final.Converting_Probabilities.map(lambda x: 1 if x > 0.40 else 0)

y_test_pred_final.head()
confusion_test = metrics.confusion_matrix(y_test_pred_final.Converted, y_test_pred_final.Conversion_predicted )

confusion_test
TP_test = confusion_test[1,1] # true positive 

TN_test = confusion_test[0,0] # true negatives

FP_test = confusion_test[0,1] # false positives

FN_test = confusion_test[1,0] # false negatives
round(((TP_test+TN_test)/(TP_test+TN_test+FP_test+FN_test))*100,1)
round((TP_test / float(TP_test+FN_test))*100,1)
round((TN_test / float(TN_test+FP_test))*100,1)
y_train_pred_final.head()
y_train_pred_final['Lead Score'] = y_train_pred_final['Converting_Probabilities']*100



y_train_pred_final['Lead Score'] = y_train_pred_final['Lead Score'].astype(int)
# keeping only relevant variables

y_train_pred_final = y_train_pred_final[['Converted','Converting_Probabilities','Lead ID','Conversion_predicted','Lead Score']]
y_train_pred_final.head()
y_test_pred_final.head()
y_test_pred_final['Lead Score'] = y_test_pred_final['Converting_Probabilities']*100



y_test_pred_final['Lead Score'] = y_test_pred_final['Lead Score'].astype(int)
y_test_pred_final.head()
Lead_Score_df = y_train_pred_final.append(y_test_pred_final)

Lead_Score_df.head()
len(Lead_Score_df)
# Ensuring the LeadIDs are unique for each lead in the finl dataframe

len(Lead_Score_df['Lead ID'].unique().tolist())
# Making Index as the column identifier

leads_df = leads_df.reset_index()

leads_df.head()
# renaming "index" to "Lead ID"

leads_df = leads_df.rename(columns={"index": "Lead ID"})

leads_df.head()
leads_df_scored = pd.merge(leads_df,Lead_Score_df[['Lead ID','Lead Score']], on='Lead ID', how='inner')

leads_df_scored.info()
leads_df_scored = leads_df_scored.drop('Lead ID', axis=1)
leads_df_scored.head()