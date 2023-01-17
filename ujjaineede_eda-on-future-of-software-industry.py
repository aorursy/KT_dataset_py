import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns 
data=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')
df=pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
data.head()
df.head()
data.shape
data.columns
data.info()
data.drop(['CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','ConvertedComp','WorkWeekHrs','WorkPlan','WorkChallenge','WorkRemote','WorkLoc','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','OrgSize','YearsCodePro','CareerSat','JobSat','MgrIdiot','MgrMoney','MgrWant','LastInt','FizzBuzz'],axis=1,inplace=True)
data.shape
data['Employment'].value_counts()
data['Employment'].value_counts().head(3).plot(kind='pie',autopct='%0.2f')
data['MainBranch'].value_counts()
data['MainBranch'].value_counts().head(3).plot(kind='pie',autopct='%0.2f')
a=data['MainBranch'].value_counts()
b=data['MainBranch'].value_counts()>3500
c=a[b].index.tolist()

c
data=data[data['MainBranch'].isin(c)]
data.shape
data['Hobbyist'].value_counts()
data['Hobbyist'].replace({'Yes':1,'No':0},inplace=True)
data['Country'].value_counts().head(10)

#top 10 Country with most number of upcoming developers
data_india=data[data['Country']=='India']

sns.barplot(y='YearsCode',x='Age',data=data_india,color='red')

data_US=data[data['Country']=='United States']

sns.barplot(y='YearsCode',x='Age',data=data_US,color='blue')

data_germany=data[data['Country']=='Germany']

sns.barplot(y='YearsCode',x='Age',data=data_US,color='green')
data['UndergradMajor'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')

#top 5 streams people are studying as their main field
data['Age'].value_counts().head(10).plot(kind='bar')

#top 10 age groups who codes
sns.barplot(x='Age',y= 'Employment',data=data)
data['BetterLife'].value_counts().plot(kind='pie',autopct='%0.2f')
data['Gender'].value_counts()
data['Gender'].value_counts().head(2).plot(kind='pie',autopct='%0.2f')

#the two genders who codes most
data['OpenSourcer'].value_counts()
data['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')

#checking for percentage
data_india=data[data['Country']=='India']

data_us=data[data['Country']=='United States']

data_germany=data[data['Country']=='Germany']
#using the top 3 countries for comparison
data_india['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')
data_us['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')
data_germany['OpenSourcer'].value_counts().plot(kind='pie',autopct='%0.2f')
data['UndergradMajor'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')

#top 5 subjects which people opt as their main field
data_india['EdLevel'].value_counts().head(5)
data_us['EdLevel'].value_counts().head(5)
data_germany['EdLevel'].value_counts().head(5)
data['SocialMedia'].value_counts().head(5).plot(kind='pie',autopct='%0.2f')