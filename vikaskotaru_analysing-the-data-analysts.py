import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import plotly.plotly as py
import plotly.graph_objs as go
import folium
cvRates = pd.read_csv('../input/conversionRates.csv')
freeForm = pd.read_csv('../input/freeformResponses.csv')
main = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")
schema = pd.read_csv('../input/schema.csv')
plt.figure(figsize=(8,8))
gender=main['GenderSelect'].value_counts()
plt.pie(gender,labels=gender.index,autopct='%.1f%%')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
plt.title('Gender Distribution')

plt.show()
print('Proportion of women in this survey: {:0.2f}% '.format(100*len(main[main['GenderSelect']=='Female'])/len(main['GenderSelect'].dropna())))
print('Proportion of men in this survey: {:0.2f}% '.format(100*len(main[main['GenderSelect']=='Male'])/len(main['GenderSelect'].dropna())))
age=main[(main['Age']>=15) & (main['Age']<=65) ]
age_series=main['Age'].value_counts()
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.boxplot( y=age['Age'],data=age)
plt.title("Age boxplot", fontsize=16)
plt.ylabel("Age", fontsize=16)

plt.subplot(1,2,2)
plt.title('Age Line Graph', fontsize=16)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Value Count',fontsize=16)
sns.lineplot(x=age_series.index, y=age_series.values)


plt.show()
country=main['Country'].value_counts().sort_values().tail(20)
plt.figure(figsize=(10,10))
sns.barplot(y=country.index, x=country.values,alpha=0.9)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Nationality Distributions')
plt.show();
top_5=0
for i in [1,2,4,5,6]:
    top_5=top_5+country.sort_values(ascending=False)[i]
top_5=100*(top_5/len(main))
print('{:0.2f}% of the instances are Americans'.format(100*len(main[main['Country']=='United States'])/len(main)))
print('{:0.2f}% of the instances are Indians'.format(100*len(main[main['Country']=='India'])/len(main)))
print('{:0.2f}% of the instances are contributed by top 5 countries'.format(top_5))
education=main['FormalEducation'].value_counts().sort_values()[:6]
plt.figure(figsize=(10,10))
plt.pie(x=education,labels=education.index,autopct='%.1f%%')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Highest Degree achieved by the participants')
plt.show()
major=main['MajorSelect'].value_counts().sort_values(ascending=True)
major
plt.figure(figsize=(10,10))
sns.barplot(x=major.values,y=major.index)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Degree Majored in')
plt.show();
informal=main['FirstTrainingSelect'].value_counts()
informal
plt.figure(figsize=(10,10))
plt.pie(x=informal.values,labels=informal.index,autopct='%1.1f%%')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Market share of different learning platforms')

plt.show()
employment=main['EmploymentStatus'].value_counts()
employment_df=pd.DataFrame(employment)
n_employed= employment[0]+employment[2]+employment[4]
n_unemployed=employment[1]+employment[3]
x=[n_employed,n_unemployed]


plt.figure(figsize=(10,10))
plt.pie(x=x,labels=['employed','unemployed'],autopct='%1.1f%%')
plt.title("Employed vs Unemployed")
plt.show();
title=main['CurrentJobTitleSelect'].value_counts().sort_values(ascending=True)
plt.figure(figsize=(10,10))
sns.barplot(x=title.values,y=title.index)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title('Distributions based on the titles')
plt.show();
change = main['CareerSwitcher'].value_counts()
plt.figure(figsize=(10,10))
plt.pie(change,labels=change.index,autopct='%1.2f%%')
plt.title('Is Data Science appealing to you')
plt.show();
lang=main[["WorkToolsFrequencyR","WorkToolsFrequencyPython"]].fillna(0)
lang.replace(to_replace=['Rarely','Sometimes','Often','Most of the time'], 
           value=[1,2,3,4], inplace=True)
lang['PythonVsR'] = [ 'R' if (freq1 >2 and freq1 > freq2) else
                    'Python' if (freq1<freq2 and freq2>2) else
                    'Both' if (freq1==freq2 and freq1 >2) else
                    'None' for (freq1,freq2) in zip(lang["WorkToolsFrequencyR"],lang["WorkToolsFrequencyPython"])]
main['PythonVsR']=lang['PythonVsR']

df = main[main['PythonVsR']!='None']
print("Python users: ",len(df[df['PythonVsR']=='Python']))
print("R users: ",len(df[df['PythonVsR']=='R']))
print("Python+R users: ",len(df[df['PythonVsR']=='Both']))
print('Out of the 16000 Participants, only 6167 participants have answered that question')
langg=lang['PythonVsR'].value_counts().drop('None')
plt.figure(figsize=(10,10))
plt.pie(x=langg,labels=langg.index,autopct='%1.2f%%')
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Eternal Battle of Python vs R')
plt.show();
d_task={}
tasks=['TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights']
for task in tasks : 
    d_task[task]={'Python':df[df['PythonVsR']=='Python'][task].mean(),
                  'R':df[df['PythonVsR']=='R'][task].mean(),
                  'Both':df[df['PythonVsR']=='Both'][task].mean()}
    
(pd.DataFrame(d_task)).transpose().plot(kind='barh',figsize=(12,8))
plt.ylabel("Task", fontsize=15)
plt.xlabel("Percentage of time", fontsize=13)
plt.title("Percentage of time devoted to specific tasks ", fontsize=16)
plt.show();
demographic_features = ['GenderSelect','Country','Age',
                        'FormalEducation','MajorSelect','ParentsEducation',
                        'EmploymentStatus', 'CurrentJobTitleSelect',
                        'DataScienceIdentitySelect','CodeWriter',
                        'CurrentEmployerType','JobFunctionSelect',
                        'SalaryChange','RemoteWork','WorkMLTeamSeatSelect',
                        'Tenure','EmployerIndustry','EmployerSize','PythonVsR',
                        'CompensationAmount']
data_dem = main[demographic_features]
data_dem.head(5)
#Convert all salaries to floats
data_dem['CompensationAmount'] = data_dem['CompensationAmount'].fillna(0)
data_dem['CompensationAmount'] = data_dem.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))
                                                       else float(x.replace(',','')))

data_dem = data_dem[(data_dem['CompensationAmount']>5000) & (data_dem['CompensationAmount']<1000000)]
data_dem = data_dem[data_dem['Country']=='United States']

plt.subplots(figsize=(15,8))
sns.distplot(data_dem['CompensationAmount'])
plt.title('Income histograms and fitted distribtion',size=15)
plt.show();
print('The median salary for US data scientist: {} USD'.format(data_dem['CompensationAmount'].median()
))
print('The mean salary for US data scientist: {:0.2f} USD'.format(data_dem['CompensationAmount'].mean()
))
temp=data_dem[data_dem.GenderSelect.isin(['Male','Female'])]
plt.figure(figsize=(10,8))
sns.violinplot( y='CompensationAmount', x='GenderSelect',data=temp)
plt.title("Salary distribution Vs Gender", fontsize=16)
plt.ylabel("Annual Salary", fontsize=16)
plt.xlabel("Gender", fontsize=16)
plt.show();
titles=list(data_dem['FormalEducation'].value_counts().index)
temp=data_dem[data_dem.FormalEducation.isin(titles)]
plt.figure(figsize=(10,8))
sns.boxplot( x='CompensationAmount', y='FormalEducation',data=temp)
plt.title("Salary distribution VS Academic degrees", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.ylabel("Academic degree", fontsize=16)
plt.show();
titles=list(data_dem['CurrentJobTitleSelect'].value_counts().index)
temp=data_dem[data_dem.CurrentJobTitleSelect.isin(titles)]
plt.figure(figsize=(10,8))
sns.violinplot( x='CompensationAmount', y='CurrentJobTitleSelect',data=temp)
plt.title("Salary distribution VS Job Titles", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.ylabel("Job Titles", fontsize=16)
plt.show();
from scipy import stats

salvage=data_dem[['Age','CompensationAmount']]
plt.figure(figsize=(20,8))
#plt.plot(salvage['Age'],salvage['CompensationAmount'])
sns.lineplot(x=salvage['Age'],y=salvage['CompensationAmount'])
plt.xlim(20,70)

xage=data_dem['Age']

plt.show();

temp=data_dem

target = temp['CompensationAmount']
target.replace(to_replace=[1,2], value=[0,1],inplace=True )
temp.drop('CompensationAmount',axis=1,inplace=True)
temp2=pd.get_dummies(data=temp,columns=list(temp))

np.random.seed(42)
perm = np.random.permutation(temp2.shape[0])
X_train , y_train = temp2.iloc[perm[0:round(0.8*temp2.shape[0])]] , target.iloc[perm[0:round(0.8*temp2.shape[0])]]
X_test , y_test = temp2.iloc[perm[round(0.8*temp2.shape[0])::]] , target.iloc[perm[round(0.8*temp2.shape[0])::]]
print('Number of US kagglers with an income lower than 130k$ : {}'.format(len(target)-target.sum()))
print('Number of US kagglers with an income higher than 130k$ : {}'.format(target.sum()))


