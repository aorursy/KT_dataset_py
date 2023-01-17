# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')
data.head()
data.shape
data.isnull().values.any()
plt.figure(figsize=(12,8))

sns.countplot(y='GenderSelect',data=data,order = data['GenderSelect'].value_counts().index)

plt.title("Gender Distribution of the suervey participants", fontsize=16)

plt.xlabel("Number of participants", fontsize=16)

plt.ylabel("Gender", fontsize=16)

plt.yticks(range(4), ['Male', 'Female', 'Different', 'Non-confirming'])

plt.show()



plt.figure(figsize=(18,15))

sns.countplot(y='Country',data=data,order = data['Country'].value_counts().index)

plt.title("Country Distribution of the suervey participants", fontsize=16)

plt.xlabel("Number of participants", fontsize=16)

plt.ylabel("Country", fontsize=16)

plt.show()
plt.figure(figsize=(12,8))

sns.distplot(data.Age)

plt.xlabel('Age', fontsize=16)

plt.ylabel('Percentage',fontsize=16)

plt.title('Age distribution',fontsize=16)

plt.show()
age_country = data[['Country', 'Age']]

# Drop the null values

age_country = age_country.dropna()

# Drop values > 60 and < 10

age_country = age_country.drop(age_country.index[(age_country['Age'] > 60) | (age_country['Age'] < 10)]).reset_index(drop=True)

# Get USA and India from the groups

age_USA = age_country.groupby('Country').get_group('United States')

age_India = age_country.groupby('Country').get_group('India')





# Count and plot 

age_count = age_USA.Age.value_counts()

plt.figure(figsize=(30,15))

sns.barplot(x=age_count.index, y=age_count.values)

plt.xlabel('Age', fontsize=16)

plt.ylabel('Count',fontsize=16)

plt.title('Age distribution in USA',fontsize=16)

plt.show()





age_count = age_India.Age.value_counts()

plt.figure(figsize=(30,15))

sns.barplot(x=age_count.index, y=age_count.values)

plt.xlabel('Age', fontsize=25)

plt.ylabel('Count',fontsize=25)

plt.title('Age distribution in India',fontsize=40)

plt.show()

plt.figure(figsize=(10,8))

sns.countplot(y='EmploymentStatus',data=data,orient='h',order = data['EmploymentStatus'].value_counts().index,palette=sns.color_palette('inferno',7))

plt.title('Employment status', fontsize=16)

plt.xlabel('Count', fontsize=16)

plt.ylabel('Type', fontsize=16)

plt.show()   
plt.figure(figsize=(12,8))

sns.countplot(y=data['MajorSelect'] ,data=data,orient='h',order = data['MajorSelect'].value_counts().index,palette=sns.color_palette('inferno',15))

plt.title("Majors of the survey participants", fontsize=16)

plt.xlabel("Number of participants", fontsize=16)

plt.ylabel("Major", fontsize=16)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y=data['CurrentJobTitleSelect'] ,data=data,order = data['CurrentJobTitleSelect'].value_counts().index,palette=sns.color_palette('inferno',16))

plt.title("Job titles", fontsize=16)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y=data['FormalEducation'] ,data=data,orient='h',order = data['FormalEducation'].value_counts().index,palette=sns.color_palette('inferno',7))

plt.title("'Formal Education'", fontsize=16)

plt.xlabel("count", fontsize=16)

plt.ylabel("Degree", fontsize=16)

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(y=data['Tenure'] ,data=data,order = data['Tenure'].value_counts().index,palette=sns.color_palette('inferno',6))

plt.title("For how many years ahve you been coding?", fontsize=16)

plt.ylabel("Coding exp. in years", fontsize=16)

plt.xlabel("Count", fontsize=16)

plt.show()
import plotly.express as px

temp=data['LanguageRecommendationSelect'].value_counts()

labels = temp.index

sizes = temp.values

fig = px.pie(temp, values=temp.values, names=temp.index, title='Type of Language')

fig.show()
temp = data[(data['CurrentJobTitleSelect'].notnull()) & ((data['LanguageRecommendationSelect'] == 'Python') | (data['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(8, 10))

sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=temp)
plt.figure(figsize=(12,8))

# Pie chart

temp=data['FirstTrainingSelect'].value_counts()

labels = temp.index

sizes = temp.values



#colors

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']

#explsion

explode = (0.05,0.05,0.05,0.05,0.05,0.05)

 

plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=-30, pctdistance=0.80, explode = explode)

#draw circle

centre_circle = plt.Circle((0,0),0.60,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle

plt.axis('equal') 

plt.title('Learning Data Science ',fontsize=18)

plt.tight_layout()

plt.show()
#Explode the Pandas Dataframe to get the number of times each Learning Platform was mentioned

data['LearningPlatformSelect'] = data['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))

s = data.apply(lambda x: pd.Series(x['LearningPlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platform'

plt.figure(figsize=(15,8))

temp = s[s != 'nan'].value_counts()

sns.barplot(y=temp.index, x=temp)
data['BlogsPodcastsNewslettersSelect'] = data['BlogsPodcastsNewslettersSelect'].astype('str').apply(lambda x: x.split(','))

s = data.apply(lambda x: pd.Series(x['BlogsPodcastsNewslettersSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platforms'

s = s[s != 'nan'].value_counts()

plt.figure(figsize=(15,8))

plt.title("Most Popular Blogs and Podcasts")

sns.barplot(y=s.index, x=s)
plt.figure(figsize = (50, 30))

ax=plt.subplot(121)

temp = data['MLToolNextYearSelect'].value_counts().head(15)

sns.barplot(y=temp.index, x=temp,palette=sns.color_palette('inferno_r',25))

plt.title("ML Tool Next Year",fontsize=50)

ax=plt.subplot(122)

temp1 = data['MLMethodNextYearSelect'].value_counts().head(15)

sns.barplot(y=temp1.index, x=temp1,palette=sns.color_palette('inferno_r',25))

plt.title("ML Method Next Year",fontsize=50)

data['HardwarePersonalProjectsSelect'] = data['HardwarePersonalProjectsSelect'].astype('str').apply(lambda x: x.split(','))

s = data.apply(lambda x: pd.Series(x['HardwarePersonalProjectsSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'hardware'

s=s[s != 'nan'].value_counts()

plt.figure(figsize=(15,8))

plt.title('Machines Used')

sns.barplot(y=s.index, x=s,palette=sns.color_palette('inferno_r',15))
data['PublicDatasetsSelect'] = data['PublicDatasetsSelect'].astype('str').apply(lambda x: x.split(','))

q = data.apply(lambda x: pd.Series(x['PublicDatasetsSelect']),axis=1).stack().reset_index(level=1, drop=True)

q.name = 'courses'

q = q[q != 'nan'].value_counts()

import plotly.express as px

#plt.title("Most Popular Dataset Platforms")

data_canada = px.data.gapminder().query("country == 'Canada'")

fig = px.bar(q, y=q.index, x=q.values)

fig.show()
!pip install python-highcharts
time_features = [x for x in data.columns if x.find('Time') != -1][4:10]

s = {}

for feature in time_features:

    s[feature[len('Time'):]] =data[feature].mean()



s = pd.Series(s)
from highcharts import Highchart

source=pd.DataFrame({'Source':s.index,'Count':s.values})

H = Highchart(width=650, height=500)



options = {

    'chart': {

        'type': 'pie',

        'options3d': {

            'enabled': True,

            'alpha': 45

        }

    },

    'title': {

        'text': "Finding Public DataSets From?"

    },

    'plotOptions': {

        'pie': {

            'innerSize': 100,

            'depth': 45

        }

    },

}



temp= source.values.tolist()



H.set_dict_options(options)

H.add_data_set(temp, 'pie', 'Count')



H
plt.figure(figsize=(8,8))

sns.countplot(y='TimeSpentStudying', data=data, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.figure(figsize=(15,8))

sns.countplot(y='ProveKnowledgeSelect', data=data,order = data['ProveKnowledgeSelect'].value_counts().index,palette=sns.color_palette('winter',10))

plt.show()
data[data['AlgorithmUnderstandingLevel'].notnull()]

plt.figure(figsize=(15,8))

sns.countplot(y='AlgorithmUnderstandingLevel', data=data,order = data['AlgorithmUnderstandingLevel'].value_counts().index)

plt.show()
data[data['JobSearchResource'].notnull()]

plt.figure(figsize=(15,8))

sns.countplot(y='JobSearchResource', data=data,order = data['JobSearchResource'].value_counts().index,palette=sns.color_palette('winter',10))

plt.title("Best Places to look for a Data Science Job")

plt.show()
data[data['EmployerSearchMethod'].notnull()]

plt.figure(figsize=(15,8))

sns.countplot(y='EmployerSearchMethod', data=data,order = data['EmployerSearchMethod'].value_counts().index,palette=sns.color_palette('winter',10))

plt.title("Top Places to get Data Science Jobs")

plt.show()
free=pd.read_csv('/kaggle/input/kaggle-survey-2017/freeformResponses.csv')



import nltk

from nltk.corpus import stopwords



import nltk

nltk.download('punkt')

nltk.download('stopwords')

stop_words=set(stopwords.words('english'))

stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...')

library=free['WorkLibrariesFreeForm'].dropna().apply(nltk.word_tokenize)

lib=[]

for i in library:

    lib.extend(i)

lib=pd.Series(lib)

lib=([i for i in lib.str.lower() if i not in stop_words])

lib=pd.Series(lib)

lib=lib.value_counts().reset_index()

lib.loc[lib['index'].str.contains('Pandas|pandas|panda'),'index']='Pandas'

lib.loc[lib['index'].str.contains('Tensorflow|tensorflow|tf|tensor'),'index']='Tensorflow'

lib.loc[lib['index'].str.contains('Scikit|scikit|sklearn'),'index']='Sklearn'

lib=lib.groupby('index')[0].sum().sort_values(ascending=False).to_frame()

R_packages=['dplyr','tidyr','ggplot2','caret','randomforest','shiny','R markdown','ggmap','leaflet','ggvis','stringr','tidyverse','plotly']

Py_packages=['Pandas','Tensorflow','Sklearn','matplotlib','numpy','scipy','seaborn','keras','xgboost','nltk','plotly']

f,ax=plt.subplots(1,2,figsize=(18,10))

lib[lib.index.isin(Py_packages)].sort_values(by=0,ascending=True).plot.barh(ax=ax[0],width=0.9,color=sns.color_palette('winter',10))

ax[0].set_title('Most Frequently Used Py Libraries')

lib[lib.index.isin(R_packages)].sort_values(by=0,ascending=True).plot.barh(ax=ax[1],width=0.9,color=sns.color_palette('winter',10))

ax[1].set_title('Most Frequently Used R Libraries')

ax[1].set_ylabel('')

plt.show()
