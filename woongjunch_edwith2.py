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
%matplotlib inline

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
mcq=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1',low_memory=False)
mcq.columns
sns.countplot(y='LanguageRecommendationSelect',data=mcq)
sns.countplot(y='CurrentJobTitleSelect',data=mcq)
sns.countplot(y=mcq['CurrentJobTitleSelect'])
# 현재 하고 있는 일에 대한 전체 응답수

mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape
mcq['CurrentJobTitleSelect'].shape
# 현재 하고있는 일에 대한 응답을 해준 사람중   Python과 R을 사용하는 사람

# 응답자들이 실제 업무에서 어떤 언어를 주로 사용하는지 볼 수 있다.



data=mcq[(mcq['CurrentJobTitleSelect'].notnull())& 

         ((mcq['LanguageRecommendationSelect']=='Python') | 

         (mcq['LanguageRecommendationSelect']=='R'))]
data
print(data.shape)
plt.figure(figsize=(8,10))

sns.countplot(y='CurrentJobTitleSelect',

                hue='LanguageRecommendationSelect',

                 data=data)
mcq_ml_tool_count = pd.DataFrame(

    mcq['MLToolNextYearSelect'].value_counts())



mcq_ml_tool_percent=pd.DataFrame(

    mcq['MLToolNextYearSelect'].value_counts(normalize=True))



mcq_ml_tool_df= mcq_ml_tool_count.merge(

    mcq_ml_tool_percent ,

    left_index=True,

    right_index=True).head(20)



mcq_ml_tool_df.columns= ['응답수','비율']

mcq_ml_tool_df
# data=mcq['MLToolNextYearSelect'].value_counts().head(20)

# data=mcq['MLToolNextYearSelect'].head(20)

data=mcq

# data.columns

# data

plt.figure(figsize=(8,10))

# data.plot.barh()

sns.countplot(y="MLToolNextYearSelect",data=data)

# data.head(20)
data=mcq['MLToolNextYearSelect'].value_counts().head(20)



# sns.countplot('MLToolNex')

# sns.barplot(y=data.index,x=data)
data= mcq['MLMethodNextYearSelect'].value_counts().head(15)



sns.barplot(y=data.index,x=data)
data=mcq['LearningPlatformSelect'].value_counts().head(15)

sns.barplot(y=data.index,x=data)
mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str')

# mcq.shape

# s = mcq['LearningPlatformSelect'].head()

mcq['LearningPlatformSelect']





s= pd.Series(mcq['LearningPlatformSelect'])

s.name = 'platform'

# s= mcq.apply(

#     lambda x : pd.Series(x['LearningPlatformSelect']),

#     axis = 1)

# s= mcq.apply(

#     lambda x : pd.Series(x['LearningPlatformSelect']),

#     axis = 1).stack()

# s= mcq.apply(

#     lambda x : pd.Series(x['LearningPlatformSelect']),

#     axis = 1).stack().reset_index(level=1,drop=True)



s
plt.figure(figsize=(6,8))

data = s[s!='nan'].value_counts().head(15)

sns.barplot(x=data,y=data.index)
# 설문 내용과 누구에게 물어봤는지를 찾아봄

question = pd.read_csv('../input/kaggle-survey-2017/schema.csv')

# qc = question.loc[question['Column'].str.contains('LearningCategory')]

qc=question[question['Column'].str.contains('LearningCategory')]

qc
use_features = [x for x in mcq.columns if x.find(

    'LearningPlatformUsefulness') != -1]

use_features
fdf ={}



for feature in use_features :

        a=mcq[feature].value_counts()

        a= a/a.sum()

        fdf[feature[len('LearningPlatformUsefulness'):]] = a



        

# fdf = pd.DataFrame(fdf)

fdf = pd.DataFrame(fdf).transpose().sort_values(

        'Very useful' ,ascending=False)

plt.figure(figsize=(10,10))

sns.heatmap(

    fdf , annot=True

    )

# fdf['Very useful']

# sns.barplot(y=fdf.index,x=fdf['Very useful'])
fdf.plot(kind='bar' , figsize=(20,8) ,

            title = "Usefullness of Learning Platforms")
cdf={}

cat_features = [x for x in mcq.columns if x.find(

    'LearningCategory') != -1]

# cat_features

for feature in cat_features:

    cdf[feature[len('LearningCategory'): ]]=mcq[feature].mean()

# cdf

# cdf = pd.DataFrame(cdf , index =range(0,1) )  

cdf = pd.Series(cdf)  

plt.pie(cdf , labels= cdf.index,autopct='%1.1f%%',shadow= True,

          startangle=140 )

plt.title("Contribution of each Platform to Learning")

plt.show()

# cdf

# cdf.memory_usage()
# cdf={}

# mcq['LearningCategorySelftTaught']

# for feature in cat_features :

#     cdf[feature[len('LearningCategory'):]] =mcq[feature].mean()



plt.pie(cdf,labels=cdf.index, )
qc = question.loc[

    question['Column'].str.contains('HardwarePersonalProjectsSelect')]

qc
mcq[mcq['HardwarePersonalProjectsSelect'].notnull()][    

    'HardwarePersonalProjectsSelect'].shape

mcq[mcq['HardwarePersonalProjectsSelect'].notnull()]['HardwarePersonalProjectsSelect']
mcq['HardwarePersonalProjectsSelect'

   ] = mcq['HardwarePersonalProjectsSelect'

    ].astype('str').apply(lambda x:

                          x.split(','))

s = mcq.apply(lambda x: 

              pd.Series(x['HardwarePersonalProjectsSelect']), 

                        axis =1).stack().reset_index(level=1, drop=True)



s.name = 'hardware'   



s

                 
s = s[s !='nan']

s

pd.DataFrame(s.value_counts())
plt.figure(figsize=(10,8))

data = mcq['TimeSpentStudying'].value_counts()

data

sns.countplot(y='TimeSpentStudying',

            data=mcq,

              hue='EmploymentStatus'

             ).legend(loc='upper right',

                     bbox_to_anchor=(1.05,0.5))

        
full_time = mcq.loc[(mcq['EmploymentStatus'] == 'Employed full-time')]



looking_for_job= mcq.loc[(mcq['EmploymentStatus'] ==

                         'Not employed, but looking for work')]

print(full_time.shape)

looking_for_job.shape
figure , (ax1, ax2) = plt.subplots(ncols=2)



figure.set_size_inches(12,5)



sns.countplot(x='TimeSpentStudying',

             data=full_time ,

             hue = 'EmploymentStatus', 

              ax=ax1)



sns.countplot(x='TimeSpentStudying' ,

             data=looking_for_job,

             hue = 'EmploymentStatus', 

              ax= ax2)