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

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
question = pd.read_csv('../input/kaggle-survey-2017/schema.csv')

question.shape
question.tail()
question.head()
mcq=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding="ISO-8859-1",low_memory=False)
mcq.shape
mcq.columns

mcq.head()
import missingno as msno

#missingno는 NaN 데이터들에 대해 시각화를 해준다.

msno.matrix(mcq,figsize=(12,5))
#sns

sns.countplot(y='GenderSelect',data=mcq)
#국가별 응답수

con_df=pd.DataFrame(mcq['Country'].value_counts())

#print(con_df)

#'country' 컬럼을 인덱스로 지정해주고

con_df['국가']=con_df.index

#컬럼의 순서대로 응답 수 , 국가로 컬럼명을 지정해줌

# con_df.columns = ['응답 수','국가']



#index 컬럼을 삭제하과 순위를 알기위해 reset_index()를 해준다.

#우리나라는 18위고 전체52개국에서 참여했지만 , 20위 까지만 본다.

con_df=con_df.reset_index().drop('index',axis=1)

con_df.head(20)
mcq['Age'].describe()
sns.distplot(mcq[mcq['Age']>0]['Age'])
sns.countplot(y='FormalEducation' , data=mcq)
#value counts 를 사용하면 그룹화된 데이터의 카운트 값을 보여준다.

#normalize=True 옵션을 사용하면,

#해당 데이터가 전체 데이터에서 어느정도의 비율을 차지하는지 알 수 있다.



mcq_major_count = pd.DataFrame(

    mcq['MajorSelect'].value_counts())

mcq_major_percent= pd.DataFrame(

    mcq['MajorSelect'].value_counts(normalize=True))

mcq_major_df = mcq_major_count.merge(

    mcq_major_percent, left_index = True,right_index=True)

mcq_major_df.columns = ['응답 수', '비율']

mcq_major_df
#재학주인 사람들의 전공 현황



plt.figure(figsize=(6,8))

sns.countplot(y='MajorSelect' ,data=mcq)
mcq_es_count = pd.DataFrame(

    mcq['EmploymentStatus'].value_counts())

mcq_es_percent = pd.DataFrame(

    mcq['EmploymentStatus'].value_counts(normalize=True))

mcq_es_df=mcq_es_count.merge(

    mcq_es_percent, left_index = True,right_index=True)

mcq_es_df.columns = ['응답 수','비율']

mcq_es_df
sns.countplot(y='EmploymentStatus', data=mcq)
korea= mcq.loc[(mcq['Country']=='South Korea')]



print('The number of Korea Interviwers: ' + str(korea.shape[0]))



sns.distplot(korea['Age'].dropna())

plt.title('Korean')

plt.show()
pd.DataFrame(korea['GenderSelect'].value_counts())
sns.countplot(x='GenderSelect', data=korea)

plt.title('Korean')
figure, (ax1, ax2) = plt.subplots(ncols=2)



figure.set_size_inches(12,5)



sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Female'].dropna(),

                norm_hist=False,color=sns.color_palette("Paired")[4],ax=ax1)



plt.title('korean Female')



sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Male'].dropna(),

                norm_hist=False, color=sns.color_palette("Paired")[0],ax=ax2)

plt.title('korean male')
sns.barplot(x=korea['EmploymentStatus'].unique() , y = korea['EmploymentStatus'].value_counts(normalize=True))

plt.xticks(rotation=30 , ha= 'right')

plt.title('Employment status of the korean')

plt.ylabel('')

plt.show()
korea['StudentStatus']=korea['StudentStatus'].fillna('No')

sns.countplot(x='StudentStatus',data=korea)

plt.title('korean')

plt.show()