# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r"/kaggle/input/covid19updated2/raw_data_upd.csv",engine ='python')

df.head(10)
df.shape

print('The dataset contains', df.shape,' Rows and Columns')
df = df.drop(['Source_1','Source_2','Source_3','State code','Status Change Date','Unnamed: 19','Unnamed: 20','Unnamed: 21','Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25'],axis=1)

df.head()
df.shape
print('Now the updated dataset has', df.shape,'Rows and Columns')
print('The datatypes of the Parameters \n',df.dtypes)
print('All the variables in the dataset except Patient Number are Categorical ')
print('Stats of Number of Patients    \n',df.describe())
df['Age Bracket'].value_counts()
print('Age bracket in the range of ',max(df['Age Bracket'].value_counts()),'are most affected, followed by the old aged')
t = df['Date Announced'].value_counts()

t.plot(kind='bar')
print('We can clearly see that number of cases went on multiplying every single day')

print('Maximum Number of cases',max(t),',were found on 13-04-2020')
(df['Gender'].value_counts()/len(df['Gender'])*100).plot(kind='bar')
print('Almost two and half times in number, males are affected than the number of women affected by the virus ')
print('Percentage of males(M) and females(F) infected in India\n',df['Gender'].value_counts()/len(df['Gender'])*100)
print('States with highest number of cases being detected\n',df['Detected State'].value_counts())
df['Detected State'].value_counts().plot(kind='bar')

plt.ylabel('Count')

plt.title('Detected States')
print('Maharashtra is the most affected state with',max(df['Detected State'].value_counts()),'patients')
z= df['Current Status'].value_counts()

print('Current Status of the patients \n')

print('Total number of patients Hospitalized :',z.Hospitalized)

print('Total number of patients Recovered :',z.Recovered)
df['Current Status'].value_counts().plot(kind='bar')

plt.ylabel('Count')

plt.title('Current Status')
z = df['Nationality'].value_counts()

print('Nationality of Coronavirus affected patients in India \n',z)
df['Nationality'].value_counts().plot.bar()

plt.ylabel('Count')

plt.title('Nationality')
print('Reasons for infection of virus \n')

df['Notes'].value_counts()
tem_df = df.loc[df['Notes']=='Travelled to Delhi']

print('Details of People travelling to Delhi and got affected')

tem_df
print('Totally',len(tem_df),'people got affected who travelled to Delhi')
print('The States that got affected')

z=tem_df['Detected State'].value_counts()

z
print('Totally', len(z),'States got affected') 
print('The States that got affected')

z=tem_df['Detected City'].value_counts()
print('Totally',len(z),'number of cities got affected')
print('Current Status of infected people:', tem_df['Current Status'].value_counts())
print('Number of Males(M) and females(F): \n',tem_df['Gender'].value_counts())
print('Dates on which these cases were found :\n',tem_df['Date Announced'].value_counts())
temp3_df = df.loc[(df['Nationality']=='India') & ((df['Notes']=='Travelled from Dubai') | (df['Notes']=='Travelled from UK'))]

print('Details of people travelling to India from abroad')

temp3_df
print('Total number: \n',temp3_df.shape)
print('Dates on which these cases were found: \n',temp3_df['Date Announced'])
temp4_df = df.loc[df['Nationality']=='India']

z=temp4_df['Notes'].value_counts()

z
print('Totally there are',len(z),'different reasons why Indians got affected')
print('Current Status of these patients:\n',temp4_df['Current Status'].value_counts())
temp4_df['Current Status'].value_counts().plot.bar()

plt.ylabel('Total number')

plt.title('Current Status of Indian patients')
temp5_df = df.loc[df['Current Status']=='Deceased']

temp5_df['Age Bracket'].value_counts().plot.bar()

plt.xlabel('Age Bracket')

plt.ylabel('Total no. of people Deceased')

plt.title('Plot representing Deceased no. of Patients ')
temp6_df = df.loc[df['Current Status']=='Recovered']

temp6_df['Age Bracket'].value_counts().plot.bar()

plt.xlabel('Age Bracket')

plt.ylabel('Total no. of people recovered')

plt.title('Plot representing recovery of Patients ')
temp7_df= df[['Age Bracket','Notes','Current Status']]

temp7_df
print('Details of people travelled from Wuhan\n',temp7_df.loc[temp7_df['Notes']=='Travelled from Wuhan'])
males = df[df['Gender']=='M']

females = df[df['Gender']=='F']



m=males['Current Status'].value_counts()/len(males)*100

f=females['Current Status'].value_counts()/len(females)*100

print('Percentage of Current Status of Male patients in India: \n',m)

print('\n')

print('Percentage of Current Status of Female patients in India: \n',f)
m.plot(kind='bar')

plt.xlabel('Percentage')

plt.title('Percentage of Current Status of Male patients in India:')
f.plot(kind='bar')

plt.xlabel('Percentage')

plt.title('Percentage of Current Status of Female patients in India:')
## Lets look for which districts are most affected in Maharashtra



temp_df=df.loc[df['Detected State']=='Maharashtra']

temp_df
print('Totally',len(temp_df),'people in Maharashtra got affected')
temp_df['Notes'].value_counts()
temp_df.replace({'Details Awaited': 'Details awaited'},inplace = True)

temp_df['Notes'].value_counts()
print('There is no clear information about how the virus affected so effectively in Maharashtra')

print('There are',len(temp_df['Notes'].value_counts()),'different number of reasons for infection')
temp_df['Detected District'].value_counts().plot(kind='bar')

plt.ylabel('Count')

plt.title('Count in Districts of Maharashtra')
print('Mumbai District of Maharashtra with',max(temp_df['Detected District'].value_counts()),'is the most affected')
## Lets look for which city are most affected in Maharashtra



temp_df['Detected City'].value_counts().plot(kind='bar')

plt.ylabel('Count')

plt.title('Count in Cities of Maharashtra')
print('Mumbai City area with',max(temp_df['Detected City'].value_counts()),'is the most affected in Maharashtra')
## Lets look for which districts are most affected in Karnataka



temp2_df=df.loc[df['Detected State']=='Karnataka']

print('Details of patients in Karnataka')

temp2_df
print('Totally',len(temp2_df),'are affected in Karnataka')
temp2_df['Detected District'].value_counts().plot(kind='bar')

plt.ylabel('Count')
df.describe()
w=df.isnull().sum()

print('Huge data is yet to be obtained, there are many empty fields in the dataset\n',w)

tem = ['Date Announced','Detected City','Gender','Detected District','Detected State','Current Status','Nationality']



for i in tem:

    print('--------------********-------------')

    print(df[i].value_counts())

    