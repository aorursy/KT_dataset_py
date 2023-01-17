

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#importing libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sb

import datetime

#Data preprocessing(cleaning)

def isfloat(value):

    try:

        float(value)

        return True

    except:

        return False

dataset = pd.read_csv('/kaggle/input/astronauts/astronauts.csv')

dataset.head()

dataset.describe()

#group

for index,row in dataset['Group'].iteritems():

    if(isfloat(row)):

        if(float(row)<1.0):

            dataset['Group'].loc[index]=np.NaN

        else:

            dataset['Group'].loc[index]=row

            

    else:

        dataset['Group'].loc[index]=np.NaN

grp=dataset['Group']

m=np.mean(grp)

dataset.Group=dataset.Group.fillna(int(m))

#year

for index,row in dataset['Year'].iteritems():

    if(isfloat(row)):

        if(float(row)<1.0):

            dataset['Year'].loc[index]=np.NaN

        else:

            dataset['Year'].loc[index]=row

            

    else:

        dataset['Year'].loc[index]=np.NaN

grp=dataset['Year']

m=np.mean(grp)

dataset.Year=dataset.Year.fillna(int(m))

#date

for index,row in dataset['Birth Date'].iteritems():

    if(row[2]!='-'):

        datetimeobject = datetime.datetime.strptime(row,'%m/%d/%Y')

        newformat = datetimeobject.strftime('%d-%m-%Y')

        dataset['Birth Date'].loc[index]=newformat

#Military Rank

dataset['Military Rank']=dataset['Military Rank'].fillna('Not Applicable')

dataset['Military Branch']=dataset['Military Branch'].fillna('Not Applicable')

#Courses

from collections import Counter

data = Counter(dataset['Undergraduate Major'])

dataset['Undergraduate Major']=dataset['Undergraduate Major'].fillna(data.most_common(1)[0][0])

data1 = Counter(dataset['Graduate Major'])

#print(data.most_common(1))

#nan=59

dataset['Graduate Major']=dataset['Graduate Major'].fillna(data1.most_common(2)[1][0])

#Missions

dataset['Missions']=dataset['Missions'].fillna('No Missions')

#death date

dataset['Death Date'][281]='04/23/2001'

for index,row in dataset['Death Date'].iteritems():

    if( str(row)[2]=='/'):

        datetimeobject = datetime.datetime.strptime(row,'%m/%d/%Y')

        newformat = datetimeobject.strftime('%d-%m-%Y')

        dataset['Death Date'].loc[index]=newformat

dataset['Death Date']=dataset['Death Date'].fillna('Not Applicable')   

dataset['Death Mission']=dataset['Death Mission'].fillna('Not Applicable') 
plt.figure(figsize=(9,9))

dataset['Space Flight (hr)'].sample(50).plot.bar()

plt.xlabel('Index')

plt.ylabel('Time in Space(hrs)')

plt.title('Astronauts Space flight hours')

plt.show()

for index,row in dataset['Space Flight (hr)'].iteritems():

    m=max(dataset['Space Flight (hr)'])

    if(int(row)==m):

        print("Most time in space:",dataset['Name'][index])

        print(index)
Univ = Counter(dataset['Alma Mater'])

print('Most astronauts are from:',Univ.most_common(1)[0][0],Univ.most_common(1)[0][1])

cnt=dataset['Alma Mater'].value_counts()

plt.figure(figsize=(12,15))

plt.title('CountPlot')

Uni=sb.countplot(y=dataset['Alma Mater'],data=dataset['Alma Mater'],order=cnt.nlargest(30).index,palette='GnBu_d')

plt.show()
df=pd.read_csv('/kaggle/input/astronauts/astronauts.csv')

Sub2=Counter(dataset['Graduate Major'])

print("Most Common subject in Graduate: ",Sub2.most_common(1)[0][0])

CollegeCount = dataset['Graduate Major'].value_counts()

plt.figure(figsize=(12,15))

plt.title('Graduate Major Courses')

SCount = df['Graduate Major'].value_counts().nlargest(50)

Scnt=df['Undergraduate Major'].value_counts().nlargest(50)

Sub=sb.countplot(y=df['Graduate Major'],data=df['Graduate Major'],

    order=SCount.nlargest(30).index)
Sub1=Counter(dataset['Undergraduate Major'])

print("Most Common subject in Undergraduate:",Sub1.most_common(1)[0][0])

plt.figure(figsize=(12,15))

plt.title('Undergraduate Major Courses')

Sub1=sb.countplot(y=df['Undergraduate Major'],data=df['Undergraduate Major'],

                  order=Scnt.nlargest(30).index,palette='rocket')

cnt1=0

cnt2=0

for index,row in dataset['Military Rank'].iteritems():

    if(row=='Not Applicable'):

        cnt1+=1

    else:

        cnt2+=1

print('From Military:',cnt2,'->',(cnt2*100/(cnt1+cnt2)),'%')

print('Not from Military:',cnt1,'->',(cnt1*100/(cnt1+cnt2)),'%')

m=['From Military Service','Not from Military Service']

y1=[cnt2,cnt1]        

plt.bar(m,y1,color='#6ea171')

plt.ylabel('No. of astronauts')

plt.show()
#Which branch?

data = Counter(dataset['Military Branch'])

print(data)

lists = sorted(data.items())

x, y = zip(*lists) 

plt.barh(x, y,color='purple')

plt.xlabel("No. of astronauts")

plt.show()



plt.figure(figsize=(10,5))

RankGraph = sb.countplot(y=df["Military Rank"], data=df['Military Rank'],

                   order=df['Military Rank'].value_counts().index,

                   palette='deep')

plt.show()
Gender = dataset['Gender'].value_counts()

print(Gender)

print("Sex Ratio of NASA Astronauts(M:F):",Gender[0]/Gender[1],":1")

plt.figure(figsize=(8,8))

plt.pie(Gender, labels=Gender.index, autopct='%1.1f%%', shadow=True ,startangle=180)

plt.show()
#Data Grouping

d=pd.crosstab(dataset['Status'],dataset['Gender'],margins=True)

print(d)

width = 0.35

i=np.arange(len(d)-1)

m=d['Male'][:len(d)-1]

f=d['Female'][:len(d)-1]

plt.bar(i,m, width, label='Men')

plt.bar(i+width,f, width, label='Women')

plt.ylabel('Number of astronauts')

plt.title('Grouped Bar Plot')

plt.xticks(i+ width / 2, ('Active', 'Deceased', 'Management','Retired'))

plt.legend(loc='best')

plt.show()
male = []

male_military = []



female = []

female_military = []



gender = df.loc[:,'Gender']

df['Military Rank'].fillna(0)

m= df['Military Rank'].apply(lambda x:0 if type(x)==float else 1)



for i in range(len(gender)):

    if gender[i] == 'Male':

        male.append(i)

    else:

        female.append(i)



for i in range(len(male)):

    for j in range(len(m)):

        if male[i] == j:

            male_military.append(m[j])

            

for k in range(len(female)):

    for l in range(len(m)):

        if female[k] == l:

            female_military.append(m[l])



sum_male_mil = (male_military.count(0),male_military.count(1))

sum_fem_mil = (female_military.count(0),female_military.count(1))





fig = plt.figure(figsize=(12,5))

ax1 = plt.subplot2grid((1,2),(0,0))

plt.pie(sum_male_mil,colors= ("blue","orange"),shadow=True,labels=("Non-Military Service","Military Service"),autopct='%1.1f%%')

plt.title('Male Astronauts in Military Service')

ax1 = plt.subplot2grid((1,2),(0,1))

plt.pie(sum_fem_mil,colors= ("blue","orange"),shadow=True,labels=("Non-Military Service","Military Service"),autopct='%1.1f%%')

plt.title('Female Astronauts in Military Service')

plt.show()
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(dataset["Space Flight (hr)"]))

threshold=3

sf=[]

for index in range(len(z)):

    if(z[index]>3):

        sf.append(index)

print("Outliers for Space Flight (hr) are:")

for i in sf:

    print(dataset['Name'][i],":",dataset["Space Flight (hr)"][i])

dataset.boxplot(column="Space Flight (hr)",by="Status", figsize=(18, 8),patch_artist=True)

plt.ylabel('Space Flight (hr)')

plt.show()
z = np.abs(stats.zscore(dataset["Space Flight (hr)"]))

threshold=3

sw=[]

for index in range(len(z)):

    if(z[index]>3):

        sw.append(index)

print("Outliers for Space Walks (hr) are:")

for i in sw:

    print(dataset['Name'][i],":",dataset["Space Walks (hr)"][i])

b=sb.boxplot(x="Space Walks (hr)",y="Status",data=dataset)

plt.show()
data = Counter(dataset['Missions'])

print(data.most_common(1)[0][0],":",data.most_common(1)[0][1])

print(data.most_common(2)[1][0],":",data.most_common(2)[1][1])

print('Hence',data.most_common(2)[1][0],'has more astronauts than other missions')

lists = (data.items())



x, y = zip(*lists)\



plt.barh(x[:10], y[:10])

plt.xlabel('Count')

plt.show()
plt.scatter(dataset['Space Walks (hr)'],dataset['Space Flight (hr)'],color='red')

plt.title("Scatter Plot")

plt.xlabel('Space Walks(hr)')

plt.ylabel('Space Flights(hr)')

plt.show()

r,p=stats.pearsonr(dataset['Space Flight (hr)'],dataset['Space Walks (hr)'])

print('Pearson coefficient is',r)
from scipy import stats

from statsmodels.stats import weightstats as stests

import math

print('Prediction using Boxplot-1:Active astronauts have spent most time in space than others')

print('H0(Null hypothesis):Astronauts other than active have spent most time in space')

print('H1(Alternate hypothesis):Active astronauts have spent most time in space')

a=[]

r=[]

active=dataset[dataset['Status']=='Active']

rem=dataset[dataset['Status']!='Active']

def samp(size,number_of_samp):

    for i in range(1,number_of_samp+1):

        ac=active['Space Flight (hr)'].sample(size)

        re=rem['Space Flight (hr)'].sample(size)

        a.append(np.mean(ac))

        r.append(np.mean(re))

    u1=np.mean(a)

    u2=np.mean(r)

    s1=np.std(a)

    s2=np.std(r)

    print("H0:Mean of active-Mean of remaining <=0")

    print("H1:Mean of active-Mean of remaining >0")

    plt.hist(a)

    plt.title('Sample of Active Astronauts')

    plt.xlabel('Space Flight (hr)')

    plt.ylabel('Index')

    plt.show()

    plt.hist(r)

    plt.title('Sample of Remaining Astronauts')

    plt.xlabel('Space Flight (hr)')

    plt.ylabel('Index')

    plt.show()

    ztest,pvalue = stests.ztest(x1=a,x2=r,value=0)

    pvalue=(stats.norm.sf(ztest))

    print(pvalue)

    print("Z Score:",ztest)

    print('P value:',"{:.10f}".format(pvalue))



    if pvalue<0.05:

        print("Reject null hypothesis")

        print("H1 is True")

        print("Active astronauts have spent most time in space than others")

    else:

        print("Accept null hypothesis")



samp(40,40)

samp(30,30)