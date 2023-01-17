import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/StudentsPerformance.csv')

data.head()
print('Math mean: ',data['math score'].mean(),

      'Reading mean: ',data['reading score'].mean(),'Writing mean:',data['writing score'].mean())

print('Number of female students in the sample: ',len(data.loc[data['gender']=='female']))
high_school=data.loc[data['parental level of education']=='high school']

associates=data.loc[data['parental level of education']=="associate's degree"]

bachelor=data.loc[data['parental level of education']=="bachelor's degree"]

some_college=data.loc[data['parental level of education']=='some college']

some_high_school=data.loc[data['parental level of education']=='some high school']

master=data.loc[data['parental level of education']=="master's degree"]

parental_education=['some_high_school','high_school','some_college','associates','bachelor','master']

edu_list=[len(some_high_school),len(high_school),len(some_college),len(associates),len(bachelor),len(master)]

plt.pie(edu_list,labels=parental_education)

plt.show()
plt.hist(x=data['math score'], bins='auto', color='orange',

                            alpha=0.7, rwidth=0.85)

plt.xlabel('Math Score')

plt.ylabel('Number of students')

plt.title('Distribution on math test')

plt.text(45,99,'Mean 66')

plt.show()
parental_edu_score_rubric={'some high school':0,

                           'high school':1,

                           'some college':2,

                           "associate's degree":3,

                           "bachelor's degree":4,

                           "master's degree":5}

lunch_rubric={'standard':1,'free/reduced':0}

course_rubric={'none':0,'completed':1}

gender_rubric={'female':0,'male':1}
lst=[]

for i in range(len(data)):

    lst.append(data['math score'][i]+data['writing score'][i]+data['reading score'][i])

    

total_score=pd.Series(lst)
features=['parental_edu','lunch','test_prep','gender']

df=pd.DataFrame(columns=features)

for i in range(len(data)):

    p=data['parental level of education'][i]

    l=data['lunch'][i]

    t=data['test preparation course'][i]

    g=data['gender'][i]

    df_temp=pd.DataFrame([[parental_edu_score_rubric[p],lunch_rubric[l],course_rubric[t],gender_rubric[g]]],

                         columns=features)

    df=df.append(df_temp,ignore_index=True,)

df.head()
model=linear_model.LinearRegression()

regr=model.fit(df,total_score)

print('coef: ',regr.coef_[0],regr.coef_[1],regr.coef_[2],regr.coef_[3])

print('R^2: ',regr.score(df,total_score))
math=np.array(data['math score']).reshape(-1,1)

reading=np.array(data['reading score']).reshape(-1,1)

writing=np.array(data['writing score']).reshape(-1,1)
plt.scatter(data['math score'],data['reading score'],alpha=0.4)

plt.xlabel('Math Score')

plt.ylabel('Reading Score')

plt.title('Reading Score vs. Math Score')

plt.show()



plt.scatter(data['math score'],data['writing score'],alpha=0.4,color='orange')

plt.xlabel('Math Score')

plt.ylabel('Writing Score')

plt.title('Writing Score vs. Math Score')

plt.show()



plt.scatter(data['reading score'],data['writing score'],alpha=0.4,color='green')

plt.xlabel('Reading Score')

plt.ylabel('Writing Score')

plt.title('Writing Score vs. Reading Score')

plt.show()
math_reading=linear_model.LinearRegression()

mr=math_reading.fit(math,reading)

print(mr.score(math,reading))
math_writing=linear_model.LinearRegression()

mwr=math_writing.fit(math,writing)

print(mwr.score(math,writing))
reading_writing=linear_model.LinearRegression()

rw=reading_writing.fit(reading,writing)

print(rw.score(reading,writing))