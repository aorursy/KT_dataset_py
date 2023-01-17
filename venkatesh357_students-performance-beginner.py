import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.shape
df.info()
#creating a pass_mark
passmark=40
df["percentage"]=((df["math score"]+df["reading score"]+df["writing score"])/300)*100
df["math_status"]=np.where(df["math score"]<passmark,'F','P')
df["reading_status"]=np.where(df["reading score"]<passmark,'F','P')
df["writing_status"]=np.where(df["writing score"]<passmark,'F','P')
df["pass/fail"]=np.where(df["percentage"]<passmark,'F','P')
df
#analysing_base_on_gender
#math_subject
fig ,ax=plt.subplots(1,3,figsize=(12,8))
sns.countplot(x='gender',data=df,hue='math_status',palette='muted',ax=ax[0])
plt.xlabel('gender(male/female)')
plt.ylabel('number of pass or fail')
#reading_status
sns.countplot(x='gender',data=df,hue='reading_status',palette='muted',ax=ax[1])
plt.xlabel('gender(male/female)')
plt.ylabel('number of pass or fail')
#writing_status
sns.countplot(x='gender',data=df,hue='writing_status',palette='muted',ax=ax[2])
plt.xlabel('gender(male/female)')
plt.ylabel('number of pass or fail')
plt.show()
#analysing_base_on_race
#math_subject
fig ,ax=plt.subplots(1,3,figsize=(12,8))
sns.countplot(x='race/ethnicity',data=df,hue='math_status',palette='muted',ax=ax[0])
plt.xlabel('groups based on race')
plt.ylabel('number of pass or fail')
#reading_status
sns.countplot(x='race/ethnicity',data=df,hue='reading_status',palette='muted',ax=ax[1])
plt.xlabel('groups based on race')
plt.ylabel('number of pass or fail')
#writing_status
sns.countplot(x='race/ethnicity',data=df,hue='writing_status',palette='muted',ax=ax[2])
plt.xlabel('groups based on race')
plt.ylabel('number of pass or fail')
plt.show()
#analysing_base_on_race
#math_subject
fig ,ax=plt.subplots(1,3,figsize=(12,8))
sns.countplot(x='test preparation course',data=df,hue='math_status',palette='muted',ax=ax[0])
plt.xlabel('course')
plt.ylabel('number of pass or fail')
#reading_status
sns.countplot(x='test preparation course',data=df,hue='reading_status',palette='muted',ax=ax[1])
plt.xlabel('course')
plt.ylabel('number of pass or fail')
#writing_status
sns.countplot(x='test preparation course',data=df,hue='writing_status',palette='muted',ax=ax[2])
plt.xlabel('groups based on race')
plt.ylabel('number of pass or fail')
plt.show()







