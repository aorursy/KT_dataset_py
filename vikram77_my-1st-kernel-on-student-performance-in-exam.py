import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#Import data
df = pd.read_csv('../input/StudentsPerformance.csv')
#shape of dataframe (Number of Rows and Columns)
df.shape
# Viewing sample of data set
df.head()
# Description of the data set
df.describe(include='all')
# Find any missing value (0-no missing value)
df.isnull().sum()
#Creating a new column of 'total score'
df['total score']=(df['math score']+df['reading score']+df['writing score'])/3

#New Column for Pass status, holds value as P > Pass and F > Fail
passing_marks=35
df['pass status'] = np.where(df['total score']<passing_marks,'F','P')
#Gender

gender_s=df.groupby('gender') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('gender')['total score'].mean()
gender_s.plot(kind='bar')
tot.plot(kind='line', color='black') #total score line plot
plt.show()

#Plot Gender with respect to pass status to see which gender performed well.
gender_ps=df.groupby(['gender','pass status']).size().to_frame('count').reset_index()

"""Here i'm creating a second plot
to see how many males and female pass/failed the test
I should have have used just count but that could have been biased 
because no. of male/female differ and that is what not required to be done
so here i'm segregating in terms of percent which removes the bias
I'll do this same step for rest of the four variables
"""


gender_ps2=gender_ps.groupby('gender').agg('sum').reset_index()
result = pd.merge(gender_ps, gender_ps2, how='left', on=['gender'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['gender'] = result["gender"]+ " : " + result["pass status"].map(str)
sns.barplot(x='gender',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()
# RACE/ETHNICITY

race_s=df.groupby('race/ethnicity') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('race/ethnicity')['total score'].mean()

#Plot race/ethnicity with average score in all three subjects and also averall total score average.
race_s.plot(kind='bar')
tot.plot(kind='line', color='black') #total score line plot
plt.show()

#Plot race/ethnicity with respect to pass status to see which race performed well.
race_ps=df.groupby(['race/ethnicity','pass status']).size().to_frame('count').reset_index()
race_ps2=race_ps.groupby('race/ethnicity').agg('sum').reset_index()
result = pd.merge(race_ps, race_ps2, how='left', on=['race/ethnicity'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['race/ethnicity'] = result["race/ethnicity"]+ " : " + result["pass status"].map(str)
sns.barplot(x='race/ethnicity',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()
# PARENTAL LEVEL OF EDUCATION

parent_s=df.groupby('parental level of education') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('parental level of education')['total score'].mean()
parent_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.xticks(rotation=90)
plt.show()

#Plot parental level of education with respect to pass status to see which level make any difference.
ploe_ps=df.groupby(['parental level of education','pass status']).size().to_frame('count').reset_index()
ploe_ps2=ploe_ps.groupby('parental level of education').agg('sum').reset_index()
result = pd.merge(ploe_ps, ploe_ps2, how='left', on=['parental level of education'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['parental level of education'] = result["parental level of education"]+ " : " + result["pass status"].map(str)
sns.barplot(x='parental level of education',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()
# LUNCH

lunch_s=df.groupby('lunch') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('lunch')['total score'].mean()
lunch_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.show()

#Plot lunch with respect to pass status to see which category performed well.
lunch_ps=df.groupby(['lunch','pass status']).size().to_frame('count').reset_index()
lunch_ps2=lunch_ps.groupby('lunch').agg('sum').reset_index()
result = pd.merge(lunch_ps, lunch_ps2, how='left', on=['lunch'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['lunch'] = result["lunch"]+ " : " + result["pass status"].map(str)
sns.barplot(x='lunch',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()
# TEST PREPARATION

testpre_s=df.groupby('test preparation course') ['math score','reading score', 'writing score'].mean()
tot=df.groupby('test preparation course')['total score'].mean()
testpre_s.plot(kind='bar')
tot.plot(kind='line') #total score line plot
plt.show()

#Plot 'test preparation course' with respect to pass status.
testpre_ps=df.groupby(['test preparation course','pass status']).size().to_frame('count').reset_index()
testpre_ps2=testpre_ps.groupby('test preparation course').agg('sum').reset_index()
result = pd.merge(testpre_ps, testpre_ps2, how='left', on=['test preparation course'])
result['percentage'] = (result['count_x']/result['count_y'])*100
result['test preparation course'] = result["test preparation course"]+ " : " + result["pass status"].map(str)
sns.barplot(x='test preparation course',y='percentage', data=result)
plt.xticks(rotation=90)
plt.show()
# Top20 Math Score

df.sort_values(by='math score', ascending=False).head(20)
# Top20 Reading Score

df.sort_values(by='reading score', ascending=False).head(20)
# Top20 Writing Score

df.sort_values(by='writing score', ascending=False).head(20)
# Top20 Student (total score)

df.sort_values(by='total score', ascending=False).head(20)
# Bottom20 Student (total score)

df.sort_values(by='total score', ascending=True).head(20)