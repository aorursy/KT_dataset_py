import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
df = pd.read_csv('../input/StudentsPerformance.csv')
df.sample(5)
#Renaming Columns
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore'] 
#Checking whether there are any missing values
df.isna().sum()
#Total Sore Percentage
df['total'] = (df['mathScore']+df['readingScore']+df['writingScore'])/3
df.sample()
#some stats..

df.groupby(['race','parentDegree']).mean()
df.groupby(['gender']).mean()
#relation between gender and course and total
course_gender = df.groupby(['gender','course']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='course', data=course_gender, kind='bar')
#Now we can observe that,Parents degree is also crucial in student's score. 
course_gender = df.groupby(['gender','parentDegree']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='parentDegree', data=course_gender, kind='bar')
df.parentDegree.unique()
for i in range(len(df)):
    if df.iloc[i,2] in ['high school', 'some high school']:
        df.iloc[i,2] = 'No_Degree'
    else:
        df.iloc[i,2] = 'has_Degree'
        
df.sample()
Lunch_course = df.groupby(['lunch','course']).mean().reset_index()
snb.factorplot(x='lunch', y='total', hue='course', data=Lunch_course, kind='bar')
df.parentDegree.value_counts()

#Now we can observe that,Parents degree is also crucial in student's score. 
course_gender = df.groupby(['gender','parentDegree']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='parentDegree', data=course_gender, kind='bar')
race_gender = df.groupby(['gender','race']).mean().reset_index()
snb.factorplot(x='gender', y='total', hue='race', data=race_gender, kind='bar')
final_df = df.groupby(['gender','parentDegree','course','lunch','race']).mean().reset_index()
after_sort = final_df.sort_values(by= ['total'],ascending = False)
after_sort.drop(columns=['mathScore','readingScore','writingScore'],inplace = True)
after_sort
#See, it's clear
print("Top students Performance \n",after_sort[:10])
#Simply, if you complete course, Have standard lunch, you   can score good grades. 
#See, it's clear
print("Bottom students Performance \n",after_sort[-10:][::-1])
#Simply, Lunch, Course are mandatory for scoring good. 

base = pd.get_dummies(final_df,columns=['gender','race','parentDegree','course','lunch'],dtype = int)
base.sample()
base.info()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
train_x,test_x,train_y,test_y = train_test_split(base.iloc[:,4:],base.iloc[:,3],test_size = 0.05)
model = XGBRegressor(max_depth = 6)
model.fit(train_x,train_y)
target = model.predict(test_x)
mean_squared_error(target,test_y)

len(target)
test_y[:5].values
target[:5]
