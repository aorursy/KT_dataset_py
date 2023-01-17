import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

df=pd.read_excel("../input/CNumerical.xls", "Sheet1")
print(df.head(10))
count_col_age=df['age'].value_counts()
print(count_col_age)
df.describe()
df['age'].hist(bins=50)

df['blood pressure'].hist(bins=50)
df.boxplot(column='age')
df.boxplot(column='sex', by = 'age')
df.boxplot(column='age', by='class')
df.boxplot(column='class')
plt.show()

#Categorical variable analysis 1

temp1 = df['sex'].value_counts(ascending=True)
temp2 = df[2:307].pivot_table(values='class', index=['sex'],aggfunc=lambda x: x.map({1:1, 0:0}).mean())
print ('Frequency Table for sex:')
print (temp1)

print ('\nProbility of getting blood pressure for each sex class:')
print (temp2)

#Categorical variable analysis 2
temp3 = df['age'].value_counts(ascending=True)
blood_sugar = df[2:307].pivot_table(values='Fasting blood sugar <120', index=['age'], aggfunc=lambda x: x.map({1:1, 0:0}).mean())

print('The frequency table for Age ')
print(temp3)

print('\n Probility of getting Blood Sugar Less than 120 for each age class')
print(blood_sugar)
#Bar Chart for Age
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Sex')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by age and sex ")
temp1.plot(kind='bar')


#Bar Chart for
ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Sex')
ax2.set_ylabel('Probility of getting blood pressure')
ax2.set_title('Probility of getting blood pressure by sex class')
#Stack Charts
temp3 = pd.crosstab(df[2:307]['sex'], df[2:307]['class'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# display all null values in xls
missing_values=df.apply(lambda x: sum(x.isnull()), axis=0)
missing_values
