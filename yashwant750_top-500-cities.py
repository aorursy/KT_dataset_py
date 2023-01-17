

#data analysis
import pandas as pd
import numpy as np
import random as rnd
from subprocess import check_output
#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib  inline
#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn .ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train=pd.read_csv('../input/cities_r2.csv')
train.info()
train.head(10)
train.shape
plt.title("No_of_cities_per_state",fontsize=30,color='navy')
plt.xlabel("state_code",fontsize=20,color='navy')
plt.ylabel("No. of cities",fontsize=20,color='navy')

train['state_code'].value_counts().plot("bar",figsize=(12,6))


total_population=train.groupby('state_code')['population_total'].sum()
total_population_per_state=np.array(total_population)
total_population.sort_values(ascending=False)




plt.figure(figsize=(10,10))
plt.title("population Distriution",fontsize=30,color='navy')
plt.pie(total_population_per_state,labels=total_population.index,shadow=True,autopct='%1.1f%%')
plt.show()




total_city_population=train.groupby('dist_code')['population_total'].sum()
total_city_population_per_state=np.array(total_city_population)
total_city_population.sort_values(ascending=False)


x=train['dist_code']
y=train['population_total']
plt.figure(figsize=(20,10))
plt.bar(x, y, align='center',alpha=0.5)
plt.xticks(x,y)
plt.ylabel('city_total_population',fontsize=30)
plt.xlabel('cities',fontsize=30)
plt.title('City_wise_population_distribution',fontsize=30)
 
plt.show()
total_female_population=train.groupby('state_code')['population_female'].sum()
total_female_population_per_state=np.array(total_female_population)
total_female_population.sort_values(ascending=False)
plt.figure(figsize=(10,10))
plt.title("female_population_distribution",fontsize=30,color='navy')
plt.pie(total_female_population_per_state,labels=total_female_population.index,shadow=True,autopct='%1.1f%%')
plt.show()
total_male_population=train.groupby('state_code')['population_male'].sum()
total_male_population_per_state=np.array(total_male_population)
total_male_population.sort_values(ascending=False)
plt.figure(figsize=(10,10))
plt.title("male_population_distribution",fontsize=30,color='navy')
plt.pie(total_male_population_per_state,labels=total_male_population.index,shadow=True,autopct='%1.1f%%')
plt.show()
train['state_code'].value_counts()
g=sns.FacetGrid(train,row='dist_code',size=2.2,aspect=1.6)
g.map(plt.hist,'sex_ratio',alpha=0.7,bins=20)
g.add_legend()
g=sns.FacetGrid(train,row='state_code',size=2.2,aspect=1.6)
g.map(plt.hist,'sex_ratio',alpha=0.7,bins=20)
g.add_legend()

total_dist_child_sex_ratio=train.groupby('dist_code')['child_sex_ratio'].mean()
total_dist_child_sex_ratio_per_dist=np.array(total_dist_child_sex_ratio)
total_dist_child_sex_ratio.sort_values(ascending=False)
total_state_child_sex_ratio=train.groupby('state_code')['child_sex_ratio'].mean()
total_state_child_sex_ratio_per_dist=np.array(total_state_child_sex_ratio)
total_state_child_sex_ratio.sort_values(ascending=False)
train.describe()
total_city_literacy=train.groupby('dist_code')['effective_literacy_rate_total'].mean()
total_city_literacy_per_state=np.array(total_city_population)
total_city_literacy.sort_values(ascending=False)

g=sns.FacetGrid(train,row='dist_code',size=2.2,aspect=1.6)
g.map(plt.hist,'effective_literacy_rate_total',alpha=0.7,bins=20)
g.add_legend()
total_city_literacy=train.groupby('state_code')['effective_literacy_rate_total'].mean()
total_city_literacy_per_state=np.array(total_city_population)
total_city_literacy.sort_values(ascending=False)
g=sns.FacetGrid(train,row='state_code',size=2.2,aspect=1.6)
g.map(plt.hist,'effective_literacy_rate_total',alpha=0.7,bins=20)
g.add_legend()
train.head()
total_male_graduates=train.groupby('state_code')['male_graduates'].sum()
total_male_graduates_per_state=np.array(total_male_graduates)
total_male_graduates.sort_values(ascending=False)
plt.figure(figsize=(10,10))
plt.title("male_graduates_distribution",fontsize=30,color='navy')
plt.pie(total_male_graduates_per_state,labels=total_male_graduates.index,shadow=True,autopct='%1.1f%%')
plt.show()
total_female_graduates=train.groupby('state_code')['female_graduates'].sum()
total_female_graduates_per_state=np.array(total_female_graduates)
total_female_graduates.sort_values(ascending=False)
plt.figure(figsize=(10,10))
plt.title("female_graduates_distribution",fontsize=30,color='navy')
plt.pie(total_female_graduates_per_state,labels=total_female_graduates.index,shadow=True,autopct='%1.1f%%')
plt.show()
total_graduates=train.groupby('state_code')['total_graduates'].sum()
total_graduates_per_state=np.array(total_graduates)
total_graduates.sort_values(ascending=False)
plt.figure(figsize=(10,10))
plt.title("total_graduates_distribution",fontsize=30,color='navy')
plt.pie(total_graduates_per_state,labels=total_graduates.index,shadow=True,autopct='%1.1f%%')
plt.show()
