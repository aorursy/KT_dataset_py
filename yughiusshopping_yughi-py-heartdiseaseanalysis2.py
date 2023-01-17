# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from plotly import __version__

print(__version__)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
heart= pd.read_csv('../input/heart.csv')

heart.head()

heart.head(6)
heart.tail()
print(heart.shape)
#get info on all the null entries 

heart.info()
heart.describe()
heart.describe().transpose()
len(heart)
len(heart.columns)
heart.dtypes
# get the unique entries in each column

print('unique entries in each column')

heart.nunique()
print(heart.columns)
# distribution of the range of ages for heart attack plot

heart.columns = ['Age', 'Gender', 'ChestPain', 'RestingBloodPressure', 'Cholestrol', 'FastingBloodSugar', 'RestingECG', 'MaxHeartRateAchivied',

       'ExerciseIndusedAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thalassemia', 'Target']
heart.isnull().sum()
#Subsetting by rows

heart[21:26]
heart[ : : -1]
heart[ : : 50]
#Subsetting by columns

heart['RestingBloodPressure']
heart['RestingBloodPressure'].head()
#Select by index

heart.iat[3,4]
heart.iloc[21:26,1:3]
#Subsetting by both

#Needs to make list first ['Age', 'ChestPain']]

heart[4:8][['Age', 'ChestPain']]
#Filtering by rows

heart.RestingBloodPressure < 130
Filter = heart.RestingBloodPressure < 130

heart[Filter]
heart.Cholestrol > 200

Filter2 = heart.Cholestrol > 200

heart[Filter2]
#Filtering by both

#Filter & Filter 2

heart[Filter & Filter2]
heart[(heart.RestingBloodPressure < 130) & (heart.Cholestrol > 200)]
#stats[stats.IncomeGroup == "High income"]

heart[heart.ChestPain == 3]
#Numerical

##Discrete  - "chest_pain" "excercise_angina" "slope" "n_major_vasel" "thal" "target"

##Continuous - "age" "rest_bp" "chol" "fasting_bloodsugar" "max_heartrate" "ST_depression"



#Categorical 

##Nominal 

##Ordinal - "sex" "rest_ecg"



#Did you make univariate plots (histograms, density plots, boxplots)?

#Did you consider correlations between variables (scatterplots)?



#Histograms  1 measure (bin field)

#Bar Charts/count plot, box, violin

##Horizontal Bar Chart  0 or more dimensions, 1 or more measures

##Stacked Bar Chart   1 or more dimensions, 1 or more measures

##Side-by-Side Bar Chart   1 or more dimensions, 1 or more measures

#Scatter Plots   0 or more dimensions, 2–4 measures; color, shape, size

#Line Charts

##Line Chart (Continuous)  1 date, 0 or more dimensions, 1 or more measures

##Line Chart (Discrete)   1 date, 0 or more dimensions, 1 or more measures



##Heat map

##Grid map
#Histograms  "age" 



#Bar Charts

##Horizontal Bar Chart  0 or more dimensions, 1 or more measures

##Vertical Bar Chart   Sex vs Cholestrol,  x=chest_pain,y=chol,fill=sex 

##Stacked Bar Chart   Age Vs cholestrol vs sex

##Side-by-Side Bar Chart   Maximum heart rate achived vs age by gender

##count plot       major vessels and chest pain type, Chest Pain Type Vs Gender

##box 

##violin



#Scatter Plots   age,chol,size=chest_pain; x=age,y=chol,color=sex,size=chest_pain

##Swarmplot      Age only, y=Age, x='ChestPainType'

##Regplot        bp and cholestrol are correlated



#Line Charts

##Line Chart (Continuous)  x=age, y=rest_bp, group=sex

##Line Chart (Discrete)   1 date, 0 or more dimensions, 1 or more measures



##Pie chart       chest pain type

##Heat map

##Grid map
heart.head()
sns.set_style("white")

sns.set_context("paper")

#context poster gives big image
#Histograms  "age" 

heart['Age'].hist(grid=False,bins=50)
plt.hist(heart["Age"], bins=50)
#Histograms  "age"

sns.distplot(heart["Age"],bins=50)
heart["Age"].iplot(kind='hist',bins=50)
heart["Age"].plot.kde()
#Density plot  "age"

sns.kdeplot(heart["Age"])
heart.plot.bar(x='Gender',y='Cholestrol')
#Vertical Bar Chart   Sex vs Cholestrol

sns.barplot(x='Gender',y='Cholestrol',data=heart)
heart.iplot(kind='bar',x='Gender',y='Cholestrol')
#Vertical Bar Chart    x=chest_pain,y=chol,fill=sex 

sns.barplot(x='ChestPain',y='Cholestrol',data=heart,hue='Gender')
#Stacked Bar Chart   Age Vs cholestrol vs sex

heart.plot.bar(x='ChestPain',y='Cholestrol',stacked=True)
#Side-by-Side Bar Chart   Maximum heart rate achived vs chestpain by gender

sns.barplot(x='ChestPain',y='MaxHeartRateAchivied',data=heart, hue='Gender')
#count plot       major vessels and chest pain type

#check the relation of major vessels and chest pain type

ax=sns.countplot(hue='ChestPain',x='MajorVessels',data=heart,palette='husl')
#count plot       Chest Pain Type Vs Gender

# do a gender comparison 

ax = sns.countplot(hue=result,x='Gender',data=heart,palette='husl')



plt.title("Chest Pain Type Vs Gender")    

plt.ylabel("")

plt.yticks([])

plt.xlabel("")

for p in ax.patches:

    ax.annotate(p.get_height(),(p.get_x()+0.05, p.get_height()+1))

ax.set_xticklabels(['Female','Male'])

print(ax.patches)
sns.countplot(hue='ChestPain',x='Gender',data=heart)
#Scatter Plots   x=age,y=chol,color=sex,size=chest_pain

heart.plot.scatter(x='Age',y='Cholestrol', c='Gender', cmap='coolwarm')
#Scatter Plots   age,chol,size=chest_pain

plt.scatter(x='Age',y='Cholestrol',data=heart,c='Gender')

plt.show()
heart.iplot(kind='scatter',mode='markers', x='Age',y='Cholestrol',size=10)
sns.regplot(x=heart["Age"], y=heart["Cholestrol"], fit_reg=False)
#Swarmplot      Age only

# subselect features of interest and then perform classification 

# divide dataset into training, test and validation 

sns.swarmplot(x=heart['ChestPain'], y=heart['Cholestrol'])
#Swarmplot      y=Age, x='ChestPainType'

result=[]

for i in heart['ChestPain']:

    if i == 0:

        result.append('Typical Angina')

    if i ==1:

        result.append('Atypical Angina')

    if i ==2:

        result.append('Non-Anginal')

    if i==3:

        result.append('Asymptomatic')

        

heart['ChestPainType']=pd.Series(result)



sns.swarmplot(x='ChestPainType', y='Age', data=heart)
#Regplot        bp and cholestrol are correlated

#check if bp and cholestrol are correlated

ax = sns.regplot(x='RestingBloodPressure', y='Cholestrol',data=heart, color="g")
heart.plot.line(x='Age',y='RestingBloodPressure',figsize=(10,6),lw=1)
#Line Chart   x=age, y=rest_bp, group=sex

plt.plot( 'Age', 'RestingBloodPressure', data=heart, color='skyblue', alpha=0.8)
sns.lineplot(x="Age", y="RestingBloodPressure", hue="Gender", style="ChestPain", data=heart)
# plot the pie chart indicating distribution of each chest pain type

ChestPain=(heart['ChestPainType']).value_counts()

percent_typAng= ChestPain[0] *100/ len(heart)

percent_AtypAng=ChestPain[1]*100/len(heart)

percent_nonAng=ChestPain[2]*100/len(heart)

percent_none=ChestPain[3]*100/len(heart)



values= [percent_typAng, percent_AtypAng, percent_nonAng, percent_none]

labels=['Typical Angina','Atypical Angina','Non-Anginal','Asymptomatic']

plt.pie(values, labels=labels,autopct='%1.1f%%')

plt.title("Chest Pain Type Percentage")    

plt.show()
sns.pairplot(heart)
sns.heatmap(heart.corr())