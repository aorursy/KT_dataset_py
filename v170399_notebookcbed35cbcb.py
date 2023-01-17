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
train_data=pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")

train_data["avg"]=0

for i in range(0,len(train_data["math score"])):

    train_data["avg"][i]=(train_data["math score"][i]+train_data["reading score"][i]+  train_data["writing score"][i])/3

    



train_data.head()
###visulationTechniquesss

import matplotlib.pyplot as plt

import seaborn as sns

train_data["Report"]="Fail"

for i in range(0,len(train_data["avg"])):

    if train_data["avg"][i]>=75:

        train_data["Report"][i]="Distinction"

    elif train_data["avg"][i]>=50 and train_data["avg"][i]<=75:

        train_data["Report"][i]="First Class"

    elif train_data["avg"][i]>=35 and train_data["avg"][i]<50:

        train_data["Report"][i]="Second Class"

    if train_data["avg"][i]<35:

        train_data["Report"][i]="Fail"

        

sns.factorplot("Report",'avg',data=train_data,k="hue")



    

        
import matplotlib.pyplot as plt



sns.catplot("gender",data=train_data,kind="count")

sns.catplot("race/ethnicity",data=train_data,kind="count")

sns.catplot("Report",data=train_data,kind="count")

#gendervscategories

sns.factorplot('Report',hue="gender",kind="count",data=train_data)

#racevsmarks

sns.factorplot('Report',hue="race/ethnicity",kind="count",data=train_data)





#tryingtounderstandtherelationbwnscoresandattritbutes

sns.boxplot(x="race/ethnicity",y="math score",data=train_data,hue="race/ethnicity")

sns.boxplot(x="lunch",y="writing score",data=train_data,hue="lunch")

sns.boxplot(x="gender",y="reading score",data=train_data,hue="gender")

sns.boxplot(x="lunch",y="math score",data=train_data,hue="lunch")

#there is considerable differences i.e people with standard lunch are perfoming better in math
sns.boxplot(x="test preparation course",y="writing score",data=train_data,hue="test preparation course")

#there is considerable differences i.e people with prepreation   are socring high in exams

from scipy.stats import pearsonr 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for i in ["gender","race/ethnicity","parental level of education","test preparation course","lunch"]:

    train_data[i]=le.fit_transform(train_data[i])

    for j in ["reading score","writing score","math score","Report"]:

        if j=="Report":

                train_data[j]=le.fit_transform(train_data[j])

        corr,pear=pearsonr(train_data[i], train_data[j])

        print(i,j,corr)
