import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import DataFrame,merge
style.use("ggplot")
%matplotlib inline
animals = pd.read_csv("../input/train.csv")
AnimalType = animals['AnimalType'].value_counts() 
AnimalType.plot(kind='bar',color='#34ABD8',rot=0)


AnimalType = animals.OutcomeType.value_counts().sort_values() 
AnimalType.plot(kind='barh',color='#34ABD8',rot=0)
AnimalType = animals[['AnimalType','OutcomeType']].groupby(['OutcomeType','AnimalType']).size().unstack()
AnimalType.plot(kind='bar',color=['#34ABD8','#E98F85'],rot=-30)
SexuponOutcome = animals['SexuponOutcome'].value_counts()
SexuponOutcome.plot(kind='bar',color=['#34ABD8'],rot=-30)
sexType = animals['SexuponOutcome'].unique()
print(sexType)

M_F = {'Neutered Male':'Male','Spayed Female':'Female','Intact Male':'Male','Intact Female':'Female','Unknown':'Unknown'}
N_T = {'Neutered Male':'Neutered','Spayed Female':'Neutered','Intact Male':'Intact','Intact Female':'Intact','Unknown':'Unknown'}

animals['Sex'] = animals.SexuponOutcome.map(M_F)
animals['Neutered'] = animals.SexuponOutcome.map(N_T)


Sex = DataFrame(animals.Sex.value_counts())
Neutered = DataFrame(animals.Neutered.value_counts())
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.bar([1,2,3],Sex['Sex'],align='center')
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(Sex.index)
ax2.bar([1,2,3],Neutered['Neutered'],align='center')
ax2.set_xticks([1,2,3])
ax2.set_xticklabels(Neutered.index)


df = DataFrame(animals[['Sex','OutcomeType']])
#df.plot(kind='bar')
OutcomeSex = df.groupby(['Sex','OutcomeType']).size().unstack()
OutcomeSex.plot(kind='bar',color=['#34ABD8','#E98F85','r'],rot=-30)
df = DataFrame(animals[['Sex','OutcomeType']])
SexOutcome = df.groupby(['OutcomeType','Sex']).size().unstack()
SexOutcome.plot(kind='bar',rot=-30)

OT_N = animals[['OutcomeType','Neutered']].groupby(['Neutered','OutcomeType']).size().unstack()
OT_N.plot(kind='bar',rot=-30)
DC = animals[['OutcomeType','Neutered','AnimalType']].groupby(['AnimalType','OutcomeType','Neutered']).size().unstack().unstack()
DC.plot(kind='bar',stacked=False,figsize=(10,8),rot=-30)
