import pandas as pd
data=pd.read_csv("../input/FIFA 2018 Statistics.csv")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from pandas.plotting import scatter_matrix
data.head()
print(list(data.columns))
data['Ball Possession %'].sort_values().head()
data.plot.scatter('Ball Possession %', 'Passes')
data[data['Ball Possession %']>74]
data.plot.scatter('Ball Possession %', 'Attempts')
data2=pd.DataFrame(data[['Goal Scored','Ball Possession %','Attempts','On-Target','Off-Target',
            'Blocked','Offsides','Free Kicks','Saves','Pass Accuracy %',
            'Passes','Distance Covered (Kms)']])
print(list(data2.columns))
plt.figure(figsize=(10,10))
for i,col in enumerate(data2.columns):
    plt.subplot(3,4,i+1)
    sns.kdeplot(data2[col],legend=True)
    
data[data['Distance Covered (Kms)']>( data['Distance Covered (Kms)'].mean()+data['Distance Covered (Kms)'].std() ) ][['Distance Covered (Kms)','Team','Opponent']]
group_stage=pd.DataFrame(data.iloc[:96])
#creating a new column, then looping and deciding the winner according to goal counts
group_stage['Result']=np.nan
for i in range(0,96,2):
    a_goals=group_stage.iloc[i]['Goal Scored']
    b_goals=group_stage.iloc[i+1]['Goal Scored']
    if a_goals>b_goals:
        group_stage.iloc[i,-1]=1
        group_stage.iloc[i+1,-1]=-1
    elif b_goals>a_goals:
        group_stage.iloc[i,-1]=-1
        group_stage.iloc[i+1,-1]=1
    else:
        group_stage.iloc[i,-1]=0
        group_stage.iloc[i+1,-1]=0
group_stage['Result'].value_counts()[1:].plot(kind='pie', figsize=(6,6),labels=['win','draw'])
plt.figure()
plt.title('Ball Possession of different results')
group_stage[group_stage['Result']==1]['Ball Possession %'].plot(kind='kde',legend=True,label='wins')
group_stage[group_stage['Result']==-1]['Ball Possession %'].plot(kind='kde',legend=True,label='losses')
group_stage[group_stage['Result']==0]['Ball Possession %'].plot(kind='kde',legend=True,label='draws')
def plot_cols(arr):
    plt.figure(figsize=(10,10))
    for i in range(len(arr)):
        plt.subplot(3,3,i+1)
        plt.title(arr[i])
        group_stage[group_stage['Result']==1][arr[i]].plot(kind='kde',legend=True,label='wins')
        group_stage[group_stage['Result']==-1][arr[i]].plot(kind='kde',legend=True,label='losses')
        group_stage[group_stage['Result']==0][arr[i]].plot(kind='kde',legend=True,label='draws')
plot_cols(['Attempts','Off-Target','Offsides','Off-Target','Passes','Corners','Pass Accuracy %','Distance Covered (Kms)','Saves'])
team_indexed=group_stage.set_index('Team')
pnts=[]
for x in group_stage.set_index('Team').index:
    pnts.append(list((group_stage.set_index('Team').loc[x]['Distance Covered (Kms)'])))
pnts[:10]
plt.figure(figsize=(8,8))
plt.title("Distance Covered during 3 matches")
for i in pnts:
    plt.plot([1,2,3],i)