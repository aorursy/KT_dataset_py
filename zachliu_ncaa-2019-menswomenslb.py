import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir('../input'))

womens_df = pd.read_csv("../input/womens-machine-learning-competition-2019-publicleaderboard.csv")

womens_df = womens_df.sort_values(by='Score') # sort by score first

womens_df = womens_df.drop_duplicates('TeamId') # keep only the best entry

womens_df['ranking'] = range(1,len(womens_df)+1) # add a ranking column

womens_df = womens_df.drop(columns=['TeamId', 'SubmissionDate']) # drop non-informative columns



mens_df = pd.read_csv("../input/mens-machine-learning-competition-2019-publicleaderboard.csv")

mens_df = mens_df.sort_values(by='Score') 

mens_df = mens_df.drop_duplicates('TeamId')

mens_df['ranking'] = range(1,len(mens_df)+1) 

mens_df = mens_df.drop(columns=['TeamId', 'SubmissionDate']) 
merged_df = pd.merge(womens_df, mens_df, how='inner', on='TeamName', suffixes=['_womens', '_mens'])
merged_df['AveragedScore'] = (merged_df['Score_mens']+merged_df['Score_womens'])/2

merged_df = merged_df.sort_values(by='AveragedScore')
print('Among', len(mens_df), 'participants in Mens game and', len(womens_df), 'participants in Womens game,', len(merged_df), 'teams participated in both')

print('that is', '{:.1f}%'.format(len(merged_df)/len(mens_df)*100) , 'in Mens and', '{:.1f}%'.format(len(merged_df)/len(womens_df)*100), 'in Womens')

sns.lmplot(x="ranking_womens", y="ranking_mens", data=merged_df);
scipy.stats.pearsonr(merged_df['ranking_womens'], merged_df['ranking_mens'])
sns.lmplot(x="Score_womens", y="Score_mens", data=merged_df);

ax = plt.axis('equal');

plt.plot(ax[0:2], ax[0:2], 'k--');

print(scipy.stats.pearsonr(merged_df['Score_womens'], merged_df['Score_mens']))
merged_df[merged_df.ranking_womens<=10].sort_values(by='ranking_womens')
merged_df[merged_df.ranking_mens<=10].sort_values(by='ranking_mens')
merged_df = merged_df.reset_index(drop='True')

merged_df['ranking_both'] = range(1,len(merged_df)+1) 

merged_df.head(60)