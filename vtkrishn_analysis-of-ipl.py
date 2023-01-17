import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
matches_df=pd.read_csv('../input/matches.csv')

matches_df.head()
matches_df.values
matches_df.shape
print(matches_df.shape[0])

print(matches_df.shape[1])
print(len(matches_df.columns))

matches_df.columns
matches_df.dtypes
print(any(matches_df['dl_applied'].isna()))


matches_df['dl_applied'].unique()
print(any(matches_df['umpire3'].isna()))

print(matches_df['umpire3'].unique())
matches_df.drop(['umpire3'],axis=1,inplace=True) #axis 1, means along the row
matches_df
#columns with null values

for i in matches_df.columns:

    if any(matches_df[i].isna()):

        print(i)



#simpler way

null_columns=matches_df.columns[matches_df.isnull().any()]

null_columns
#find only having null values

for i in matches_df.columns:

    if any(matches_df[i].isna()):

        print(i +':'+ str(matches_df[i].dtypes))
matches_df.fillna(0,inplace=True) #filling it in place to the original data frame
matches_df.isna().any()
matches_df.head(5)
## Some Basic Analysis
print("Total Seasons :: ",matches_df['season'].nunique())

total_seasons=sorted(matches_df['season'].unique())

print("list of Seasons :: " + str(total_seasons))
print("Total Matches :: ", matches_df.shape[0])
total_cities=matches_df['city'].unique()

print("List of City :: " + str(total_cities))
total_teams=matches_df['team1'].unique()

print("List of Teams :: ",str(total_teams))
toss=matches_df['toss_winner'].describe()

print("Most of the toss win by :: ", toss[2])

print("Win frequency :: ", toss[3])
toss_decision=matches_df['toss_decision'].describe()

print("Most of the team prefered to :: ", toss_decision[2])

print("frequency :: ", toss_decision[3])
match_won=matches_df['winner'].describe()

print("Most of the matches won by  :: ", match_won[2])

print("frequency :: ", match_won[3])
player_match_won=matches_df['player_of_match'].describe()

print("Most of the matches won by  :: ", player_match_won[2])

print("frequency :: ", player_match_won[3])
print("Total venues :: ",matches_df['venue'].nunique())

venue=matches_df['venue'].describe()

print("Most of the matches played at  :: ", venue[2])

print("frequency :: ", venue[3])
print("Total umpires :: ",matches_df['umpire1'].nunique())

umpire1=matches_df['umpire1'].describe()

print("Most of the matches umpired by  :: ", umpire1[2])

print("frequency :: ", umpire1[3])

umpire2=matches_df['umpire2'].describe()

print("Most of the matches leg umpired by  :: ", umpire2[2])

print("frequency :: ", umpire2[3])
matches_df[matches_df['win_by_runs'] > 125].sort_values(by='win_by_runs',ascending=False)
matches_df[matches_df['win_by_wickets'] == 10].sort_values(by='win_by_wickets',ascending=False)
matches_df[matches_df['win_by_wickets']==1].sort_values(by='win_by_wickets',ascending=False)
matches_df[matches_df['result'] != 'normal']
winner_df=matches_df

winner_df['win_count'] = 1

winner_output=winner_df[['winner','win_count']].groupby(['winner']).sum().sort_values(by='win_count',ascending=False)

winner_output=winner_output.reset_index()

display(winner_output.head(1))

display(winner_output.tail(1))
pd.DataFrame(matches_df['winner'].value_counts())
total_df=matches_df

total_df['total_count']=1

total_as_team1=total_df[['team1','total_count']].groupby(['team1']).sum().sort_values(by='total_count',ascending=False)

total_as_team2=total_df[['team2','total_count']].groupby(['team2']).sum().sort_values(by='total_count',ascending=False)

total_df=total_as_team1.append(total_as_team2).reset_index()

total_output=total_df.groupby(['index']).sum().sort_values(by='total_count',ascending=False)

total_output=total_output.reset_index()

total_output
percentage_df=total_output.merge(winner_output, left_on='index',right_on='winner')

percentage_df.drop(columns='winner',inplace=True)

percentage_df['win_percentage']=(percentage_df['win_count']/percentage_df['total_count'])*100

percentage_df=percentage_df.rename(columns={'index':'percentage'})

percentage_df[['percentage','win_percentage']].sort_values(by='win_percentage',ascending=False)
display(matches_df.index)

display(winner_df.index)

display(percentage_df.index)
display(percentage_df.columns)
percentage_df.dtypes
percentage_df.get_dtype_counts()
winner_df.select_dtypes(include=['object']) #option to select the content based on dtypes
percentage_df.values #display values
percentage_df.axes #show list of total axes available
display(percentage_df.size) # count of rows x columns

display(len(percentage_df)) #1 column and row

display(len(percentage_df['percentage'])) #1 column and row
percentage_df.shape
display(percentage_df.memory_usage())

display(percentage_df.memory_usage(index=False))

display(percentage_df.memory_usage(deep=True))
percentage_df.empty
percentage_df.is_copy
dummy_pcnt_df=percentage_df
dummy_pcnt_df['win_percentage']=dummy_pcnt_df['win_percentage'].astype('object')
dummy_pcnt_df
dummy_pcnt_df.dtypes
dummy_pcnt_df.at[0,'win_percentage']
dummy_pcnt_df.head(2)
dummy_pcnt_df.tail(2)
dummy_pcnt_df.infer_objects().dtypes
dummy_pcnt_df.iat[0,3]
df = pd.DataFrame({'a': [1,2,3],'b': [3,4,5],'c':[6,7,8]},index=['one','two','three'])

df
df.loc['two':'three']

df.iloc[:2]
df.insert(column='d',value=[22,33,44],loc=2)
df.insert(column='dd',value=[22,33,44],loc=1)
df
for i,v in df.items():

    print(i,list(v))

for i,v in df.iteritems():

    print(i,list(v))
for i in df.keys():

    print(i)
for i in df.iterrows():

    print(i[1][0])
df
df.where(df['dd'] > 22)
df.mask(df['dd'] > 22)
df.isin([22])
list(df.xs('one'))
list(df.loc['one'])
df.pivot(index='a', columns='b', values='d')
list(pd.date_range(start='1/1/2018', end='1/08/2018',freq='D'))
list(pd.period_range(start='2017-01-01', end='2018-01-01', freq='M'))
s = pd.Series(dummy_pcnt_df['win_count'])
dummy_pcnt_df
s.nlargest()
matches_df['winner'].value_counts()