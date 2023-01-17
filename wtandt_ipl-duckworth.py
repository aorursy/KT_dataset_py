#start of duck worth analysis for ipl





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#import ggplot



import matplotlib.pyplot as plt

%matplotlib inline

#plotting code



plt.style.use('ggplot')

import seaborn as sns



delivdf  = pd.read_csv('../input/deliveries.csv')



df = delivdf[['match_id','inning','total_runs','over','ball','player_dismissed']]



def wicket_test(row):

    if isinstance(row['player_dismissed'], str) == True:

    #if np.isnan(row['player_dismissed']) == True:

        return 1

    else:

        return 0



df['wicket'] = df.apply(wicket_test, axis=1)

df['cum_ball'] = df.groupby(['match_id','inning'])['ball'].cumcount()

df['cumsum'] = df.groupby(['match_id','inning'])['total_runs'].cumsum()

df['cum_wickets'] = df.groupby(['match_id','inning'])['wicket'].cumsum()

df['overs_remaining'] = 20 - df['over']

df['wickets_remaining'] = 10 - df['cum_wickets']

#maxdf = df.groupby(['match_id','inning','over','ball'], as_index=False )['cumsum'].max()

maxdf = df.groupby(['match_id','inning'], as_index=False )['cumsum'].max()



df1 = pd.merge(df, maxdf, on=['match_id','inning'], how='outer')

df1['delta']  = df1['cumsum_y'] - df1['cumsum_x']

#add avg score

df1['deltares']  = df1['delta']/150

df1['resources'] = (df['wickets_remaining']*df1['overs_remaining'])/(10*20)

#print df1

df2 = pd.pivot_table(df1, values='delta', columns=['wickets_remaining'], index=['overs_remaining'], aggfunc=np.sum)

df3 = pd.pivot_table(df1, values='delta', columns=['wickets_remaining'], index=['overs_remaining'], aggfunc=np.average)

#plt.interactive(False)

#print df2

#df2 = df2.replace(r'\s+', np.nan, regex=True)

#plt.style.use(['dark_background'])

df2.plot().invert_xaxis()

df3.plot().invert_xaxis()
df4 = pd.pivot_table(df1, values='deltares', columns=['wickets_remaining'], index=['overs_remaining'], aggfunc=np.sum)

df5 = pd.pivot_table(df1, values='deltares', columns=['wickets_remaining'], index=['overs_remaining'], aggfunc=np.average)



df4.plot().invert_xaxis()

df5.plot().invert_xaxis()
