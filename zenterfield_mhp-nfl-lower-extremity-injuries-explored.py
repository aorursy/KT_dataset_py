# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plot

import datetime



from scipy.stats import ttest_ind

import statsmodels.api as sm

from statsmodels.formula.api import ols

import statsmodels.api as sm

from statsmodels.graphics.factorplots import interaction_plot

import statistics



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



pd.options.display.float_format = '{:.3f}'.format #prevent scientific notation in dataframes, display #.### instead



%whos #outputs table of variables and their info



## used to expand Jupyter Notebook to full browser width for easier reading of long lines

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



## template for monitoring/recording run-times for blocks of code

overall_startDT = datetime.datetime.now()

print('Started at', overall_startDT)

#code

print(datetime.datetime.now() - overall_startDT)

# usually around 0:##:##.# on my laptop, ##m##s on Kaggle
## import playtrack data

startDT = datetime.datetime.now()

print('Starting import of playtrackDF at', startDT)

#original key without padding

col_dtypes = {'time':float, 

              'x':float, 'y':float, 

              'dir':float, 'dis':float, 'o':float, 's':float}

use_cols = ['PlayKey', 'time', 'dir', 'dis', 'o', 's']

#playtrackDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv', dtype=col_dtypes, usecols=use_cols)

playtrackDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv', dtype=col_dtypes)





## there are 2 rows with NaN values for 'dir' and 'o', they appear to be meaningless glitches and deletable

#print(playtrackDF.columns)

#print(playtrackDF.isna().sum())

#dropableRows = list(playtrackDF[playtrackDF['dir'].isna()].index)

#for row in dropableRows:

#    display(playtrackDF[row-2:row+3])

playtrackDF.dropna(axis='index', subset=['dir', 'o'], inplace=True)



print(playtrackDF.shape)

print(datetime.datetime.now() - startDT, 'to import playtrack data and drop NA rows')

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for initial import')

#usually around 1m45s on Kaggle
## reducing memory usage of this massive dataframe by converting columns to *100 ints

startDT = datetime.datetime.now()

for col in playtrackDF.columns:

    if playtrackDF[col].dtype != object:

        print(col, playtrackDF[col].min(), playtrackDF[col].max())

        playtrackDF[col] = playtrackDF[col]*100

        if playtrackDF[col].min() < 0:

            playtrackDF[col] = playtrackDF[col].astype(np.int16) #stores -32,768 to 32,767

        else:

            playtrackDF[col] = playtrackDF[col].astype(np.uint16) #stores 0-65,535

print(datetime.datetime.now() - startDT, 'to reduce memory usage') ## usually around 15s on Kaggle

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for *100 ints')



## reformatting the PlayKey from 12345-1-1 to 1234512123 to enable storage as number with proper sorting

startDT = datetime.datetime.now()

print('Starting rekeying at', startDT)

playtrackDF.loc[:, 'PlayKey'] = playtrackDF['PlayKey'].apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )

playtrackDF.loc[:, 'PlayKey'] = playtrackDF['PlayKey'].astype(np.int64) #uint32's max value is just below the largest key

print(datetime.datetime.now() - startDT, 'to rekey DF') ## usually around 2m on Kaggle

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs for reduced/rekeyed')
#import injuries and plays data

injuriesDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')

print("injuriesDF shape", injuriesDF.shape)

playsDF = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')

print("playsDF shape", playsDF.shape)



startDT = datetime.datetime.now()

injuriesDF['PlayerKey'] = injuriesDF['PlayerKey'].apply(lambda v: '{0:0>5}'.format(v) )

injuriesDF['GameID'] = injuriesDF['GameID'].apply(lambda v: '{0:0>5}{1:0>2}'.format(*v.split('-') ) )

injuriesDF['PlayKey'].fillna(value='0-0-0', inplace=True)

injuriesDF['PlayKey'] = injuriesDF['PlayKey'].astype(str).apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )

playsDF['PlayerKey'] = playsDF['PlayerKey'].apply(lambda v: '{0:0>5}'.format(v) )

playsDF['GameID'] = playsDF['GameID'].apply(lambda v: '{0:0>5}{1:0>2}'.format(*v.split('-') ) )

playsDF['PlayKey'] = playsDF['PlayKey'].apply(lambda v: '{0:0>5}{1:0>2}{2:0>3}'.format(*v.split('-') ) )

injuriesDF.loc[:, 'PlayKey'] = injuriesDF['PlayKey'].astype(np.int64)

playsDF.loc[:, 'PlayKey'] = playsDF['PlayKey'].astype(np.int64)

print(datetime.datetime.now() - startDT, 'to rekey smaller dataframes') # usually around 1s on Kaggle



print('Dataframes created from injuries and plays CSVs (playtrack CSV to be handled separately due to size)')
#collapse player-level info into smaller DF

playersDF = playsDF.groupby(by=['PlayerKey', 'RosterPosition']).size().reset_index().rename(columns={0:'Plays'})

print("playersDF shape", playersDF.shape)



#collapse game-level info into smaller DF

groupCols = [playsDF['PlayerKey'], playsDF['GameID'], playsDF['StadiumType'].fillna('unknown'), playsDF['FieldType'], playsDF['Temperature'], playsDF['Weather'].fillna('unknown'), playsDF['PlayerDay']]

gamesDF = playsDF.groupby(by=groupCols).size().reset_index().rename(columns={0:'Plays'})

print("gamesDF shape", gamesDF.shape)

gamesDF = gamesDF.sort_values(by=['PlayerKey', 'PlayerDay'])

gamesDF.reset_index(drop=True, inplace=True)



print(len(injuriesDF['PlayerKey'].unique()), 'unique players in injuriesDF')



display(gamesDF.head())
injuryGames = pd.DataFrame(injuriesDF.groupby(by=['GameID']).size())

injuryGames.reset_index(inplace=True)

gamesDF['InjOcc'] = 0

for idVal in injuryGames['GameID']:

    gamesDF.loc[gamesDF['GameID'] == idVal, 'InjOcc'] = 1

## there's 104 unique GameIDs for injured players and 5712 unique GameIDs in the play data (gamesDF)



## cleaning StadiumType and determining FieldExposed

#print(gamesDF['StadiumType'].unique())

outdoorList = ['Open', 'Outdoor', 'Oudoor', 'Outdoors', 'Ourdoor', 'Outddors', 'Heinz Field', 'Outdor', 'Outside']

indoorList = ['Indoors', 'Closed Dome', 'Domed, closed', 'Dome', 'Indoor', 'Domed', 'Retr. Roof-Closed', 'Outdoor Retr Roof-Open', 'Indoor, Roof Closed', 'Retr. Roof - Closed', 'Retr. Roof-Open', 'Dome, closed', 'Indoor, Open Roof', 'Domed, Open', 'Domed, open', 'Retr. Roof - Open', 'Retr. Roof Closed', 'Retractable Roof']

unknownList = ['unknown', 'Bowl', 'Cloudy']

gamesDF.loc[gamesDF['StadiumType'].isin(outdoorList), 'FieldExposed'] = 1

gamesDF.loc[gamesDF['StadiumType'].isin(indoorList), 'FieldExposed'] = 0

gamesDF.loc[gamesDF['StadiumType'].isin(unknownList), 'FieldExposed'] = 1 ## assuming exposed b/c not specified

print('FieldExposed vals:', gamesDF['FieldExposed'].unique())



## cleaning Weather and determining FieldWet

wetDescs = []

dryDescs = []

unkDescs = []

for desc in list(gamesDF['Weather'].unique()):

    if 'rain' in desc or 'Rain' in desc or 'showers' in desc or 'Showers' in desc or 'snow' in desc or 'Snow' in desc:

        wetDescs.append(desc)

    else: 

        if 'unknown' in desc or 'Unknown' in desc:

            unkDescs.append(desc)

        else:

            dryDescs.append(desc)

gamesDF.loc[(gamesDF['Weather'].isin(wetDescs)) & (gamesDF['FieldExposed'] == 1), 'FieldWet'] = 1

gamesDF.loc[gamesDF['Weather'].isin(dryDescs), 'FieldWet'] = 0

gamesDF.loc[gamesDF['FieldExposed'] == 0, 'FieldWet'] = 0

gamesDF.loc[gamesDF['Weather'].isin(unkDescs), 'FieldWet'] = 0 ## assuming field is not wet b/c not specified

print('FieldWet vals:', gamesDF['FieldWet'].unique())



## cleaning Temperature

avgIndoorTemp = gamesDF.loc[(gamesDF['FieldExposed'] != 1) & (gamesDF['Temperature'] != -999), 'Temperature'].mean()

#print('Avg (valid) temp for indoor games is', round(avgIndoorTemp, 0))

gamesDF.loc[(gamesDF['FieldExposed'] != 1) & (gamesDF['Temperature'] == -999), 'Temperature'] = round(avgIndoorTemp, 0)



## making a one-hot variable for turf type

print('FieldType vals:', gamesDF['FieldType'].unique())

gamesDF['Synthetic'] = 1 * (gamesDF['FieldType'] == 'Synthetic')



display(gamesDF.head())
print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs before additional columns')

startDT = datetime.datetime.now()

playtrackDF.loc[:, 'twist'] = abs(playtrackDF['dir'].astype(np.int32) - playtrackDF['o'].astype(np.int32))

playtrackDF.loc[playtrackDF['twist'] > 18000, 'twist'] = 36000 - playtrackDF.loc[playtrackDF['twist'] > 18000, 'twist']

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs with twist col added')

playtrackDF.loc[:, 'twist'] = playtrackDF['twist'].astype(np.int16)

print(datetime.datetime.now() - startDT, 'to calculate twist column') ## usually around 5s on Kaggle

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs with twist col as int16')



cols_for_diffs = ['dir', 'o', 's', 'twist']



for col in cols_for_diffs:

    startDT = datetime.datetime.now()

    playtrackDF.loc[:, 'd_'+col ] = abs(playtrackDF[col].astype(np.int32).diff())

    playtrackDF.loc[0, 'd_'+col ] = 0 ## these rows represent a new play and shouldn't be compared to the row above

    playtrackDF.loc[:, 'd_'+col] = playtrackDF['d_'+col].astype(np.int32)

    print(datetime.datetime.now() - startDT, 'to calculate', 'd_'+col, 'column') ## each around 2-5s on Kaggle

    

print(round(playtrackDF.memory_usage().sum() / 1024**3, 3), 'GBs currently')



display(playtrackDF.head())
for col in playtrackDF.columns:

    if playtrackDF[col].dtype != object:

        print(col, playtrackDF[col].dtype, round(playtrackDF[col].memory_usage() / 1024**3, 3), 'GBs', playtrackDF[col].min(), playtrackDF[col].max() )

    else:

        print(col, playtrackDF[col].dtype, round(playtrackDF[col].memory_usage() / 1024**3, 3), 'GBs' )
startDT = datetime.datetime.now()



playLens = pd.DataFrame(playtrackDF[['PlayKey']].groupby(by=['PlayKey']).size())

playLens.columns = ['obs']



## any stats here are divided by 100 to account for earlier *100 for sake of reduced memory via int datatypes 



## identifying the field "coordinates" for the player's final location per play

playFinalLocs = playtrackDF[['PlayKey', 'x', 'y']].groupby(by=['PlayKey']).tail(1).set_index('PlayKey')/100

## assuming each quadrant of the field is interchangeable, the location values should be 

## relative to midfield rather than a corner of the field

playFinalLocs.loc[:, 'x'] = abs(playFinalLocs['x'] - (120/2)) # converting to yds away from midfield

playFinalLocs.loc[:, 'y'] = abs(playFinalLocs['y'] - (53.3/2)) # converting to yds away from midfield

playFinalLocs.columns = 'rel_' + playFinalLocs.columns + '_final'



## aggregate between-observation changes per play, first as sums...

playSums = playtrackDF[['PlayKey','d_dir', 'd_o', 'd_s', 'd_twist']].groupby(by=['PlayKey']).sum()/100

playSums.columns = playSums.columns + '_sum'



## ... then as means

playAvgs = playtrackDF[['PlayKey','d_dir', 'd_o', 'd_s', 'd_twist']].groupby(by=['PlayKey']).mean()/100

playAvgs.columns = playAvgs.columns + '_avg'



## join all features together

playStats = playsDF[['PlayKey', 'FieldType']].drop_duplicates() ## assumed to be consistent for all plays with same gameID prefix in PlayKey

playStats = playStats.merge(playFinalLocs, on='PlayKey')

playStats = playStats.merge(playLens, on='PlayKey')

playStats = playStats.merge(playSums, on='PlayKey')

playStats = playStats.merge(playAvgs, on='PlayKey')



print(datetime.datetime.now() - startDT, 'to generate playStats') ## about 20s on Kaggle



display(playStats.head(2))
## just verifying no weird values resulted from the calculations above

for col in playStats.columns:

    if playStats[col].dtype != 'object':

        print(col, '\t', playStats[col].dtype, '\t', round(playStats[col].min(),3), '\t', round(playStats[col].max(), 3) )
# identifying injury plays

print(playStats.shape)

startDT = datetime.datetime.now()

injPlays = list(injuriesDF['PlayKey'].unique())

playStats['Inj'] = 0

playStats.loc[playStats['PlayKey'].isin(injPlays), 'Inj'] = 1

print(datetime.datetime.now() - startDT, 'to identify Inj plays') ## about 0.05s on Kaggle

print(playStats.shape)



display(playStats[playStats['Inj'] == 1].head())
display(playStats.groupby(['Inj', 'FieldType']).size().round(3))
display(playStats.groupby(['Inj', 'FieldType']).mean().round(3))
display(playStats.groupby(['Inj', 'FieldType']).median().round(3))
startDT = datetime.datetime.now()

plot_fts = playStats.columns[2:-1]

for i_1, feature_1 in enumerate( plot_fts ):

    simple_stats = playStats[['Inj', 'FieldType', feature_1]].copy(deep=True)

    simple_stats['plays'] = 1

    pivot_for_interax = simple_stats.groupby(['Inj', 'FieldType']).mean()

    #features = list(pivot_for_interax.columns)

    axes = list(pivot_for_interax.index.names)

    pivot_for_interax.reset_index(inplace=True)

    for col in axes:

        pivot_for_interax.loc[:, col] = pivot_for_interax[col].astype('str')

    #print(pivot_for_interax)

    ## https://www.statsmodels.org/dev/generated/statsmodels.graphics.factorplots.interaction_plot.html

    interax_plot = interaction_plot(x = pivot_for_interax['FieldType'], trace = pivot_for_interax['Inj'], response = pivot_for_interax[feature_1], plottype = 'both')

    interax_plot.suppressComposite #output plot only once, otherwise two copies are displayed

print(datetime.datetime.now() - startDT, 'to plot interactions between FieldType and Inj') ## about 1.0s on Kaggle
# adding additional columns for player-level, game-level, and play-level info

mlDF = playStats.copy(deep=True)

print(mlDF.shape)

mlDF['PlayerKey'] = mlDF['PlayKey'].astype(str).str[0:5]

mlDF['GameID'] = mlDF['PlayKey'].astype(str).str[0:7]

mlDF['endOutsideHash'] = 1*(mlDF['rel_y_final'] > (18.5/3/2) ) ## 3.0833 yds from midfield to hashmarks



mlDF = mlDF.merge(gamesDF[['GameID', 'Temperature', 'FieldExposed', 'FieldWet']], left_on='GameID', right_on='GameID')

mlDF = mlDF.merge(playersDF[['PlayerKey', 'RosterPosition']], left_on='PlayerKey', right_on='PlayerKey')

mlDF = mlDF.merge(playsDF[['PlayKey', 'Position', 'PositionGroup']], left_on='PlayKey', right_on='PlayKey')



print(mlDF.shape)

mlDF.head()
## this isn't strictly necessary but it helps with keeping the data concise



mlDF.loc[mlDF['RosterPosition'] == 'Quarterback', 'RosterPosition'] = 'QB'

mlDF.loc[mlDF['RosterPosition'] == 'Wide Receiver', 'RosterPosition'] = 'WR'

mlDF.loc[mlDF['RosterPosition'] == 'Linebacker', 'RosterPosition'] = 'LB'

mlDF.loc[mlDF['RosterPosition'] == 'Running Back', 'RosterPosition'] = 'RB'

mlDF.loc[mlDF['RosterPosition'] == 'Defensive Lineman', 'RosterPosition'] = 'DL'

mlDF.loc[mlDF['RosterPosition'] == 'Tight End', 'RosterPosition'] = 'TE'

mlDF.loc[mlDF['RosterPosition'] == 'Safety', 'RosterPosition'] = 'S'

mlDF.loc[mlDF['RosterPosition'] == 'Cornerback', 'RosterPosition'] = 'CB'

mlDF.loc[mlDF['RosterPosition'] == 'Offensive Lineman', 'RosterPosition'] = 'OL'

mlDF.loc[mlDF['RosterPosition'] == 'Kicker', 'RosterPosition'] = 'K'



mlDF['RosterPosition'].unique()
## Because this process needs to be repeated for three very similar variables, the resulting column names need to be prefixed.

## pd.get_dummies() could have been used, but it was using too much memory so this semi-manual process was used instead.



## convert position categorical features into one-hot feature sets

print('Converting RosterPosition...')

for rosterPos in mlDF['RosterPosition'].unique():

    #print(rosterPos)

    mlDF['ros_' + rosterPos] = 0

    mlDF.loc[mlDF['RosterPosition'] == rosterPos, 'ros_' + rosterPos] = 1

mlDF.drop('RosterPosition', axis='columns', inplace=True)



print('Converting (play) Position...')

for playPos in mlDF['Position'].unique():

    #print(playPos)

    mlDF['play_' + playPos] = 0

    mlDF.loc[mlDF['Position'] == playPos, 'play_' + playPos] = 1

mlDF.drop('Position', axis='columns', inplace=True)



print('Converting (play) PositionGroup...')

for playPosGrp in mlDF['PositionGroup'].unique():

    #print(playPosGrp)

    mlDF['playGrp_' + playPosGrp] = 0

    mlDF.loc[mlDF['PositionGroup'] == playPosGrp, 'playGrp_' + playPosGrp] = 1

mlDF.drop('PositionGroup', axis='columns', inplace=True)



mlDF.drop(['PlayerKey', 'GameID'], axis='columns', inplace=True)



mlDF['SynTurf'] = 0

mlDF.loc[mlDF['FieldType'] == 'Synthetic', 'SynTurf'] = 1

mlDF.drop('FieldType', axis='columns', inplace=True)



print('Done!')



## verifying results with a side-scrollable display of the dataframe

HTML(mlDF.head().to_html())
print('Variance in whether plays result in injury:', mlDF['Inj'].var() )
pd.options.display.float_format = '{:.10f}'.format ## need  greater level of detail due to tiny injury occurence proportions



display(mlDF[['Inj', 'SynTurf']].groupby('SynTurf').mean())

display(ttest_ind(mlDF.loc[mlDF['SynTurf']==1,'Inj'], mlDF.loc[mlDF['SynTurf']==0,'Inj']))



## p-value of 0.0436 suggests statistically significant difference in likelihood of injury based on turf type



inj_turf_corr = mlDF['Inj'].corr(mlDF['SynTurf'])

print('The correlation of turf type and injury occurence is only', round(inj_turf_corr, 6), 'which means only', round((inj_turf_corr**2), 10), ' of the variance in injury occurence would be explainable by turf type (possibly through other factors)' )
corrDF = pd.DataFrame(columns = ['Feature', 'Corr_w_Inj'])



feat_for_corr = list(mlDF.columns)[1:-1] #exclude PlayKey and SynTurf

feat_for_corr.remove('Inj')



for i_col, col in enumerate(feat_for_corr):

    if col == 'Inj':

        print(col)

    corrDF.loc[i_col] = [col, mlDF[col].corr(mlDF['Inj'])]



corrDF['R2'] = corrDF['Corr_w_Inj']**2



display(corrDF.sort_values('R2').loc[corrDF['R2'] > (inj_turf_corr**2)])



features_to_test = list(corrDF.sort_values('R2').loc[corrDF['R2'] > (inj_turf_corr**2)]['Feature'])



features_cat = list()

features_con = list()



for ft in features_to_test:

    if(mlDF[ft].max() == 1 and mlDF[ft].min() == 0):

        features_cat.append(ft)

    else:

        features_con.append(ft)
print('Categorical:', features_cat)
startDT = datetime.datetime.now()

catFtsDF = pd.DataFrame(columns=['Feature', 'FalseInj', 'TrueInj', 'p-val'])



for i_cat, cat_ft in enumerate(features_cat):

    inj_ps = mlDF[['Inj', cat_ft]].groupby(cat_ft).mean()['Inj']

    pval = ttest_ind(mlDF.loc[mlDF[cat_ft]==1,'Inj'], mlDF.loc[mlDF[cat_ft]==0,'Inj']).pvalue

    catFtsDF.loc[i_cat] = [cat_ft, inj_ps[0], inj_ps[1], pval]

    

display(catFtsDF)

print(datetime.datetime.now() - startDT, 'to test all categorical features') ## about 0.1s on Kaggle
startDT = datetime.datetime.now()

model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf', data=mlDF).fit()

print(model_InjCouple.pvalues)

print()



for cat_ft in features_cat:

    model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf*' + cat_ft, data=mlDF).fit()

    print(model_InjCouple.pvalues)

    print()

    

for cat_ft1 in features_cat:

    for cat_ft2 in features_cat:

        if cat_ft1 != cat_ft2:

            print(cat_ft1, '*', cat_ft2)

            both1 = mlDF[(mlDF[cat_ft1] == 1) & (mlDF[cat_ft2] == 1)].shape[0]

            both0 = mlDF[(mlDF[cat_ft1] == 0) & (mlDF[cat_ft2] == 0)].shape[0]

            first1 = mlDF[(mlDF[cat_ft1] == 1) & (mlDF[cat_ft2] == 0)].shape[0]

            second1 = mlDF[(mlDF[cat_ft1] == 0) & (mlDF[cat_ft2] == 1)].shape[0]

            if(both1 == 0 or both0 == 0 or first1 == 0 or second1 == 0):

                print("One feature is a subset of the other, no interaction effect necessary")

            else:

                model_InjCouple = sm.Logit.from_formula('Inj ~' + cat_ft1 + '*' + cat_ft2, data=mlDF).fit()

                print(model_InjCouple.pvalues)

            print()

print(datetime.datetime.now() - startDT, 'to test interactions between categorical features') ## about 15s on Kaggle
startDT = datetime.datetime.now()

plot_fts = ['SynTurf'] + features_cat

for i_1, feature_1 in enumerate( plot_fts ):

    for i_2, feature_2 in enumerate( plot_fts ):

        if i_1 < i_2:

            simple_mlDF = mlDF[['Inj', feature_1, feature_2]].copy(deep=True)

            simple_mlDF['plays'] = 1



            pivot_for_interax = simple_mlDF.groupby([feature_1, feature_2]).sum()



            for col in pivot_for_interax.columns:

                pivot_for_interax.loc[:, col] = pivot_for_interax[col] / pivot_for_interax['plays']



            pivot_for_interax.drop(labels='plays', axis='columns', inplace=True)

            #features = list(pivot_for_interax.columns)

            axes = list(pivot_for_interax.index.names)

            pivot_for_interax.reset_index(inplace=True)

            for col in axes:

                pivot_for_interax.loc[:, col] = pivot_for_interax[col].astype('str')



            #print(pivot_for_interax)



            ## https://www.statsmodels.org/dev/generated/statsmodels.graphics.factorplots.interaction_plot.html



            interax_plot = interaction_plot(x = pivot_for_interax[feature_1], trace = pivot_for_interax[feature_2], response = pivot_for_interax['Inj'], plottype = 'both')

            interax_plot.suppressComposite #output plot only once, otherwise two copies are displayed

print(datetime.datetime.now() - startDT, 'to plot interactions between categorical features') ## about 1.0s on Kaggle
print('Continuous:', features_con)
startDT = datetime.datetime.now()

for con_ft in features_con:

    model_InjCouple = sm.Logit.from_formula('Inj ~ ' + con_ft, data=mlDF).fit()

    print(model_InjCouple.pvalues)

    print()

print(datetime.datetime.now() - startDT, 'to test each continuous feature') ## about 10s on Kaggle
startDT = datetime.datetime.now()

model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf', data=mlDF).fit()

print(model_InjCouple.pvalues)

print()



for con_ft in features_con:

    model_InjCouple = sm.Logit.from_formula('Inj ~ SynTurf*' + con_ft, data=mlDF).fit()

    print(model_InjCouple.pvalues)

    print()

print(datetime.datetime.now() - startDT, 'to test interaction effects between turf type and continuous features') ## about 10s on Kaggle
print(mlDF['SynTurf'].corr(mlDF['d_o_avg']) )

box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'd_o_avg'], mlDF.loc[mlDF['SynTurf'] == 1, 'd_o_avg']]

display(plot.boxplot(box_data, vert=False, showfliers = False))
print(mlDF['SynTurf'].corr(mlDF['d_o_sum']) )

box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'd_o_sum'], mlDF.loc[mlDF['SynTurf'] == 1, 'd_o_sum']]

display(plot.boxplot(box_data, vert=False, showfliers = False))
print(mlDF['SynTurf'].corr(mlDF['obs']) )

box_data = [mlDF.loc[mlDF['SynTurf'] == 0, 'obs'], mlDF.loc[mlDF['SynTurf'] == 1, 'obs'] ]

display(plot.boxplot(box_data, vert=False, showfliers = False))
startDT = datetime.datetime.now()

for i_ft1, con_ft1 in enumerate(features_con):

    for i_ft2, con_ft2 in enumerate(features_con):

        if i_ft1 < i_ft2:

            print(con_ft1, '*', con_ft2)

            model_InjCouple = sm.Logit.from_formula('Inj ~' + con_ft1 + '*' + con_ft2, data=mlDF).fit()

            print(model_InjCouple.pvalues)

            print()

print(datetime.datetime.now() - startDT, 'to test interactions within continuous features') ## about 45s on Kaggle
mlDF['d_s_sum'].corr(mlDF['rel_y_final'])
startDT = datetime.datetime.now()

model_fts = ['SynTurf', 'd_o_avg', 'd_o_sum', 'rel_y_final', 'd_s_sum', 'd_s_avg']

X_train, X_test, y_train, y_test = train_test_split(mlDF[model_fts], mlDF['Inj'], test_size=0.33, random_state=42)

print(sum(y_train == 1), 'injuried in training set,', sum(y_test == 1), 'injuried in test set',)

logreg = LogisticRegression(penalty='none', solver='lbfgs')

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

print('AUC:', logit_roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plot.figure()

plot.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plot.plot([0, 1], [0, 1],'r--')

plot.xlim([0.0, 1.0])

plot.ylim([0.0, 1.05])

plot.xlabel('False Positive Rate')

plot.ylabel('True Positive Rate')

plot.title('Receiver operating characteristic')

plot.legend(loc="lower right")

plot.savefig('Log_ROC')

plot.show()

print(datetime.datetime.now() - startDT, 'to fit, test, and evaluate logistic regression model') ## about 10s on Kaggle
print('Entire workbook runs in ', datetime.datetime.now() - overall_startDT)