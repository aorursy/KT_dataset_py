%reload_ext autoreload

%autoreload 2



from fastai import *

from fastai.tabular import *

from sklearn.metrics import log_loss
path = Path('data/MarchMadness')

dest = path

dest.mkdir(parents=True, exist_ok=True)

# input_path = '../input/mens-machine-learning-competition-2019'

# massey_path = (f'{input_path}/masseyordinals_thru_2019_day_128')

massey_path = Path('../input/masseyordinals_thru_2019_day_128')
# !ls {massey_path}
# !cp -r ../input/ncaa-feature-engineering/data/MarchMadness/* {path}/
!ls {path}
# results = pd.read_csv(f'{path}/results_adv_stats.csv', low_memory=False)

# tourneys = pd.read_csv(f'{path}/tourneys_adv_stats.csv', low_memory=False)

# results = pd.read_csv(f'{input_path}/stage2datafiles/RegularSeasonDetailedResults.csv', 

#                       low_memory=False)

# tourneys = pd.read_csv(f'{input_path}/stage2datafiles/NCAATourneyDetailedResults.csv', 

#                        low_memory=False)

results = pd.read_csv('../input/stage2datafiles/RegularSeasonDetailedResults.csv', 

                      low_memory=False)

tourneys = pd.read_csv('../input/stage2datafiles/NCAATourneyDetailedResults.csv', 

                       low_memory=False)
!ls '../input/'
table_names = ['TeamCoaches', 'GameCities', 'Cities', 'Teams', 'Seasons',

               'NCAATourneySeedRoundSlots', 'NCAATourneySeeds', 'NCAATourneySlots',

               'ConferenceTourneyGames', 'Conferences', 'TeamConferences']

              

              #

# tables = [pd.read_csv(f'{input_path}/stage2datafiles/{fname}.csv', 

#                       low_memory=False) for fname in table_names]

tables = [pd.read_csv(f'../input/stage2datafiles/{fname}.csv', 

                      low_memory=False) for fname in table_names]

coaches, game_cities, cities, teams, seasons, seed_slots, seeds, slots = tables[:8]

conf_games, conferences, team_conferences = tables[8:]

   # , game_cities, results,  tourneys, 



# sub = pd.read_csv(f'{input_path}/SampleSubmissionStage2.csv', low_memory=False)

massey = pd.read_csv(f'{massey_path}/MasseyOrdinals_thru_2019_day_128.csv', low_memory=False)

sub = pd.read_csv('../input/SampleSubmissionStage2.csv', low_memory=False)

massey = massey[(massey.RankingDayNum<=128) & (massey.SystemName=='POM')]; # 132 (daynum)

coaches = coaches[coaches.LastDayNum==154]

def join_df(left, right, left_on, right_on=None, on=None, how='left', suffix='_y'):

    if right_on is None: right_on = left_on

    return left.merge(right, left_on=left_on, right_on=right_on,

                      on=on, how=how, suffixes=("", suffix))
massey_grp = massey.groupby(['Season', 'TeamID'], as_index=False).OrdinalRank.median()



def add_massey(df):

    df = join_df(df, massey_grp, ['Season', 'WTeamID'], ['Season', 'TeamID'])

    df = df.rename(columns={'OrdinalRank': 'WRank'}).drop(['TeamID'], axis=1)



    df = join_df(df, massey_grp, ['Season', 'LTeamID'], ['Season', 'TeamID'])

    df = df.rename(columns={'OrdinalRank': 'LRank'}).drop(['TeamID'], axis=1)

    

    return df



results = add_massey(results)

tourneys = add_massey(tourneys)
include = ['Season', 'DayNum', 'WTeamID', 'LTeamID', 'WLoc', 'WRank', 'LRank'] 

    # 'CRType', 'CityID', 'WCoachName', 'LCoachName', 'WSeed', 'LSeed'



base_cols = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 

                 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF'] 

    # , 'OffRtg', 'DefRtg', 'NetRtg', 

#                  'AstR', 'TOR', 'TSP', 'eFGP', 'FTAR', 'ORP', 'DRP', 'RP', 'PIE'

            

def prep_data(df):

    

    win_cols = ['W'+c for c in base_cols]

    loss_cols = ['L'+c for c in base_cols] 

        

    df_base = df[include + win_cols + loss_cols].copy()

    df_swap = df[include + loss_cols + win_cols].copy()



    df_base['result'] = 1

    df_swap['result'] = 0



    f = lambda x: 'A' if x is 'H' else 'H' if x is 'A' else x

    df_swap['WLoc'] = df_swap.WLoc.map(f) # swap Loc in df_swap 



    # rename columns in df_swap and df

    swap_cols = {'WTeamID': 'TeamId_2', 'LTeamID': 'TeamId_1', 

                 'WLoc': 'Loc', 'WRank': 'Rank_2', 'LRank': 'Rank_1'}

        #, 'WCoachName': 'Coach_2', 'LCoachName': 'Coach_1' 

        #, 'WSeed': 'Seed_2', 'LSeed': 'Seed_1'

    df_cols = {'WTeamID': 'TeamId_1', 'LTeamID': 'TeamId_2', 

               'WLoc': 'Loc', 'WRank': 'Rank_1', 'LRank': 'Rank_2'}

        # , 'WCoachName': 'Coach_1', 'LCoachName': 'Coach_2' 

        # , 'WSeed': 'Seed_1', 'LSeed': 'Seed_2'

    

    for c,d in zip(win_cols, loss_cols):

        swap_cols[d] = d[1:]+'_1'

        swap_cols[c] = c[1:]+'_2'

        df_cols[c] = c[1:]+'_1'

        df_cols[d] = d[1:]+'_2'

    

    df_swap.rename(columns=swap_cols, inplace=True)

    df_base.rename(columns=df_cols, inplace=True)



    out = pd.concat([df_base, df_swap], sort=False).sort_index().reset_index(drop=True)

        

    out['PointDiff_1'] = out.Score_1 - out.Score_2

    out['RankDiff_1'] = out.Rank_1 - out.Rank_2

    return out



season_results = prep_data(results)

tourney_results = prep_data(tourneys)
k_cols = ['PointDiff_1', 'RankDiff_1']  #

box_cols = []

for c in base_cols:

    box_cols.append(c+'_1')

    box_cols.append(c+'_2')
regResults_grp = season_results.groupby(['Season', 'TeamId_1'], 

                                        as_index=False)[k_cols+box_cols]
def get_joined(df1,df2,df3):

    out = join_df(df1,df2,['Season', 'TeamId_1'])

    return join_df(out,df3,['Season', 'TeamId_2'])





season_stats = regResults_grp.median()

#season_stats['TeamId_2'] = season_stats.TeamId_2.astype('int64')





#season_stats.head()



season_stats_t1 = season_stats.copy()

season_stats_t2 = season_stats.copy()



season_stats_t1.columns = [c.replace('_1','').replace('_2', '_Opp')+'_1' 

                           for c in list(season_stats_t1)]

season_stats_t2.columns = [c.replace('_1', '').replace('_2', '_Opp')+'_2' 

                           for c in list(season_stats_t2)]



season_stats_t1.rename(columns={'Season_1': 'Season'}, inplace=True)

season_stats_t2.rename(columns={'Season_2': 'Season'}, inplace=True)
'''

exerimenting with rolling median



box_cols = ['PointDiff_1']

#box_cols = []

for c in base_cols:

    box_cols.append(c+'_1')

    box_cols.append(c+'_2')

box_cols



regResultsExp =season_results.groupby(['Season', 'TeamId_1'])[box_cols].expanding(5)

tourneyResultsExp = tourney_results.groupby(['Season', 'TeamId_1'])[box_cols].expanding(5)



season_resultsExp = season_results.copy()

tourney_resultsExp = tourney_results.copy()



season_resultsExp = regResultsExp.median().bfill().reset_index([0,1])

tourney_resultsExp = tourneyResultsExp.median().bfill().reset_index([0,1])



season_results.loc[season_resultsExp.index, season_resultsExp.columns[2:]] =season_resultsExp.loc[:,season_resultsExp.columns[2:]]

tourney_results.loc[tourney_resultsExp.index, tourney_resultsExp.columns[2:]] =tourney_resultsExp.loc[:,tourney_resultsExp.columns[2:]]



season_data = season_results.copy()

tourney_data = tourney_results.copy()



'''
keep_cols = ['Season', 'DayNum', 'TeamId_1', 'TeamId_2', 'Loc', 'Rank_1', 'Rank_2',

             'result'] # 'Score_1', 'Score_2',  

                #'Loc', 'CRType', 'CityID', 'Coach_1', 'Coach_2', , 'Seed_1', 'Seed_2',  



season_results = season_results.loc[:,keep_cols]

tourney_results = tourney_results.loc[:,keep_cols]

                              

#display(season_results.head(), tourney_results.head())



season_data = get_joined(season_results, season_stats_t1, season_stats_t2)

tourney_data = get_joined(tourney_results, season_stats_t1, season_stats_t2)

display(season_data.head(), tourney_data.head())
def add_coach(df):

    df = join_df(df, coaches, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

    df = join_df(df, coaches, ['Season', 'TeamId_2'], ['Season', 'TeamID'])

    df.drop(['FirstDayNum', 'FirstDayNum_y', 'LastDayNum', 'LastDayNum_y', 'TeamID', 

                  'TeamID_y'], axis=1, inplace=True)

    df.rename(columns={'CoachName': 'Coach_1', 'CoachName_y': 'Coach_2'}, inplace=True)

    return df



def add_seed(df):

    df = join_df(df, seeds, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

    df = join_df(df, seeds, ['Season', 'TeamId_2'], ['Season', 'TeamID'])

    df.drop(['TeamID', 'TeamID_y'], axis=1, inplace=True)

    df.rename(columns={'Seed': 'Seed_1', 'Seed_y': 'Seed_2'}, inplace=True)

    

    return df



season_data = add_coach(season_data)

season_data = add_seed(season_data)



season_data.Seed_1.fillna('UnR', inplace=True)

season_data.Seed_2.fillna('UnR', inplace=True)



tourney_data = add_coach(tourney_data)

tourney_data = add_seed(tourney_data)
def add_conference(df):

    df = join_df(df, team_conferences, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

    df = join_df(df, team_conferences, ['Season', 'TeamId_2'], ['Season', 'TeamID'])

    df.drop(['TeamID', 'TeamID_y'], axis=1, inplace=True)

    df.rename(columns={'ConfAbbrev': 'ConfAbbrev_1', 'ConfAbbrev_y': 'ConfAbbrev_2'}, inplace=True)

    

    return df



season_data = add_conference(season_data)

tourney_data = add_conference(tourney_data)
def is_confGm(df): return True if (df.ConfAbbrev_1==df.ConfAbbrev_2) else False



season_data['Is_ConfGm'] = season_data.apply(is_confGm, axis=1)

tourney_data['Is_ConfGm'] = tourney_data.apply(is_confGm, axis=1)
df_msr = tourney_data[tourney_data.Season>=2014].copy()

tourney_data = tourney_data[tourney_data.Season<2014]

df = pd.concat([season_data,tourney_data], sort=False)
major_conf =  ['acc', 'big_ten', 'big_twelve', 'pac_twelve', 'sec', 

               'big_east', 'aac', 'a_ten', 'mwc', 'pac_ten']





mid_major = ['a_sun', 'aac', 'aec', 'asc', 'awc', 'big_eight', 'big_sky', 'big_south',

 'big_west', 'caa', 'cusa', 'ecacm', 'ecacn', 'ecacs', 'ecc', 'gmwc', 'gsc', 'gwc', 'horizon',

 'ind', 'ivy', 'maac', 'mac', 'meac', 'metro', 'mid_cont', 'mvc', 'mw_city', 'mw_coll',

 'nac', 'nec', 'ovc', 'patriot', 'pcaa', 'southern', 'southland', 'summit', 'sun_belt',

 'swac', 'swc', 'wac', 'wcc']



df['isMajor_1'] = df.apply(lambda x: True if x.ConfAbbrev_1 in major_conf else False, axis=1)

df['isMajor_2'] = df.apply(lambda x: True if x.ConfAbbrev_2 in major_conf else False, axis=1)
df_test = sub.copy()



df_test['Season'] = df_test.ID.map(lambda x: x.split('_')[0]).astype('int64')

#df_test['DayNum'] = 130

df_test['TeamId_1'] = df_test.ID.map(lambda x: x.split('_')[1]).astype('int64')

df_test['TeamId_2'] = df_test.ID.map(lambda x: x.split('_')[2]).astype('int64')

#df_test.drop(['ID', 'Pred'], inplace=True, axis=1)

#df_test['Loc'] = 'N'





df_test.drop(['ID', 'Pred'], axis=1, inplace=True)        

df_test.shape, sub.shape
# def tourney_daynum(df):

#     strong_seeds = ['01', '02', '03', '04', '05', '06', '07', '08']

#     weak_seeds = ['16', '15', '14', '13', '12', '11', '10', '09']

#     R1 = [c for c in zip(strong_seeds, weak_seeds)]

#     R2 = [R1[i]+R1[-i-1] for i in range(len(R1)//2)]

#     R3 = [(R2[i]+R2[-i-1]) for i in range(len(R2)//2)]

#     R4 = (strong_seeds+weak_seeds)

#     print(R1, R2, R3, R4)



#     if ('W' in df.Seed or 'X' in df.Seed) and ('Y' in df.Seed_2 or 'Z' in df.Seed_2):

#         return int(154) # 'R6CH'

#     if ('W' in df.Seed and 'X' in df.Seed_2) or ('Y' in df.Seed and 'Z' in df.Seed_2): 

#         return int(152) # 'R5WX'

#     if ('X' in df.Seed and 'W' in df.Seed_2) or ('Z' in df.Seed and 'Y' in df.Seed_2):

#         return int(152)

#     if df.Seed[0][0]==df.Seed_2[0][0]:

#         if (df.Seed.str[1:3], df.Seed_2.str[1:3]) in R1: return int(136)

#         if# 'R1W1'

# #         if ('01' in df.Seed or '16' in df.Seed) and ('08' in df.Seed_2 or '09' in df.Seed_2):

# #             return int(138)



# df_test['DayNum'] = df_test.apply(tourney_daynum, axis=1)
# slots_df = join_df(seeds, seed_slots, 'Seed')

# slots_df = join_df(slots_df, slots[slots.Season>=2014], ['Season', 'GameSlot'], ['Season', 'Slot'])

# slots_df
seeds = seeds[seeds.Season==2019]

# df_test = join_df(df_test, seeds, ['Season', 'TeamId_1'], ['Season', 'TeamID'])



# df_test = join_df(df_test, seeds, ['Season', 'TeamId_2'], ['Season', 'TeamID'],

#                         suffix='_2')



df_test = add_seed(df_test)
df_test = add_coach(df_test)
# df_test.rename(columns={'Seed': 'Seed_1'}, inplace=True)

# df_test.drop(['TeamID', 'TeamID_2'], axis=1, inplace=True)



# df_test = join_df(df_test, coaches, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

# df_test.drop(['FirstDayNum', 'LastDayNum'], axis=1, inplace=True)

# df_test = join_df(df_test, coaches, ['Season', 'TeamId_2'], ['Season', 'TeamID'])

# df_test.drop(['FirstDayNum', 'LastDayNum', 'TeamID', 'TeamID_y'], axis=1, inplace=True)

# df_test.rename(columns={'CoachName': 'Coach_1', 'CoachName_y': 'Coach_2'}, inplace=True)

# df_test.shape, sub.shape

df_test = get_joined(df_test, season_stats_t1, season_stats_t2)



#df.sort_values(by=['Season', 'DayNum'], inplace=True); 



df_test = join_df(df_test, massey_grp, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

df_test = df_test.rename(columns={'OrdinalRank': 'Rank_1'}).drop(['TeamID'], axis=1)



df_test = join_df(df_test, massey_grp, ['Season', 'TeamId_2'], ['Season', 'TeamID'])

df_test = df_test.rename(columns={'OrdinalRank': 'Rank_2'}).drop(['TeamID'], axis=1)
df_test = add_conference(df_test)

df_test['Is_ConfGm'] = df_test.apply(is_confGm, axis=1)
df_test['isMajor_1'] = df_test.apply(lambda x: True if x.ConfAbbrev_1 in major_conf else False, 

                                    axis=1)

df_test['isMajor_2'] = df_test.apply(lambda x: True if x.ConfAbbrev_2 in major_conf else False, 

                                    axis=1)
# [c for c in df.columns if c not in df_test.columns]
val_idxs = df.index[df.DayNum>118];

#val_idxs = df.index[df.Season>=2016]

len(val_idxs), len(val_idxs) / df.shape[0]

#df.columns
#df.drop(['Score_1', 'Score_2'], axis=1, inplace=True) # 'TeamId_Opp_1', 'TeamId_Opp_2'

#df.rename(columns={'Score_1_y': 'Score_1', 'Score_2_y': 'Score_2'})

#df.shape, df.head()



#df_test['DayNum'] = 134

df_test['Loc'] = 'N'

df_test['result'] = 0
#drop_flds = [c for c in df.columns if 'Score' in c]; 

df.drop(['DayNum'], axis=1, inplace=True) # , 'Loc'

#df.drop(drop_flds, axis=1, inplace=True)
#drop_flds = [c for c in df_test.columns if 'Score' in c] + ['TeamId_Opp_1', 'TeamId_Opp_2']



#df_test.drop(drop_flds, axis=1, inplace=True)
df_test.shape, df.shape, len(val_idxs)
#[c for c in df.columns if c not in df_test.columns]
def high_ranks(df):

    df['Top5_1'] = df.apply(lambda row: True if (row.Rank_1<=5) else False, axis=1)

    df['Top5_2'] = df.apply(lambda row: True if (row.Rank_2<=5) else False, axis=1)

    df['Top25_1'] = df.apply(lambda row: True if (row.Rank_1<=25) else False, axis=1)

    df['Top25_2'] = df.apply(lambda row: True if (row.Rank_2<=25) else False, axis=1)

    df['Top50_1'] = df.apply(lambda row: True if (row.Rank_1<=50) else False, axis=1)

    df['Top50_2'] = df.apply(lambda row: True if (row.Rank_2<=50) else False, axis=1)



for f in (df, df_test): high_ranks(f)



display(df.head(), df_test.head())
#data.save('data_save.csv')

df.to_csv(f'{path}/df.csv', index=False)

df_test.to_csv(f'{path}/df_test.csv', index=False)

df_msr.to_csv(f'{path}/df_msr.csv', index=False)

np.save(f'{path}/val_idxs.npy', val_idxs)
# '''for testing'''



# dep_var = 'result'

# cat_vars = ['Season', 'TeamId_1', 'TeamId_2', 'Loc', 'Rank_1', 'Rank_2', 

#             'Seed_1', 'Seed_2', 'Coach_1', 'Coach_2', 'Top5_1', 'Top5_2',

#             'Top25_1', 'Top25_2', 'Top50_1', 'Top50_2',

#             'ConfAbbrev_1', 'ConfAbbrev_2', 'Is_ConfGm', 'isMajor_1', 'isMajor_2'] # , , 

# cont_vars = [c for c in df.columns if c not in cat_vars]

# cont_vars.remove('result')



# test = TabularList.from_df(df_test.copy(), path=path, cat_names=cat_vars, cont_names=cont_vars)



# procs=[FillMissing, Categorify, Normalize]



# data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

# #                    .split_by_idx(val_idxs)

#                    .split_none()

#                    .label_from_df(cols=dep_var)

#                    .add_test(test)

#                    .databunch(bs=128))



# learn = tabular_learner(data, layers=[1000, 500], ps=[0.001,0.01], emb_drop=0.04,

#                         metrics=[accuracy]) # 



# learn.lr_find()

# learn.recorder.plot()
#  learn.fit_one_cycle(1)
# preds, _ = learn.get_preds(DatasetType.Test) # DatasetType.Test