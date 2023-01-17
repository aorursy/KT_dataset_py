%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.tabular import *

from sklearn.metrics import log_loss
path = Path('data/MarchMadness')

dest = path

dest.mkdir(parents=True, exist_ok=True)

input_path = '../input/mens-machine-learning-competition-2019'

data_path = '../input/ncaa-19-dataprep/data/MarchMadness'
!ls ../input/
!cp -r ../input/ncaa-19-tabular/data/MarchMadness/* {path}/

# !cp -r ../input/ncaa-19-dataprep/data/MarchMadness/* {path}/
!ls {path}
df_test = pd.read_csv(f'{data_path}/df_test.csv', low_memory=False)

df_msr = pd.read_csv(f'{data_path}/df_msr.csv', low_memory=False)

df = pd.read_csv(f'{data_path}/df.csv', low_memory=False)

sub = pd.read_csv(f'{input_path}/SampleSubmissionStage2.csv', low_memory=False)

seeds = pd.read_csv(f'{input_path}/datafiles/NCAATourneySeeds.csv', low_memory=False)

val_idxs = np.load(f'{data_path}/val_idxs.npy')
display(df.tail(), df_test.tail())
# val_idxs = np.arange(df.shape[0]-int(df.shape[0]*0.2), df.shape[0])

# len(val_idxs)
def random_seed(seed_value, use_cuda):

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

#Remember to use num_workers=0 when creating the DataBunch.
def join_df(left, right, left_on, right_on=None, on=None, how='left', suffix='_y'):

    if right_on is None: right_on = left_on

    return left.merge(right, left_on=left_on, right_on=right_on,

                      on=on, how=how, suffixes=("", suffix))
# for c in df.columns: print(c)
base_cols = ['Score'] # , , 'FGM', 'FGM3', 'FTM'

#     , 'OffRtg', 'DefRtg', 'NetRtg', 'AstR', 'TSP', 'FTAR', 

#              'TOR', 'ORP', 'DRP', 'RP', 'PIE', 'eFGP'



drop_cols = ['Loc', 'PointDiff_1', 'RankDiff_1', 'Seed_1', 'Seed_2'] 

# 'Season', 'TeamId_1', 'TeamId_2', 'result',  'Coach_1', 'Coach_2', 

# , 'PointDiff_1', , 'Rank_1', 'Rank_2'



for c in base_cols: 

    drop_cols.append(c+'_1')

    drop_cols.append(c+'_Opp_1')

    drop_cols.append(c+'_2')

    drop_cols.append(c+'_Opp_2')



df.drop(drop_cols, axis=1, inplace=True)

df_test.drop(drop_cols, axis=1, inplace=True)

# df = df.loc[:,keep_cols]

# df_test = df_test.loc[:,keep_cols]
dep_var = 'result'

cat_vars = ['Season', 'TeamId_1', 'TeamId_2', 'Coach_1', 'Coach_2',

            'Top5_1', 'Top5_2', 'Top25_1', 'Top25_2', 'Top50_1', 'Top50_2',

            'ConfAbbrev_1', 'ConfAbbrev_2', 'Is_ConfGm', 'isMajor_1', 'isMajor_2'] 

        # 'Loc', 'Rank_1', 'Rank_2', , 'Seed_1', 'Seed_2', 

cont_vars = [c for c in df.columns if c not in cat_vars]

cont_vars.remove('result')



test = TabularList.from_df(df_test.copy(), path=path, cat_names=cat_vars, cont_names=cont_vars)



procs=[FillMissing, Categorify, Normalize]
random_seed(42, True)



src = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, 

                           procs=procs)

                   .split_none())

                  #.split_by_idx(val_idxs))



data = (src.label_from_df(cols=dep_var)

                  .add_test(test)

                  .databunch(bs=512, num_workers=0))
#np.random.seed(2)

learn = tabular_learner(data, layers=[200,100], emb_drop=0.2,

                        metrics=[accuracy]) # , ps=[0.001] , emb_szs=emb_szs

                        #  



learn.model
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, wd=0.05) #  
learn.save('m_stage2_1')

#learn.load('m_stage1_26')
# learn.recorder.plot_losses() # last=-1
preds, _ = learn.get_preds(DatasetType.Test) # DatasetType.Test



eps = 1e-5

#df_valid = df.iloc[val_idxs].copy()

df_test['Pred'] = np.clip(preds[:,1], eps, 1-eps)



df_test = df_test[['Season', 'TeamId_1', 'TeamId_2', 'Pred']]

df_msr = df_msr[['Season', 'TeamId_1', 'TeamId_2', 'result']]

df_msr.reset_index(inplace=True, drop=True)



df_m = join_df(df_msr, df_test, ['Season', 'TeamId_1', 'TeamId_2'])

df_m = df_m[df_m.Pred.notnull()]; df_m.head()



# measure logloss

y_true = df_m.result # df_test.result

y_pred = df_m.Pred # df_test.Pred



log_loss(y_true, y_pred, eps=eps)
df_test.head()
# sub['Pred'] = df_test.Pred

# sub.head()
# sub.to_csv('sub_41.csv', index=False)
# interp = ClassificationInterpretation.from_learner(learn)

# interp_trn = ClassificationInterpretation.from_learner(learn, ds_type=DatasetType.Train)



# interp.plot_confusion_matrix()



# losses, idxs = interp.top_losses()



# interp_trn.plot_confusion_matrix()



# losses_trn, idxs_trn = interp_trn.top_losses()



# keep_idxs = torch.cat((idxs[1800:], idxs_trn[100:]))
#data.valid_ds.x[idxs]
# seeds = seeds[seeds.Season>=2014]

# df_test_seeds = join_df(df_test, seeds, ['Season', 'TeamId_1'], ['Season', 'TeamID'])

# df_test_seeds = join_df(df_test_seeds, seeds, ['Season', 'TeamId_2'], ['Season', 'TeamID'],

#                         suffix='_2')

# df_test_seeds.drop(['TeamID', 'TeamID_2'], axis=1, inplace=True)



# def champ1(row):

#     if ('W' in row.Seed or 'X' in row.Seed) and ('Y' in row.Seed_2 or 'Z' in row.Seed_2):

#         return 0.975

#     if ('Y' in row.Seed or 'Z' in row.Seed) and ('W' in row.Seed_2 or 'X' in row.Seed_2):

#         return 0.975

#     else:

#         return row.Pred

    

# def champ2(row):

#     if ('W' in row.Seed or 'X' in row.Seed) and ('Y' in row.Seed_2 or 'Z' in row.Seed_2):

#         return 0.025

#     if ('Y' in row.Seed or 'Z' in row.Seed) and ('W' in row.Seed_2 or 'X' in row.Seed_2):

#         return 0.025

#     else:

#         return row.Pred



# df_test_seeds['Pred'] = df_test_seeds.apply(champ1, axis=1)

# df_test = df_test_seeds.copy()