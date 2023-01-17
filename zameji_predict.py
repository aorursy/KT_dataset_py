# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import log_loss

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, GaussianNoise

from keras import regularizers

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

from math import ceil



import gc

gc.enable()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/womens-machine-learning-competition-2019/stage2wdatafiles"))

print(os.listdir("../input/season-stats"))

# Any results you write to the current directory are saved as output.
recent = pd.read_csv("../input/season-stats/RecentStatsSince1998.csv", index_col=0)

seeds = pd.read_csv('../input/womens-machine-learning-competition-2019/stage2wdatafiles/WNCAATourneySeeds.csv')

tourney_dresults = pd.read_csv('../input/womens-machine-learning-competition-2019/stage2wdatafiles/WNCAATourneyCompactResults.csv')



sub = pd.read_csv('../input/womens-machine-learning-competition-2019/WSampleSubmissionStage2.csv')
recent.describe()
seeds["Seed"] = seeds["Seed"].replace("\D", "", regex=True).astype("int8")
recent = recent.loc[recent["Games"] != 0]



recent.loc[recent["Season"] > 2009,"Score"] = 2*recent.loc[recent["Season"] > 2009,"FGM"] + recent.loc[recent["Season"] > 2009,"FGM3"] + recent.loc[recent["Season"] > 2009,"FTM"]

recent.loc[recent["Season"] > 2009,"Score_A"] = 2*recent.loc[recent["Season"] > 2009,"FGM_A"] + recent.loc[recent["Season"] > 2009,"FGM3_A"] + recent.loc[recent["Season"] > 2009,"FTM_A"]    

    

recent["ScoreDf"] = recent["Score"] - recent["Score_A"]

recent["TODf"] = recent["TO"] - recent["Stl"]

recent[["Wins", "Score", "Score_A", "ScoreDf", "FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF", "FGM_A", "FGA_A", "FGM3_A", "FTM_A", "FTA_A", "OR_A", "DR_A", "Ast_A", "TO_A", "Stl_A", "Blk_A", "PF_A"]] = recent[["Wins", "Score", "Score_A", "ScoreDf", "FGM", "FGA", "FGM3", "FTM", "FTA", "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF", "FGM_A", "FGA_A", "FGM3_A", "FTM_A", "FTA_A", "OR_A", "DR_A", "Ast_A", "TO_A", "Stl_A", "Blk_A", "PF_A"]].values/np.reshape(recent["Games"].values, [-1,1])

recent.columns = ["Recent"+x if (x not in ["TeamID", "Season"]) else x for x in recent.columns ]



stsc = MinMaxScaler()

recent[[x for x in recent.columns if x not in ["TeamID", "Season"]]] = stsc.fit_transform(recent[[x for x in recent.columns if x not in ["TeamID", "Season"]]])



nulls = [x for x in recent.columns if x not in ["TeamID", "Season", "RecentGames", "RecentScore"]]

for x in nulls:

    recent.loc[(recent["Season"].isin(range(1998,2010)) & recent["RecentGames"]>0) & np.isnan(recent[x]), x] = 0.5



dt = seeds.merge(recent, on=["TeamID", "Season"])



dt.head()
tourney_dresults["ID"] = tourney_dresults.apply(lambda r: '_'.join(map(str, [r['Season'], min(r['WTeamID'], r["LTeamID"]), max(r["WTeamID"],r["LTeamID"])])), axis=1)

tourney_dresults["mTeam"] = tourney_dresults.apply(lambda r: min(r['WTeamID'], r["LTeamID"]), axis=1)

tourney_dresults["Pred"] = 0



tourney_dresults.loc[tourney_dresults["mTeam"]==tourney_dresults["WTeamID"],"Pred"] = 1

tourney_dresults.drop(["mTeam"],1, inplace=True)



tourney_dresults['WLoc'] = 3

tourney_dresults['Season'] = tourney_dresults['ID'].map(lambda x: x.split('_')[0])

tourney_dresults['Season'] = tourney_dresults['Season'].astype(int)

tourney_dresults['Team1'] = tourney_dresults['ID'].map(lambda x: x.split('_')[1])

tourney_dresults['Team2'] = tourney_dresults['ID'].map(lambda x: x.split('_')[2])



tourney_dresults['IDTeams'] = tourney_dresults.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)

tourney_dresults['IDTeam1'] = tourney_dresults.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

tourney_dresults['IDTeam2'] = tourney_dresults.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)



tourney_dresults = tourney_dresults[["ID", "Pred", "Season", "Team1", "Team2"]]



tourney_dresults[["Team1", "Team2", "Season"]] = tourney_dresults[["Team1", "Team2", "Season"]].astype("int16")



tourney_dresults = tourney_dresults.merge(dt, left_on=["Team1", "Season"], right_on=["TeamID", "Season"])

tourney_dresults = tourney_dresults.merge(dt, left_on=["Team2", "Season"], right_on=["TeamID", "Season"])

cols = ["Seed", "RecentTODf", 'RecentGames', 'RecentWins', "RecentScore", "RecentScore_A", "RecentScoreDf", 'RecentFGM', 'RecentFGA', 'RecentFGM3',

                          'RecentFTM', 'RecentFTA', 'RecentOR', 'RecentDR', 'RecentAst',

                          'RecentTO', 'RecentStl', 'RecentBlk', 'RecentPF',

                         "RecentFGM_A", "RecentFGA_A", "RecentFGM3_A", "RecentFTM_A",

                          "RecentFTA_A", "RecentOR_A", "RecentDR_A",

                          "RecentAst_A", "RecentTO_A", "RecentStl_A", "RecentBlk_A", "RecentPF_A"]



diff = ["Dif"+x for x in cols]



difs = pd.DataFrame((tourney_dresults[[x+"_x" for x in cols]].values+0.1)/(0.1+

                    tourney_dresults[[x+"_y" for x in cols]].values), columns=diff)



tourney_dresults = pd.concat([tourney_dresults, difs], 1)
sub = pd.read_csv('../input/womens-machine-learning-competition-2019/WSampleSubmissionStage2.csv')



sub['WLoc'] = 3

sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])

sub['Season'] = sub['Season'].astype(int)

sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])

sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])



sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)

sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)



sub = sub[["ID", "Pred", "Season", "Team1", "Team2"]]



sub[["Team1", "Team2", "Season"]] = sub[["Team1", "Team2", "Season"]].astype("int16")

sub = sub.merge(dt, left_on=["Team1", "Season"], right_on=["TeamID", "Season"])

sub = sub.merge(dt, left_on=["Team2", "Season"], right_on=["TeamID", "Season"])



difs = pd.DataFrame((0.1+sub[[x+"_x" for x in cols]].values)/(0.1+

                    sub[[x+"_y" for x in cols]].values), columns=diff)



sub = pd.concat([sub, difs], 1)

tourney_dresults.columns
import seaborn as sns



sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifRecentScore"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifRecentScore_A"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifRecentScoreDf"])

plt.show()
sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifRecentScoreDf"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["RecentScoreDf_x"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["RecentScoreDf_y"])

plt.show()
sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifRecentWins"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["RecentWins_x"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["RecentWins_y"])

plt.show()
sns.boxplot(tourney_dresults["Pred"], tourney_dresults["DifSeed"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["Seed_x"])

plt.show()

sns.boxplot(tourney_dresults["Pred"], tourney_dresults["Seed_y"])

plt.show()
dat = "both"

rs = 1999



tr_val_test = [0.8,0.1,0.1]

standardize = True



ensemble = 15

epochs = 1500

patience = 50

alpha_factor = 0.4



col = [x for x in tourney_dresults.columns if x not in ["ID", "Season", "Pred", "Games", "Team1", "Team2", "TeamID_x", "TeamID_y", "RecentGames", "DifRecentGames","DifGames"]]    

    

minicol = ["Seed_x", "Seed_y", "DifRecentWins", "DifSeed", "RecentScoreDf_x",

          "RecentScoreDf_y", "DifRecentScoreDf"]



if dat == "mini":

    train = tourney_dresults.copy()

    train.drop([x for x in train.columns if x not in minicol], 1, inplace=True)

    y_train = tourney_dresults["Pred"].ravel()



elif dat == "maxi":

    train = tourney_dresults.copy()

    train.drop([x for x in train.columns if x not in col], 1, inplace=True)

    y_train = tourney_dresults["Pred"].ravel()

    

else:

    train_min = tourney_dresults.copy()

    train_min.drop([x for x in train_min.columns if x not in minicol], 1, inplace=True)

    y_train_min = tourney_dresults["Pred"].ravel()   



    train_max = tourney_dresults.loc[tourney_dresults["Season"]>2009].copy()

    train_max.drop([x for x in train_max.columns if x not in col], 1, inplace=True)

    y_train_max = tourney_dresults.loc[tourney_dresults["Season"]>2009, "Pred"].ravel()    



if dat in ["mini", "maxi"]:    

    train, test, y_train, y_test = train_test_split(train, y_train, test_size=tr_val_test[2], random_state=rs

                                               )

    sets = []

    for x in range(ensemble):

        train, val, y_train, y_val = train_test_split(train, y_train, test_size=tr_val_test[1]/(tr_val_test[0]+tr_val_test[1]), random_state=rs+x+1)

        sets.append([train, val, y_train, y_val])



    test_s = sub.copy()

    if dat == "mini":

        test_s.drop([x for x in test_s.columns if x not in minicol], 1, inplace=True)

    else:

        test_s.drop([x for x in test_s.columns if x not in col], 1, inplace=True)

    

    stsc = MinMaxScaler()

    if dat == "mini":

        stsc.fit(tourney_dresults[minicol].values.astype("float64"))

    else:

        stsc.fit(tourney_dresults[col].values.astype("float64"))



    if standardize == True:

        test = stsc.transform(test)

        for x in range(ensemble):

            t,v,y_t,y_v = sets[x]

            t = stsc.transform(t)

            v = stsc.transform(v)

            sets[x] = [t,v,y_t,y_v]

        test_s = stsc.transform(test_s)  

    

    else:

        test = test.values

        for x in range(ensemble):

            t,v,y_t,y_v = sets[x]

            t = t.values

            v = v.values

            sets[x] = [t,v,y_t,y_v]

        test_s = test_s.values



else:    

    train_min, test_min, y_train_min, y_test_min = train_test_split(train_min, y_train_min, test_size=tr_val_test[2], random_state=rs

                                               )

    sets_min = []

    for x in range(ensemble):

        train_min, val_min, y_train_min, y_val_min = train_test_split(train_min, y_train_min, test_size=tr_val_test[1]/(tr_val_test[0]+tr_val_test[1]), random_state=rs+x+1                                               )

        sets_min.append([train_min, val_min, y_train_min, y_val_min])



    test_s_min = sub.copy()

    test_s_min.drop([x for x in test_s_min.columns if x not in minicol], 1, inplace=True)



    stsc = MinMaxScaler()

    stsc.fit(tourney_dresults[minicol].values.astype("float64"))



    if standardize == True:

        test_min = stsc.transform(test_min)

        for x in range(ensemble):

            t,v,y_t,y_v = sets_min[x]

            t = stsc.transform(t)

            v = stsc.transform(v)

            sets_min[x] = [t,v,y_t,y_v]

        test_s_min = stsc.transform(test_s_min)  

    

    else:

        test_min = test_min.values

        for x in range(ensemble):

            t,v,y_t,y_v = sets_min[x]

            t = t.values

            v = v.values

            sets_min[x] = [t,v,y_t,y_v]

        test_s_min = test_s_min.values



    train_max, test_max, y_train_max, y_test_max = train_test_split(train_max, y_train_max, test_size=tr_val_test[2], random_state=rs

                                               )

    sets_max = []

    for x in range(ensemble):

        train_max, val_max, y_train_max, y_val_max = train_test_split(train_max, y_train_max, test_size=tr_val_test[1]/(tr_val_test[0]+tr_val_test[1]), random_state=rs+x+1                                               )

        sets_max.append([train_max, val_max, y_train_max, y_val_max])



    test_s_max = sub.copy()

    test_s_max.drop([x for x in test_s_max.columns if x not in col], 1, inplace=True)



    stsc = MinMaxScaler()

    stsc.fit(tourney_dresults[col].values.astype("float64"))



    if standardize == True:

        test_max = stsc.transform(test_max)

        for x in range(ensemble):

            t,v,y_t,y_v = sets_max[x]

            t = stsc.transform(t)

            v = stsc.transform(v)

            sets_max[x] = [t,v,y_t,y_v]

        test_s_max = stsc.transform(test_s_max)  

    

    else:

        test_max = test_max.values

        for x in range(ensemble):

            t,v,y_t,y_v = sets_max[x]

            t = t.values

            v = v.values

            sets_max[x] = [t,v,y_t,y_v]

        test_s_max = test_s_max.values        

        
if dat in ["mini", "maxi"]:

    models = []

    histories = []

    pred = np.zeros([len(test),1], dtype="float64")

    for x in range(ensemble):

        print("Fitting model %i" % x)

        train, val, y_train, y_val = sets[x]

        param = {'num_leaves':31, 'num_trees':1000, 'objective':'binary'}

        num_round = 10

        model = lgb.train(param, [train, y_train], num_round, valid_sets=[val, y_val], early_stopping_rounds=50)



        #history = model.fit(train, y_train, validation_data=[val, y_val], callbacks=[EarlyStopping(min_delta=0.0001, patience=300,restore_best_weights=True)], batch_size=len(y_train), epochs=2000, verbose=0)    

        models.append(model)

        #histories.append(history)

        pred += model.predict(test)

        

else:        

    models_min = []

    histories_min = []

    pred_min = np.zeros([len(test_min),1], dtype="float64")

    print("Fitting MINI")

    for x in range(ensemble):

        train, val, y_train, y_val = sets_min[x]

        param = {'num_leaves':31, 'num_trees':1000, 'objective':'binary', "colsample_bytree":0.3}

        num_round = 10

        model = lgb.LGBMClassifier(max_depth=-1,

                               n_estimators=50000,

                               learning_rate=0.01,

                               colsample_bytree=0.1,

                               objective='binary', 

                               n_jobs=-1)        

        model.fit(train, y_train, eval_set=[(val, y_val)],verbose=0, early_stopping_rounds=500)

        models_min.append(model)

        histories_min.append(log_loss(y_val, np.reshape(model.predict_proba(val)[:,-1], [-1,1])))

        

        #print(model.predict_proba(test_min)[:,-1])

        pred_min += np.reshape(model.predict_proba(test_min)[:,-1], [-1,1])

        

        try:

            print("\t%i/%i: %.4f" % (x+1, ensemble, histories_min[-1]))  

        except:

            pass

    models_max = []

    histories_max = []

    pred_max = np.zeros([len(test_max),1], dtype="float64")

    print("Fitting MAXI")

    for x in range(ensemble):

        train, val, y_train, y_val = sets_max[x]

        param = {'num_leaves':31, 'num_trees':1000, 'objective':'binary', "colsample_bytree":0.3}

        num_round = 10

        model = lgb.LGBMClassifier(max_depth=-1,

                               n_estimators=50000,

                               learning_rate=0.01,

                               colsample_bytree=0.1,

                               objective='binary', 

                               n_jobs=-1)        

        history = model.fit(train, y_train, eval_set=[(val, y_val)], verbose=0, early_stopping_rounds=1000)        

        models_max.append(model)

        histories_max.append(log_loss(y_val, np.reshape(model.predict_proba(val)[:,-1], [-1,1])))

        pred_max +=  np.reshape(model.predict_proba(test_max)[:,-1], [-1,1])

        try:

            print("\t%i/%i: %.4f" % (x+1, ensemble, histories_max[-1]))   

        except:

            pass
def pad(c, dim):

    c = c + [min(c)]*(dim-len(c))

    return(c)
alphas = []

hists = []



hmin = histories_min

hmax = histories_max

bests = [x for x in hmin+hmax]

for x in range(ceil(ensemble*alpha_factor)):

    #print(bests)

    best = np.argmin(bests)

    #print(best)

    #print(bests[best])

    if best < ensemble:

        alphas.append(["min",models_min[best]])

        hists.append(hmin[best])

        bests[best] = 20

    else:

        alphas.append(["max",models_max[best-ensemble]])

        hists.append(hmax[best-ensemble])        

        bests[best] = 20     
nalphas = []

nhists = []



hmin = [log_loss(y_test_min, x[1].predict_proba(test_min)[:,-1]) for x in alphas if x[0]=="min"]

hmax = [log_loss(y_test_max, x[1].predict_proba(test_max)[:,-1]) for x in alphas if x[0]=="max"]

alphas = [x for x in alphas if x[0]=="min"]+[x for x in alphas if x[0]=="max"]



bests = hmin+hmax

for x in range(ceil(len(bests)*alpha_factor)):

    best = np.argmin(bests)

    if alphas[best][0] == "min":

        nalphas.append(alphas[best])

        nhists.append(bests[best])

        bests[best] = 20

    else:

        nalphas.append(alphas[best])

        nhists.append(bests[best])            

        bests[best] = 20

        



loss = sum(nhists)/len(nhists)



print("Optimized LOSS: %.3f" % loss)

print(nhists)
if dat in ["mini", "maxi"]:

    pred = np.zeros([len(test_s),1])

    for x in range(ensemble):

        pred += models[x].predict_proba(test_s)[:,-1]

else:

    pred = np.zeros([len(test_s_min),1])

    for x in range(len(alphas)):

        if alphas[x][0] == "min":

            pred += np.reshape(alphas[x][1].predict_proba(test_s_min)[:,-1], [-1,1])

        else:

            pred += np.reshape(alphas[x][1].predict_proba(test_s_max)[:,-1], [-1,1])

subm = pd.DataFrame(np.zeros([len(sub), 2]), columns=["ID", "Pred"]) 

subm["Pred"] = pred/len(alphas)

subm = subm[["ID", "Pred"]]

subm["ID"] = sub["ID"].astype("str")

subm.to_csv("submission.csv", index=False)