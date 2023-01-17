import numpy as np

import pandas as pd

pd.options.display.max_rows = 999

pd.options.mode.chained_assignment = None

import json

from sklearn.model_selection import train_test_split
def CustomParser(data):

    import json

    j1 = json.loads(data)

    return j1
#train_chunks = pd.read_csv("../input/data-science-bowl-2019/train.csv", chunksize=100000 ,converters={'event_data':CustomParser}, parse_dates=["timestamp"])

train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")
agg_dict = {'timestamp':'max',

            'event_count':'max',

            'game_time': "sum",

            "title" : "max",

            'is_Clip' : "sum",

            "is_Activity" : "sum",

            "is_Game" : "sum",

            "is_Assessment" : "sum",

            "is_NONE": "sum",

            "is_MAGMAPEAK" : "max",

            "is_TREETOPCITY" : "max",

            "is_CRYSTALCAVES" : "max",

            "Assessment" : "sum",

            "correct" : "sum",

            "incorrect": "sum"}
def process_chunk(chunk):

#    chunk["event_data"] = chunk["event_data"].apply(lambda row: json.loads(row)) #parse JSON

    chunk["game_time"] = chunk["game_time"]/(1000*60) #game_time in MINUTES

    chunk_with_dummies = pd.get_dummies(chunk, prefix="is", columns=["type", "world"])

    chunk_with_dummies["Assessment"] = chunk_with_dummies["event_data"][chunk_with_dummies.is_Assessment == 1].apply(lambda row: 1 if ((row["event_code"] == 4100 ) | (row["event_code"] == 4110 )) else 0)

    chunk_with_dummies["correct"] = chunk_with_dummies["event_data"][chunk_with_dummies.Assessment == True].apply(lambda row: 1 if row["correct"]== True else 0)

    chunk_with_dummies["incorrect"] = chunk_with_dummies["event_data"][chunk_with_dummies.Assessment == True].apply(lambda row: 1 if row["correct"]== False else 0)

    inst_id_game_session = chunk_with_dummies.groupby(["installation_id", "game_session"]).agg(agg_dict)

    return inst_id_game_session
def concat_chunk_remove_dup(df_chunk):

    chunk_list = []  



    for chunk in df_chunk:  

        chunk_filter = process_chunk(chunk)

        chunk_list.append(chunk_filter)

    

    with_dup = pd.concat(chunk_list).reset_index()

    without_dup = with_dup.groupby(["installation_id", "game_session"]).agg(agg_dict)

    time_sorted_without_dup = without_dup.sort_values(by=["installation_id","timestamp"])

    return time_sorted_without_dup
#train_without_dup = concat_chunk_remove_dup(train_chunks)
#List of installation_ids that have at least one assessment.

def check_inst_ids(df):

    list_inst_id = set([i[0] for i in df.index[df.Assessment > 0]])

    print("Number of unique installation_id that have done at least one assessment: " + str(len(list_inst_id)))

    print("Total number of game_session : " + str(len(df.index)))

    list_game_session = [i[1] for i in df.index[df.Assessment > 0]]

    print("Number of game_session that include one assessment: " + str(len(list_game_session)))

    return list_inst_id
#list_inst_id = check_inst_ids(train_without_dup)
def add_features(df, id_list):

    data = []

    for inst_id in id_list:

        inst = df.loc[inst_id, :]

        inst["clips_MAGMA"] = inst.is_Clip * inst.is_MAGMAPEAK

        inst["activities_MAGMA"] = inst.is_Activity.apply(lambda x: 1 if x > 0 else 0) * inst.is_MAGMAPEAK

        inst["games_MAGMA"] = inst.is_Game.apply(lambda x: 1 if x > 0 else 0) * inst.is_MAGMAPEAK

        inst["clips_TREE"] = inst.is_Clip * inst.is_TREETOPCITY

        inst["activities_TREE"] = inst.is_Activity.apply(lambda x: 1 if x > 0 else 0) * inst.is_TREETOPCITY

        inst["games_TREE"] = inst.is_Game.apply(lambda x: 1 if x > 0 else 0)  * inst.is_TREETOPCITY

        inst["clips_CAVES"] = inst.is_Clip * inst.is_CRYSTALCAVES

        inst["activities_CAVES"] = inst.is_Activity.apply(lambda x: 1 if x > 0 else 0) * inst.is_CRYSTALCAVES

        inst["games_CAVES"] = inst.is_Game.apply(lambda x: 1 if x > 0 else 0) * inst.is_CRYSTALCAVES

        inst.insert(24, "cl_MAGMA_today", 0)

        inst.insert(25, "ac_MAGMA_today", 0)

        inst.insert(26, "ga_MAGMA_today", 0)

        inst.insert(27, "cl_TREE_today", 0)

        inst.insert(28, "ac_TREE_today", 0)

        inst.insert(29, "ga_TREE_today", 0)

        inst.insert(30, "cl_CAVES_today", 0)

        inst.insert(31, "ac_CAVES_today", 0)

        inst.insert(32, "ga_CAVES_today", 0)

        inst["installation_id"] = inst_id

        inst["date"] = inst.timestamp.dt.date

        inst.insert(35, "relevant_clips", 0)

        inst.insert(36, "relevant_activities", 0)

        inst.insert(37, "relevant_games", 0)

        inst.insert(38, "rel_clips_today", 0)

        inst.insert(39, "rel_act_today", 0)

        inst.insert(40, "rel_games_today", 0)

        inst["weekday"] = inst.timestamp.dt.weekday.apply(lambda x: 1 if x <5 else 0)

        inst["hour"] = inst.timestamp.dt.hour

        inst.insert(43, "game_time_total", 0)

        inst.insert(44, "incorrect_to_date", 0)

        inst.insert(45, "correct_to_date", 0)

        inst.insert(46, "same_correct_before",0)

        inst["correct_magma"] = inst.correct * inst.is_MAGMAPEAK #47

        inst["incorrect_magma"] = inst.incorrect * inst.is_MAGMAPEAK #48

        inst["correct_treetop"] = inst.correct * inst.is_TREETOPCITY # 49

        inst["incorrect_treetop"] = inst.correct * inst.is_TREETOPCITY # 50

        inst["correct_crystal"] = inst.correct * inst.is_CRYSTALCAVES # 51

        inst["incorrect_crystal"] = inst.correct * inst.is_CRYSTALCAVES # 52

        inst.insert(53, "same_incorrect_before", 0)

        inst["is_mushroom"] = inst.title.apply(lambda x: 1 if x == "Mushroom Sorter (Assessment)" else 0) 

        inst["is_chest"] = inst.title.apply(lambda x: 1 if x == "Chest Sorter (Assessment)" else 0)

        inst["is_cart"] = inst.title.apply(lambda x: 1 if x == "Cart Balancer (Assessment)" else 0)

        inst["is_cauldron"] = inst.title.apply(lambda x: 1 if x == "Cauldron Filler (Assessment)" else 0)

        inst["is_bird"] = inst.title.apply(lambda x: 1 if x == "Bird Measurer (Assessment)" else 0)

        inst["correct_mushroom"] = inst.correct * inst.is_mushroom

        inst["correct_chest"] = inst.correct * inst.is_chest

        inst["correct_cart"] = inst.correct * inst.is_cart

        inst["correct_cauldron"] = inst.correct * inst.is_cauldron

        inst["correct_bird"] = inst.correct * inst.is_bird

        inst.insert(64, "corr_mushroom_before", 0)

        inst.insert(65, "corr_chest_before", 0)        

        inst.insert(66, "corr_cart_before", 0)

        inst.insert(67, "corr_cauldron_before", 0)        

        inst.insert(68, "corr_bird_before", 0)

        inst["incorrect_mushroom"] = inst.incorrect * inst.is_mushroom

        inst["incorrect_chest"] = inst.incorrect * inst.is_chest

        inst["incorrect_cart"] = inst.incorrect * inst.is_cart

        inst["incorrect_cauldron"] = inst.incorrect * inst.is_cauldron

        inst["incorrect_bird"] = inst.incorrect * inst.is_bird

        inst.insert(74, "incorr_mushroom_before", 0)

        inst.insert(75, "incorr_chest_before", 0)        

        inst.insert(76, "incorr_cart_before", 0)

        inst.insert(77, "incorr_cauldron_before", 0)        

        inst.insert(78, "incorr_bird_before", 0)

        inst["attempts"] = inst.correct + inst.incorrect

        inst.insert(80, "attempts_before", 0)

                

        for i in range(len(inst)):

            if inst.iloc[i, 7] > 0: #For each assessment attempt (is_Assessment), calculates all clips/activities/games by world done before.

                inst.iloc[i,15] = np.sum(inst.clips_MAGMA[:i]) #Clips watched in Magma so far

                inst.iloc[i,16] = np.sum(inst.activities_MAGMA[:i])

                inst.iloc[i,17] = np.sum(inst.games_MAGMA[:i])

                inst.iloc[i,18] = np.sum(inst.clips_TREE[:i])

                inst.iloc[i,19] = np.sum(inst.activities_TREE[:i])

                inst.iloc[i,20] = np.sum(inst.games_TREE[:i])

                inst.iloc[i,21] = np.sum(inst.clips_CAVES[:i])

                inst.iloc[i,22] = np.sum(inst.activities_CAVES[:i])

                inst.iloc[i,23] = np.sum(inst.games_CAVES[:i])

                inst.iloc[i,24] = np.sum(inst.clips_MAGMA[:i][inst.date == inst.iloc[i,34]]) #inst.iloc[i,34] is the date

                inst.iloc[i,25] = np.sum(inst.activities_MAGMA[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,26] = np.sum(inst.games_MAGMA[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,27] = np.sum(inst.clips_TREE[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,28] = np.sum(inst.activities_TREE[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,29] = np.sum(inst.games_TREE[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,30] = np.sum(inst.clips_CAVES[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,31] = np.sum(inst.activities_CAVES[:i][inst.date == inst.iloc[i,34]])

                inst.iloc[i,32] = np.sum(inst.games_CAVES[:i][inst.date == inst.iloc[i,34]])

                if inst.iloc[i, 9] == 1: #is_MAGMA

                    inst.iloc[i,35] = inst.iloc[i,15] #to date

                    inst.iloc[i,36] = inst.iloc[i, 16]

                    inst.iloc[i,37] = inst.iloc[i, 17]

                    inst.iloc[i,38] = inst.iloc[i, 24] #today

                    inst.iloc[i,39] = inst.iloc[i, 25]

                    inst.iloc[i,40] = inst.iloc[i, 26]

                    inst.iloc[i,46] = np.sum(inst.correct_magma[:i])#somme des correct_magma avant

                    inst.iloc[i,53] = np.sum(inst.incorrect_magma[:i]) # somme incorrect_magma avant

                if inst.iloc[i, 10] == 1: #is_TREETOP

                    inst.iloc[i,35] = inst.iloc[i,18]

                    inst.iloc[i,36] = inst.iloc[i, 19]

                    inst.iloc[i,37] = inst.iloc[i, 20]

                    inst.iloc[i,38] = inst.iloc[i, 27]

                    inst.iloc[i,39] = inst.iloc[i, 28]

                    inst.iloc[i,40] = inst.iloc[i, 29]

                    inst.iloc[i,46] = np.sum(inst.correct_treetop[:i])

                    inst.iloc[i,53] = np.sum(inst.incorrect_treetop[:i]) 

                if inst.iloc[i, 11] == 1: #is_CRYSTAL

                    inst.iloc[i,35] = inst.iloc[i,21]

                    inst.iloc[i,36] = inst.iloc[i, 22]

                    inst.iloc[i,37] = inst.iloc[i, 23]

                    inst.iloc[i,38] = inst.iloc[i, 30]

                    inst.iloc[i,39] = inst.iloc[i, 31]

                    inst.iloc[i,40] = inst.iloc[i, 32]

                    inst.iloc[i,46] = np.sum(inst.correct_crystal[:i])

                    inst.iloc[i,53] = np.sum(inst.incorrect_crystal[:i])

                inst.iloc[i,43] = np.sum(inst.game_time[:i+1])

                inst.iloc[i,44] = np.sum(inst.incorrect[:i])

                inst.iloc[i,45] = np.sum(inst.correct[:i])

                inst.iloc[i,64] = np.sum(inst.correct_mushroom[:i])

                inst.iloc[i,65] = np.sum(inst.correct_chest[:i])

                inst.iloc[i,66] = np.sum(inst.correct_cart[:i])

                inst.iloc[i,67] = np.sum(inst.correct_cauldron[:i])

                inst.iloc[i,68] = np.sum(inst.correct_bird[:i])

                inst.iloc[i,74] = np.sum(inst.incorrect_mushroom[:i])

                inst.iloc[i,75] = np.sum(inst.incorrect_chest[:i])

                inst.iloc[i,76] = np.sum(inst.incorrect_cart[:i])

                inst.iloc[i,77] = np.sum(inst.incorrect_cauldron[:i])

                inst.iloc[i,78] = np.sum(inst.incorrect_bird[:i])

                inst.iloc[i,80] = np.sum(inst.attempts[:i])

        data.append(inst)

    return pd.concat(data)
#train_with_features = add_features(train_without_dup, list_inst_id)
#train_with_features = train_with_features.sort_values(by=["installation_id", "timestamp"])
cols = ["installation_id", "timestamp", "game_time","title", 'clips_MAGMA', 'activities_MAGMA', 'games_MAGMA',

       'clips_TREE', 'activities_TREE', 'games_TREE', 'clips_CAVES',

       'activities_CAVES', 'games_CAVES', 'cl_MAGMA_today', 'ac_MAGMA_today',

       'ga_MAGMA_today', 'cl_TREE_today', 'ac_TREE_today', 'ga_TREE_today',

       'cl_CAVES_today', 'ac_CAVES_today', 'ga_CAVES_today', "relevant_clips", "relevant_activities",

        "relevant_games", "rel_clips_today", "rel_act_today", "rel_games_today",

        "date", "hour", "weekday", "Assessment", "correct", "incorrect",

        "game_time_total", "correct_to_date", "incorrect_to_date",

        "same_correct_before", "same_incorrect_before", "corr_mushroom_before",

        "corr_chest_before", "corr_cart_before", "corr_cauldron_before",

        "corr_bird_before","incorr_mushroom_before",

        "incorr_chest_before", "incorr_cart_before", "incorr_cauldron_before",

        "incorr_bird_before", "attempts_before"

       ]

#train_reordered = train_with_features[cols]
#df_for_model = train_reordered[train_reordered.Assessment > 0]
#subset = [game_session for game_session in df_for_model.index if game_session in list(train_labels.game_session)]

#print(len(subset))

#df_for_model = df_for_model.loc[subset]
#df_for_model.to_csv("train_df_for_model.csv")
df_for_model = pd.read_csv("../input/traindata12/train_df_for_model.csv", index_col="game_session", parse_dates=["timestamp"])

df_for_model = pd.get_dummies(df_for_model, columns=["title"])

#print(df_for_model.shape)

cols_uploaded = df_for_model.columns
df_for_model = df_for_model.join(train_labels.set_index('game_session')["accuracy_group"])
from xgboost import XGBClassifier

from xgboost import XGBRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
features = ['clips_MAGMA', 'activities_MAGMA', 'games_MAGMA',

       'clips_TREE', 'activities_TREE', 'games_TREE', 'clips_CAVES',

       'activities_CAVES', 'games_CAVES', 'cl_MAGMA_today', 'ac_MAGMA_today', 'ga_MAGMA_today', 'cl_TREE_today',

       'ac_TREE_today', 'ga_TREE_today', 'cl_CAVES_today', 'ac_CAVES_today',

       'ga_CAVES_today', "relevant_clips", "relevant_activities",

            "relevant_games", "rel_clips_today", "rel_act_today", "rel_games_today",

        'title_Bird Measurer (Assessment)', 'title_Cart Balancer (Assessment)',

       'title_Cauldron Filler (Assessment)', 'title_Chest Sorter (Assessment)',

       'title_Mushroom Sorter (Assessment)', "weekday", "hour",

            "game_time_total", "incorrect_to_date", "correct_to_date",

            "same_correct_before", "same_incorrect_before",

           "corr_mushroom_before",

        "corr_chest_before", "corr_cart_before", "corr_cauldron_before",

        "corr_bird_before", "incorr_mushroom_before",

        "incorr_chest_before", "incorr_cart_before", "incorr_cauldron_before",

        "incorr_bird_before"]



X = df_for_model[features]

y = df_for_model[["accuracy_group"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



xgb = XGBRegressor()

xgb.fit(X_train, y_train)

scores_test = xgb.predict(X_test)

#thresholds = np.percentile(scores_test, [24,35,47])

thresholds = [1.4, 1.89, 1.96]

#print(thresholds)

X_test["scores"] = scores_test

X_test["accuracy_group"] = X_test.scores.apply(lambda x : 3 if x>thresholds[2] else (2 if x>thresholds[1] else (1 if x>thresholds[0] else 0)))



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, X_test.accuracy_group)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



test_chunks = pd.read_csv("../input/data-science-bowl-2019/test.csv", chunksize=100000, converters={'event_data':CustomParser}, parse_dates=["timestamp"])
test_without_dup = concat_chunk_remove_dup(test_chunks)

test_list_inst_id = set([i[0] for i in test_without_dup.index[test_without_dup.is_Assessment == 1.0]])

print("Number of unique installation_id in test data that have done at least one assessment: " + str(len(test_list_inst_id)))

print("Total number of game_session in test data : " + str(len(test_without_dup.index)))

list_game_session_test = [i[1] for i in test_without_dup.index[test_without_dup.Assessment > 0]]

print("Number of game_session that include one assessment: " + str(len(list_game_session_test)))
test_with_features = add_features(test_without_dup, test_list_inst_id)

submission_1000 = test_with_features.groupby(["installation_id"]).last()

submission_1000 = pd.get_dummies(submission_1000, columns=["title"])

submission_1000 = submission_1000[[x for x in cols_uploaded if x not in ["installation_id"]]]

#print(submission_1000.columns)
X_test_data = submission_1000[features]



model_all_data = XGBRegressor()

model_all_data.fit(X, y)

test_labels = model_all_data.predict(X_test_data)
submission = pd.DataFrame(test_labels, X_test_data.index)

submission.columns = ["accuracy_group"]
submission.accuracy_group = submission.accuracy_group.apply(lambda x : 3 if x>thresholds[2] else (2 if x>thresholds[1] else (1 if x>thresholds[0] else 0)))

submission.accuracy_group.value_counts()
submission.head()
submission.to_csv("submission.csv")