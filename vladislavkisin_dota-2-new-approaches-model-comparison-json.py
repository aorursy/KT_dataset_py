import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

from sklearn.metrics import roc_auc_score



from sklearn.ensemble import RandomForestClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

PATH_TO_DATA = '../input/'



#Importing initial training dataset with targets and test dataset



train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), 

                                    index_col='match_id_hash')

train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'train_targets.csv'), 

                                   index_col='match_id_hash')

test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 

                                            'test_features.csv'), 

                                   index_col='match_id_hash')

print(train_df.shape, train_targets.shape, test_df.shape)
train_targets.head()
%%time

y_train = train_targets.radiant_win #extract the target variable



#Now make a train-test split, we'll see that results on the holdout set correlate with CV results.



X_train_part, X_valid, y_train_part, y_valid = train_test_split(train_df, y_train, test_size = 0.3, random_state=0) #fixing random_state



#Settling a CV scheme.

cv = ShuffleSplit(n_splits=5, random_state=1) #using a shuffle split for CV 



#Implement RF with just 100 estimators not to wait too long.

rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(X_train_part, y_train_part)



#Count CV scoring and houldout scoring: 

holdout_score = roc_auc_score(y_valid, rf.predict_proba(X_valid)[:,1])

cv_score = cross_val_score(rf, train_df, y_train, cv=cv, scoring = 'roc_auc') 

#Let's look at the results.

print('CV scores: ', cv_score)

print('CV mean: ', cv_score.mean())

print('Holdout score: ', holdout_score)
train_df.head()
idx_split = train_df.shape[0]

full_df = pd.concat((train_df, test_df))



print(train_df.shape, test_df.shape, full_df.shape)
cols = [] 

for i in full_df.columns[5:29]: #list of columns for r1 player

    if i[3:] != 'hero_id' and i[3:] != 'firstblood_claimed':

        cols.append(i[3:]) #drop r1_

print(cols)
def substract_numeric_features (df, feature_suffixes):

    col_names=[]

    df_out = df.copy()

    for feat_suff in feature_suffixes:

        for index in range(1,6):

            df_out[f'{index}_{feat_suff}_substract'] = df[f'r{index}_{feat_suff}'] - df[f'd{index}_{feat_suff}'] # e.g. r1_kills - d1_kills

            col_names.append(f'd{index}_{feat_suff}')

            col_names.append(f'r{index}_{feat_suff}')

    df_out.drop(columns = col_names, inplace=True)

    return df_out



#Run the function

full_df_mod = substract_numeric_features(full_df, cols)

full_df_mod.head()
def combine_sub_features (df_out, feature_suffixes):

    for feat_suff in feature_suffixes:

            player_col_names = [f'{i}_{feat_suff}_substract' for i in range(1,6)] # e.g. 1_gold_substract

            

            df_out[f'{feat_suff}_max_substract'] = df_out[player_col_names].max(axis=1) # e.g. gold_max_substract

            

            df_out[f'{feat_suff}_min_substract'] = df_out[player_col_names].min(axis=1) # e.g. gold_min_substract

            

            df_out[f'{feat_suff}_sum_substract'] = df_out[player_col_names].sum(axis=1) # e.g. gold_sum_substract



            

            df_out.drop(columns=player_col_names, inplace=True) # remove teammembers' substract features from the dataset

    return df_out



#Run the function. Suffixes remain the same

full_df_mod = combine_sub_features(full_df_mod, cols)

full_df_mod.head()
%%time

#Remember we need to use only training part of the full set

X_train_part_1, X_valid_1, y_train_part_1, y_valid_1 = train_test_split(full_df_mod[:idx_split], y_train, test_size = 0.3, random_state=0) #fixing random_state



rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(X_train_part_1, y_train_part_1)



#Count CV scoring and houldout scoring: 

holdout_score_1 = roc_auc_score(y_valid_1, rf.predict_proba(X_valid_1)[:,1])

cv_score_1 = cross_val_score(rf, full_df_mod[:idx_split], y_train, cv=cv, scoring = 'roc_auc') 
#New results.

print('CV scores: ', cv_score_1)

print('CV mean: ', cv_score_1.mean())

print('CV std:', cv_score_1.std())

print('Holdout score: ', holdout_score_1)

print('Better results on CV: ', cv_score_1>cv_score)
def herotype_approach(df):

    r_heroes = ['r%s_hero_id' %i for i in range(1,6)] # e.g. r1_hero_id...

    d_heroes = ['d%s_hero_id' %i for i in range(1,6)] # e.g. d1_hero_id...

    r_herotypes = ['r%s_hero_type' %i for i in range(1,6)] # e.g. r1_hero_type...

    d_herotypes = ['d%s_hero_type' %i for i in range(1,6)] # e.g. d1_hero_type...



    df['r_hero_invar_sum'] = np.log(df[r_heroes]).sum(axis=1) #sum of logs of hero ids for the team r

    df['d_hero_invar_sum'] = np.log(df[d_heroes]).sum(axis=1) #sum of logs of hero ids for the team d

    df['hero_invar_sum_diff'] = df['r_hero_invar_sum'] - df['d_hero_invar_sum'] #their difference (don't try to find the meaning)

    

    df[r_herotypes] = df[r_heroes].apply(lambda x: (x//40)+1) #hero types like 1,2,3 supposing there's about equal number of heroes of each type

    df[d_herotypes] = df[d_heroes].apply(lambda x: (x//40)+1)

    

    df['r_invar_herotype_sum'] = np.log(df[r_herotypes]).sum(axis=1).astype(str) # findning an invariant sum to treat as categorial

    df['d_invar_herotype_sum'] = np.log(df[d_herotypes]).sum(axis=1).astype(str)

    

    return df



full_df_mod = herotype_approach(full_df_mod)
def hero_approach(df):

    for team in 'r', 'd':

        players = [f'{team}{i}' for i in range(1, 6)]

        hero_columns = [f'{player}_hero_id' for player in players]



        d = pd.get_dummies(df[hero_columns[0]])

        for c in hero_columns[1:]:

            d += pd.get_dummies(df[c])

        df = pd.concat([df, d.add_prefix(f'{team}_hero_')], axis=1)

        df.drop(columns=hero_columns, inplace=True)

    return df



full_df_mod = hero_approach(full_df_mod)
r_firstblood = ['r%s_firstblood_claimed' %i for i in range(1,6)] 

d_firstblood = ['d%s_firstblood_claimed' %i for i in range(1,6)] 

r_herotypes = ['r%s_hero_type' %i for i in range(1,6)]

d_herotypes = ['d%s_hero_type' %i for i in range(1,6)]



full_df_dum = pd.get_dummies(full_df_mod, columns = ['r_invar_herotype_sum', 'd_invar_herotype_sum'] + r_firstblood + d_firstblood)

full_df_dum.head()
%%time

#Remember we need to use only training part of the full set

X_train_part_2, X_valid_2, y_train_part_2, y_valid_2 = train_test_split(full_df_dum[:idx_split], y_train, test_size = 0.3, random_state=0) #fixing random_state



rf = RandomForestClassifier(n_estimators=100, random_state=1)

rf.fit(X_train_part_2, y_train_part_2)



#Count CV scoring and houldout scoring: 

holdout_score_2 = roc_auc_score(y_valid_2, rf.predict_proba(X_valid_2)[:,1])

cv_score_2 = cross_val_score(rf, full_df_dum[:idx_split], y_train, cv=cv, scoring = 'roc_auc') 
#New results.

print('CV scores: ', cv_score_2)

print('CV mean: ', cv_score_2.mean())

print('CV std:', cv_score_2.std())

print('Holdout score: ', holdout_score_2)

print('Better results on CV: ', cv_score_2>cv_score_1)
%%time

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

full_df_scaled = scaler.fit_transform(full_df_dum) #scalling the full dataset, that's not correct but it saves time



X_train_part_lr, X_valid_lr, y_train_part_lr, y_valid_lr = train_test_split(full_df_scaled[:idx_split], 

                                                                        y_train, test_size = 0.3, random_state=0) #fixing random_state

lr = LogisticRegression(random_state=0, solver='liblinear')

lr.fit(X_train_part_lr, y_train_part_lr)



lr_ho_score =  roc_auc_score(y_valid_lr, lr.predict_proba(X_valid_lr)[:,1])

lr_cv_score = cross_val_score(lr, full_df_scaled[:idx_split], y_train, cv=cv, scoring = 'roc_auc') 



del full_df_dum
#Logistic regression results.

print('CV scores LR: ', lr_cv_score)

print('CV mean LR: ', lr_cv_score.mean())

print('CV std LR:', lr_cv_score.std())

print('Holdout score LR: ', lr_ho_score)
rf_cv_score, rf_ho_score = cv_score_2, holdout_score_2 



print('CV scores RF: ', rf_cv_score)

print('CV mean RF: ', rf_cv_score.mean())

print('CV std RF:', rf_cv_score.std())

print('Holdout score RF: ', rf_ho_score)

%%time

from catboost import CatBoostClassifier

#We'll use full_df_mod without dummies and mark categorial vars

X_train_part_ctb, X_valid_ctb, y_train_part_ctb, y_valid_ctb = train_test_split(full_df_mod[:idx_split], 

                                                                        y_train, test_size = 0.3, random_state=0) #fixing random_state

cat_vars = ['r_invar_herotype_sum', 'd_invar_herotype_sum'] + r_firstblood + d_firstblood #all the vars that we got dummies of



#Let it train for 200 iterations not to wait too long

ctb = CatBoostClassifier(iterations = 200, random_state=1, verbose=False, task_type='GPU', eval_metric='AUC', cat_features=cat_vars)



#We'll look at an online validation plot

ctb.fit(X_train_part_ctb, y_train_part_ctb.astype(float), eval_set=(X_valid_ctb, y_valid_ctb.astype(float)), plot=True)



ctb_ho_score =  roc_auc_score(y_valid_ctb.astype(float), ctb.predict_proba(X_valid_ctb)[:,1])

ctb_cv_score = cross_val_score(ctb, full_df_mod[:idx_split], y_train.astype(float), cv=cv, scoring = 'roc_auc') 

print('CV scores CTB: ', ctb_cv_score)

print('CV mean CTB: ', ctb_cv_score.mean())

print('CV std CTB:', ctb_cv_score.std())

print('Holdout score CTB: ', ctb_ho_score)
#!pip install keras

#!pip install tensorflow
from keras.models import Sequential

from keras.layers import Dense, BatchNormalization

from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf



from keras import backend as K

from keras import regularizers

from keras import optimizers



#Defining ROC AUC in Keras for evaluation.

def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
# to find a number of input dimensions

full_df_scaled.shape
def model_function():

    model = Sequential()

    model.add(Dense(50, input_dim = 396, kernel_initializer='normal', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])

    return model
keras_net = KerasClassifier(build_fn=model_function, epochs=10, batch_size=32, verbose=1) #it's an sklearn wrapper for model

keras_net.fit(X_train_part_lr, y_train_part_lr.astype(float)) #we'll use scaled training part like for LR

nn_ho_score =  roc_auc_score(y_valid_lr.astype(float), keras_net.predict_proba(X_valid_lr)[:,1])
keras_net = KerasClassifier(build_fn=model_function, epochs=10, batch_size=32, verbose=False) #turn off the verbose



nn_cv_score = cross_val_score(keras_net, full_df_scaled[:idx_split], y_train.astype(float), cv=cv, scoring = 'roc_auc') 
print('CV scores nn: ', nn_cv_score)

print('CV mean nn: ', nn_cv_score.mean())

print('CV std nn:', nn_cv_score.std())

print('Holdout score nn: ', nn_ho_score)
import json 



#Collect needed columns names

with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:

        for i in range(150):

            first_line = fin.readline()

            data_json = json.loads(first_line)

data_json.keys()

key = []

for i in data_json['players'][9].keys():

    if i not in cols: #remember we've settled columns from full dataset

        key.append(i)
import collections

from tqdm import tqdm_notebook

def read_matches(matches_file):

    

    MATCHES_COUNT = {

        'test_matches.jsonl': 10000,

        'train_matches.jsonl': 39675,

    }

    _, filename = os.path.split(matches_file)

    total_matches = MATCHES_COUNT.get(filename)

    

    with open(matches_file) as fin:

        for line in tqdm_notebook(fin, total=total_matches):

            yield json.loads(line)



#Extracting function



def extract_features_csv(match, keys):

    row = [

        ('match_id_hash', match['match_id_hash']),

    ]

        

    for slot, player in enumerate(match['players']):

        if slot < 5:

            player_name = 'r%d' % (slot + 1)

        else:

            player_name = 'd%d' % (slot - 4)

# The main idea: if we have int or float or bool - return it, else - return the length of the item

        for field in keys:

            if (type(player[field]) == int) or (type(player[field]) == float) or (type(player[field]) == bool): 

                column_name = '%s_%s' % (player_name, field)

                row.append((column_name, player[field]))

            else:

                column_name = '%s_%s' % (player_name, field)

                row.append((column_name, len(player[field])))

    return collections.OrderedDict(row)


#df_new_features = []

#for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):

#    match_id_hash = match['match_id_hash']

#    features = extract_features_csv(match, key)



#    df_new_features.append(features)

#df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')

#df_new_features.head()