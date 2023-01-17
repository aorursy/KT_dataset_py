import warnings

warnings.filterwarnings('ignore')



import pandas as pd, numpy as np, seaborn as sns

import math, json, os, random

from matplotlib import pyplot as plt

from tqdm import tqdm



import tensorflow as tf

import tensorflow_addons as tfa

import keras.backend as K



from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from sklearn.cluster import KMeans
def seed_everything(seed = 34):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    

seed_everything()
#get comp data

train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
print(train.columns)
#sneak peak

print(train.shape)

if ~train.isnull().values.any(): print('No missing values')

train.head()
#sneak peak

print(test.shape)

if ~test.isnull().values.any(): print('No missing values')

test.head()
#sneak peak

print(sample_sub.shape)

if ~sample_sub.isnull().values.any(): print('No missing values')

sample_sub.head()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

sns.kdeplot(train['signal_to_noise'], shade=True, ax=ax[0])

sns.countplot(train['SN_filter'], ax=ax[1])



ax[0].set_title('Signal/Noise Distribution')

ax[1].set_title('Signal/Noise Filter Distribution');
print(f"Samples with signal_to_noise greater than 1: {len(train.loc[(train['signal_to_noise'] > 1 )])}")

print(f"Samples with SN_filter = 1: {len(train.loc[(train['SN_filter'] == 1 )])}")

print(f"Samples with signal_to_noise greater than 1, but SN_filter == 0: {len(train.loc[(train['signal_to_noise'] > 1) & (train['SN_filter'] == 0)])}")
def read_bpps_sum(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))

    return bpps_arr



def read_bpps_max(df):

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))

    return bpps_arr



def read_bpps_nb(df):

    #mean and std from https://www.kaggle.com/symyksr/openvaccine-deepergcn 

    bpps_nb_mean = 0.077522

    bpps_nb_std = 0.08914

    bpps_arr = []

    for mol_id in df.id.to_list():

        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")

        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]

        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        bpps_arr.append(bpps_nb)

    return bpps_arr 



train['bpps_sum'] = read_bpps_sum(train)

test['bpps_sum'] = read_bpps_sum(test)

train['bpps_max'] = read_bpps_max(train)

test['bpps_max'] = read_bpps_max(test)

train['bpps_nb'] = read_bpps_nb(train)

test['bpps_nb'] = read_bpps_nb(test)



#sanity check

train.head()
fig, ax = plt.subplots(3, figsize=(15, 10))

sns.kdeplot(np.array(train['bpps_max'].to_list()).reshape(-1),

            color="Blue", ax=ax[0], label='Train')

sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_max'].to_list()).reshape(-1),

            color="Red", ax=ax[0], label='Public test')

sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_max'].to_list()).reshape(-1),

            color="Green", ax=ax[0], label='Private test')

sns.kdeplot(np.array(train['bpps_sum'].to_list()).reshape(-1),

            color="Blue", ax=ax[1], label='Train')

sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_sum'].to_list()).reshape(-1),

            color="Red", ax=ax[1], label='Public test')

sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_sum'].to_list()).reshape(-1),

            color="Green", ax=ax[1], label='Private test')

sns.kdeplot(np.array(train['bpps_nb'].to_list()).reshape(-1),

            color="Blue", ax=ax[2], label='Train')

sns.kdeplot(np.array(test[test['seq_length'] == 107]['bpps_nb'].to_list()).reshape(-1),

            color="Red", ax=ax[2], label='Public test')

sns.kdeplot(np.array(test[test['seq_length'] == 130]['bpps_nb'].to_list()).reshape(-1),

            color="Green", ax=ax[2], label='Private test')



ax[0].set_title('Distribution of bpps_max')

ax[1].set_title('Distribution of bpps_sum')

ax[2].set_title('Distribution of bpps_nb')

plt.tight_layout();
AUGMENT=True
aug_df = pd.read_csv('../input/openvaccineaugmented/aug_data_n2.csv')

print(aug_df.shape)

aug_df.head()
def aug_data(df):

    target_df = df.copy()

    new_df = aug_df[aug_df['id'].isin(target_df['id'])]

                         

    del target_df['structure']

    del target_df['predicted_loop_type']

    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')



    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])

    df['log_gamma'] = 100

    df['score'] = 1.0

    df = df.append(new_df[df.columns])

    return df
print(f"Samples in train before augmentation: {len(train)}")

print(f"Samples in test before augmentation: {len(test)}")



if AUGMENT:

    train = aug_data(train)

    test = aug_data(test)



print(f"Samples in train after augmentation: {len(train)}")

print(f"Samples in test after augmentation: {len(test)}")



print(f"Unique sequences in train: {len(train['sequence'].unique())}")

print(f"Unique sequences in test: {len(test['sequence'].unique())}")
DENOISE = False
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

    base_fea = np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )

    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]

    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]

    return np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea], 2)
if DENOISE:

    train = train[train['signal_to_noise'] > .25]
len(token2int)
# https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211

def rmse(y_actual, y_pred):

    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)

    return K.sqrt(mse)



def mcrmse(y_actual, y_pred, num_scored=len(target_cols)):

    score = 0

    for i in range(num_scored):

        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored

    return score
def gru_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.GRU(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer='orthogonal'))



def lstm_layer(hidden_dim, dropout):

    return tf.keras.layers.Bidirectional(

                                tf.keras.layers.LSTM(hidden_dim,

                                dropout=dropout,

                                return_sequences=True,

                                kernel_initializer='orthogonal'))



def build_model(rnn='gru', convolve=False, conv_dim=512, 

                dropout=.4, sp_dropout=.2, embed_dim=200,

                hidden_dim=256, layers=3,

                seq_len=107, pred_len=68):

    

###############################################

#### Inputs

###############################################



    inputs = tf.keras.layers.Input(shape=(seq_len, 5))

    categorical_feats = inputs[:, :, :3]

    numerical_feats = inputs[:, :, 3:]



    embed = tf.keras.layers.Embedding(input_dim=len(token2int),

                                      output_dim=embed_dim)(categorical_feats)

    reshaped = tf.reshape(

        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

    reshaped = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)

    hidden = tf.keras.layers.SpatialDropout1D(sp_dropout)(reshaped)

    

    if convolve:

        hidden = tf.keras.layers.Conv1D(conv_dim, 5, padding='same', activation=tf.keras.activations.swish)(hidden)



###############################################

#### RNN Layers

###############################################



    if rnn is 'gru':

        for _ in range(layers):

            hidden = gru_layer(hidden_dim, dropout)(hidden)

        

    elif rnn is 'lstm':

        for _ in range(layers):

            hidden = lstm_layer(hidden_dim, dropout)(hidden)



###############################################

#### Output

###############################################



    out = hidden[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation='linear')(out)

    

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()

    model.compile(optimizer=adam, loss=mcrmse)



    return model
test_model = build_model(rnn='gru')

test_model.summary()
def train_and_infer(rnn, STRATIFY=True, FOLDS=4, EPOCHS=50, BATCH_SIZE=64,

                    REPEATS=3, SEED=34, VERBOSE=2):



    #get test now for OOF 

    public_df = test.query("seq_length == 107").copy()

    private_df = test.query("seq_length == 130").copy()

    private_preds = np.zeros((private_df.shape[0], 130, 5))

    public_preds = np.zeros((public_df.shape[0], 107, 5))

    public_inputs = preprocess_inputs(public_df)

    private_inputs = preprocess_inputs(private_df)



    #to evaluate TTA effects/post processing

    holdouts = []

    holdout_preds = []

    

    #to view learning curves

    histories = []

    

    #put similar RNA in the same fold

    gkf = GroupKFold(n_splits=FOLDS)

    kf=KFold(n_splits=FOLDS, random_state=SEED)

    kmeans_model = KMeans(n_clusters=200, random_state=SEED).fit(preprocess_inputs(train)[:,:,0])

    train['cluster_id'] = kmeans_model.labels_



    for _ in range(REPEATS):

        

        for f, (train_index, val_index) in enumerate((gkf if STRATIFY else kf).split(train,

                train['reactivity'], train['cluster_id'] if STRATIFY else None)):



            #define training callbacks

            lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=8, 

                                                               factor=.1,

                                                               #min_lr=1e-5,

                                                               verbose=VERBOSE)

            save = tf.keras.callbacks.ModelCheckpoint(f'model-{f}.h5')



            #define sample weight function

            epsilon = .1

            sample_weighting = np.log1p(train.iloc[train_index]['signal_to_noise'] + epsilon) / 2



            #get train data

            trn = train.iloc[train_index]

            trn_ = preprocess_inputs(trn)

            trn_labs = np.array(trn[target_cols].values.tolist()).transpose((0, 2, 1))



            #get validation data

            val = train.iloc[val_index]

            val_all = preprocess_inputs(val)

            val = val[val.SN_filter == 1]

            val_ = preprocess_inputs(val)

            val_labs = np.array(val[target_cols].values.tolist()).transpose((0, 2, 1))



            #pre-build models for different sequence lengths

            model = build_model(rnn=rnn)

            model_short = build_model(rnn=rnn,seq_len=107, pred_len=107)

            model_long = build_model(rnn=rnn,seq_len=130, pred_len=130)



            #train model

            history = model.fit(

                trn_, trn_labs,

                validation_data = (val_, val_labs),

                batch_size=BATCH_SIZE,

                epochs=EPOCHS,

                sample_weight=sample_weighting,

                callbacks=[save, lr_callback],

                verbose=VERBOSE

            )



            histories.append(history)



            #load best models

            model.load_weights(f'model-{f}.h5')

            model_short.load_weights(f'model-{f}.h5')

            model_long.load_weights(f'model-{f}.h5')



            holdouts.append(train.iloc[val_index])

            holdout_preds.append(model.predict(val_all))



            public_preds += model_short.predict(public_inputs) / (FOLDS * REPEATS)

            private_preds += model_long.predict(private_inputs) / (FOLDS * REPEATS)

        

        del model, model_short, model_long

        

    return holdouts, holdout_preds, public_df, public_preds, private_df, private_preds, histories
gru_holdouts, gru_holdout_preds, public_df, gru_public_preds, private_df, gru_private_preds, gru_histories = train_and_infer(rnn='gru')
lstm_holdouts, lstm_holdout_preds, public_df, lstm_public_preds, private_df, lstm_private_preds, lstm_histories = train_and_infer(rnn='lstm')
def plot_learning_curves(results):



    fig, ax = plt.subplots(1, len(results['histories']), figsize = (20, 10))

    

    for i, result in enumerate(results['histories']):

        for history in result:

            ax[i].plot(history.history['loss'], color='C0')

            ax[i].plot(history.history['val_loss'], color='C1')

            ax[i].set_title(f"{results['models'][i]}")

            ax[i].set_ylabel('MCRMSE')

            ax[i].set_xlabel('Epoch')

            ax[i].legend(['train', 'validation'], loc = 'upper right')

            

results = {

            "models" : ['GRU', 'LSTM'],    

            "histories" : [gru_histories, lstm_histories],

            }
#https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model

def format_predictions(test_df, test_preds, val=False):

    preds = []

    

    for df, preds_ in zip(test_df, test_preds):

        for i, uid in enumerate(df['id']):

            single_pred = preds_[i]



            single_df = pd.DataFrame(single_pred, columns=target_cols)

            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            if val: single_df['SN_filter'] = df[df['id'] == uid].SN_filter.values[0]



            preds.append(single_df)

    return pd.concat(preds).groupby('id_seqpos').mean().reset_index() if AUGMENT else pd.concat(preds)
def get_error(preds):

    val = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)



    val_data = []

    for mol_id in val['id'].unique():

        sample_data = val.loc[val['id'] == mol_id]

        sample_seq_length = sample_data.seq_length.values[0]

        for i in range(68):

            sample_dict = {

                           'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),

                           'reactivity_gt' : sample_data['reactivity'].values[0][i],

                           'deg_Mg_pH10_gt' : sample_data['deg_Mg_pH10'].values[0][i],

                           'deg_Mg_50C_gt' : sample_data['deg_Mg_50C'].values[0][i],

                           }

            

            val_data.append(sample_dict)

            

    val_data = pd.DataFrame(val_data)

    val_data = val_data.merge(preds, on='id_seqpos')



    rmses = []

    mses = []

    

    for col in ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']:

        rmse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean() ** .5

        mse = ((val_data[col] - val_data[col+'_gt']) ** 2).mean()

        rmses.append(rmse)

        mses.append(mse)

        print(col, rmse, mse)

    print(np.mean(rmses), np.mean(mses))

    print('')
plot_learning_curves(results)
gru_val_preds = format_predictions(gru_holdouts, gru_holdout_preds, val=True)

lstm_val_preds = format_predictions(lstm_holdouts, lstm_holdout_preds, val=True)



print('-'*25); print('Unfiltered training results'); print('-'*25)

print('GRU training results'); print('')

get_error(gru_val_preds)

print('LSTM training results'); print('')

get_error(lstm_val_preds)

print('-'*25); print('SN_filter == 1 training results'); print('-'*25)

print('GRU training results'); print('')

get_error(gru_val_preds[gru_val_preds['SN_filter'] == 1])

print('LSTM training results'); print('')

get_error(lstm_val_preds[lstm_val_preds['SN_filter'] == 1])
gru_preds = [gru_public_preds, gru_private_preds]

lstm_preds = [gru_public_preds, gru_private_preds]

test_df = [public_df, private_df]

gru_preds = format_predictions(test_df, gru_preds)

lstm_preds = format_predictions(test_df, lstm_preds)
gru_weight = .5

lstm_weight = .5
blended_preds = pd.DataFrame()

blended_preds['id_seqpos'] = gru_preds['id_seqpos']

blended_preds['reactivity'] = gru_weight*gru_preds['reactivity'] + lstm_weight*lstm_preds['reactivity']

blended_preds['deg_Mg_pH10'] = gru_weight*gru_preds['deg_Mg_pH10'] + lstm_weight*lstm_preds['deg_Mg_pH10']

blended_preds['deg_pH10'] = gru_weight*gru_preds['deg_pH10'] + lstm_weight*lstm_preds['deg_pH10']

blended_preds['deg_Mg_50C'] = gru_weight*gru_preds['deg_Mg_50C'] + lstm_weight*lstm_preds['deg_Mg_50C']

blended_preds['deg_50C'] = gru_weight*gru_preds['deg_50C'] + lstm_weight*lstm_preds['deg_50C']
submission = sample_sub[['id_seqpos']].merge(blended_preds, on=['id_seqpos'])

submission.head()
submission.to_csv(f'submission_new.csv', index=False)

print('Submission saved')