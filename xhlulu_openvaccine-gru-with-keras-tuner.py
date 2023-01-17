import pickle

import os



import pandas as pd

import numpy as np

import json

import tensorflow.keras.layers as L

import tensorflow as tf

import plotly.express as px

import kerastuner as kt

from kerastuner.tuners import RandomSearch

from kerastuner import HyperModel

from sklearn.model_selection import train_test_split
os.makedirs('/kaggle/tmp/', exist_ok=True)
# This will tell us the columns we are predicting

pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
def parse_trial_state(trial):

    state = trial.get_state()

    out = {}

    out['best_step'] = state['best_step']

    out['trial_id'] = state['trial_id']

    out['score'] = state['score']

    out.update(state['hyperparameters']['values'])

    

    return out
class HyperGRU(HyperModel):



    def __init__(self, embed_size, seq_len=107, pred_len=68):

        self.embed_size = embed_size

        self.seq_len = seq_len

        self.pred_len = pred_len



    def MCRMSE(self, y_true, y_pred):

        colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

        return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)



    def gru_layer(self, hidden_dim, dropout):

        return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

    

    def build(self, hp):

        # Hyperparameters we will explore

        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        dropout = hp.Choice('dropout', values=[0., 0.1, 0.25, 0.5])

        embed_dim = hp.Int('embed_dim', min_value=50, max_value=150, step=25)

        hidden_dim = hp.Int('hidden_dim', min_value=32, max_value=256, step=32)

        n_layers = hp.Int('n_layers', 2, 3)

        

        inputs = L.Input(shape=(self.seq_len, 3))



        embed = L.Embedding(input_dim=self.embed_size, output_dim=embed_dim)(inputs)

        hidden = tf.reshape(

            embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])

        )

        

        for i in range(n_layers):

            hidden = self.gru_layer(hidden_dim, dropout)(hidden)



        # Since we are only making predictions on the first part of each sequence, we have

        # to truncate it

        truncated = hidden[:, :self.pred_len]



        out = L.Dense(5, activation='linear')(truncated)



        model = tf.keras.Model(inputs=inputs, outputs=out)



        model.compile(tf.keras.optimizers.Adam(lr), loss=self.MCRMSE)



        return model
def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):

    return np.transpose(

        np.array(

            df[cols]

            .applymap(lambda seq: [token2int[x] for x in seq])

            .values

            .tolist()

        ),

        (0, 2, 1)

    )
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)

sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
train_inputs = preprocess_inputs(train, token2int)

train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
# Is seed 34 a magic number? We shall find out

x_train, x_val, y_train, y_val = train_test_split(

    train_inputs, train_labels, test_size=.1, random_state=34

)
hypermodel = HyperGRU(embed_size=len(token2int))
tuner = kt.tuners.RandomSearch(

    hypermodel,

    objective='val_loss',

    max_trials=35,

    executions_per_trial=3,

    seed=2020,

    directory='/kaggle/tmp/',

    project_name='open_vaccine'

)



tuner.search(

    x_train, y_train,

    batch_size=64,

    epochs=100,

    verbose=0,

    validation_data=(x_val, y_val),

    callbacks=[

        tf.keras.callbacks.ReduceLROnPlateau(patience=4),

        tf.keras.callbacks.EarlyStopping(patience=8)

    ]

)
trials_df = pd.DataFrame([

    parse_trial_state(t) for t in tuner.oracle.trials.values()

])



trials_df.to_csv('trials_table.csv', index=False)



trials_df
best_hp = tuner.get_best_hyperparameters(1)[0]

best_hp.get_config()['values']
pickle.dump(best_hp, open('best_hp.pickle', 'wb'))



best_model = tuner.get_best_models(1)[0]

best_model.save('best_model.h5')
public_df = test.query("seq_length == 107").copy()

private_df = test.query("seq_length == 130").copy()



public_inputs = preprocess_inputs(public_df, token2int)

private_inputs = preprocess_inputs(private_df, token2int)
# Caveat: The prediction format requires the output to be the same length as the input,

# although it's not the case for the training data.

model_short = HyperGRU(seq_len=107, pred_len=107, embed_size=len(token2int)).build(best_hp)

model_long = HyperGRU(seq_len=130, pred_len=130, embed_size=len(token2int)).build(best_hp)



model_short.load_weights('best_model.h5')

model_long.load_weights('best_model.h5')
public_preds = model_short.predict(public_inputs)

private_preds = model_long.predict(private_inputs)
print(public_preds.shape, private_preds.shape)
preds_ls = []



for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

    for i, uid in enumerate(df.id):

        single_pred = preds[i]



        single_df = pd.DataFrame(single_pred, columns=pred_cols)

        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



        preds_ls.append(single_df)



preds_df = pd.concat(preds_ls)
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

submission.to_csv('submission.csv', index=False)