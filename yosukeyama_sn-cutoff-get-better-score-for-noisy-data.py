import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf
# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    inputs = L.Input(shape=(seq_len, 3))

    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    hidden = gru_layer(hidden_dim, dropout)(reshaped)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    # Since we are only making predictions on the first part of each sequence, we have
    # to truncate it
    truncated = hidden[:, :pred_len]
    
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss='mse')
    
    return model
def pandas_list_to_array(df):
    """
    Inputs:
        df: dataframe of shape (x, y), containing list of length l
    Return:
        np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(
            df.values
            .tolist()
        ),
        (0, 2, 1)
    )
def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )
data_dir = '/kaggle/input/stanford-covid-vaccine/'
train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
sample_df = pd.read_csv(data_dir + 'sample_submission.csv')
train_sn05 = train[train.signal_to_noise > 0.5]
train_sn10 = train[train.signal_to_noise > 1]
train_sn15 = train[train.signal_to_noise > 1.5]
train_sn20 = train[train.signal_to_noise > 2]
train_sn25 = train[train.signal_to_noise > 2.5]
train_sn30 = train[train.signal_to_noise > 3]
train_sn35 = train[train.signal_to_noise > 3.5]
train_sn40 = train[train.signal_to_noise > 4.0]

# train_noisy is used for noisy test data
train_noisy = train[train.signal_to_noise < 0.5]

for i, train_sn in enumerate([train_sn05, train_sn10, train_sn15, train_sn20, train_sn25, train_sn30, train_sn35, train_sn40]):    
    print(f'the length of train_sn{(i+1)*5} is {len(train_sn)}')
    
print(f'the length of train_noisy is {len(train_noisy)}')
# We will use this dictionary to map each character to an integer
# so that it can be used as an input in keras
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])

train_inputs_05 = preprocess_inputs(train_sn05, token2int)
train_inputs_10 = preprocess_inputs(train_sn10, token2int)
train_inputs_15 = preprocess_inputs(train_sn15, token2int)
train_inputs_20 = preprocess_inputs(train_sn20, token2int)
train_inputs_25 = preprocess_inputs(train_sn25, token2int)
train_inputs_30 = preprocess_inputs(train_sn30, token2int)
train_inputs_35 = preprocess_inputs(train_sn35, token2int)
train_inputs_40 = preprocess_inputs(train_sn40, token2int)

train_labels_05 = pandas_list_to_array(train_sn05[pred_cols])
train_labels_10 = pandas_list_to_array(train_sn10[pred_cols])
train_labels_15 = pandas_list_to_array(train_sn15[pred_cols])
train_labels_20 = pandas_list_to_array(train_sn20[pred_cols])
train_labels_25 = pandas_list_to_array(train_sn25[pred_cols])
train_labels_30 = pandas_list_to_array(train_sn30[pred_cols])
train_labels_35 = pandas_list_to_array(train_sn35[pred_cols])
train_labels_40 = pandas_list_to_array(train_sn40[pred_cols])

test_inputs = preprocess_inputs(train_noisy, token2int)
test_labels = pandas_list_to_array(train_noisy[pred_cols])
model = build_model(embed_size=len(token2int))
model.summary()
train_sn_list = [(train_inputs_05, train_labels_05,),
                (train_inputs_10, train_labels_10,),
                (train_inputs_15, train_labels_15,),
                (train_inputs_20, train_labels_20,),
                (train_inputs_25, train_labels_25,),
                (train_inputs_30, train_labels_30,),
                (train_inputs_35, train_labels_35,),
                (train_inputs_40, train_labels_40,)]

loss_list = []

for i, (train_inputs, train_labels) in enumerate(train_sn_list):
    print('=========================================')
    print(f'SN={(i+1)*0.5}')
    model = build_model(embed_size=len(token2int))
    history = model.fit(
    train_inputs, train_labels, 
    batch_size=64,
    epochs=60,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint(f'model_sn{(i+1)*5}.h5')
    ],
    validation_split=0.2
)
    loss_list.append(min(history.history['val_loss']))
# minimum val_loss of each model
for i, loss in enumerate(loss_list):
    print(f'SN={(i+1)*0.5}: val_loss = {loss}')
# loss for noisy test data
for i in range(8):
    model.load_weights(f'./model_sn{(i+1)*5}.h5')
    results = model.evaluate(test_inputs, test_labels, batch_size=8, verbose=0)
    print(f'SN={(i+1)*0.5}: test_loss = {results}')
