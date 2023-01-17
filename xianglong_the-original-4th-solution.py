import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import time

from sklearn.metrics import accuracy_score

from kaggle.competitions import twosigmanews



env = twosigmanews.make_env()

(market_train, _) = env.get_training_data()
pd.__version__
cat_cols = ['assetCode']

num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',

                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',

                    'returnsOpenPrevMktres10']



from sklearn.model_selection import train_test_split

#train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.95)

#train_indices, val_indices = train_test_split(market_train.index.values,test_size=0.25)

train_indices, val_indices = train_test_split(market_train.index.values,test_size=0)
# handle cat features

def encode(encoder, x):

    len_encoder = len(encoder)

    try:

        ind = encoder[x]

    except KeyError:

        ind = len_encoder

    return ind



encoders = [{} for cat in cat_cols]



for i, cat in enumerate(cat_cols):

    print('encoding %s'%cat, end='\n')

    encoders[i] = {l:ind for ind, l in enumerate(market_train[cat].unique())}

    market_train[cat] = market_train[cat].apply(lambda x:encode(encoders[i],x))

    

embed_sizes = [len(encoder) + 1 for encoder in encoders]
# handle num features

from sklearn.preprocessing import StandardScaler

market_train[num_cols] = market_train[num_cols].fillna(0)

scaler = StandardScaler()

market_train[num_cols] = scaler.fit_transform(market_train[num_cols])
# Define NN

from keras.models import Model

from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization,LeakyReLU,Dropout,Average

from keras.losses import binary_crossentropy, mse, mae



cat_inputs = []

for cat in cat_cols:

    cat_inputs.append(Input(shape=[1], name=cat))

    

cat_embs = []

for i,cat in enumerate(cat_cols):

    cat_embs.append(Embedding(embed_sizes[i], 10)(cat_inputs[i]))



#cat_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in cat_embs])

cat_logits = Flatten()(cat_embs[0])

cat_logits = Dense(32)(cat_logits)

cat_logits = LeakyReLU(0.1)(cat_logits)

cat_logits = Dropout(0.5)(cat_logits)

num_input = Input(shape = (len(num_cols),), name='num')

num_logits = num_input

num_logits = BatchNormalization()(num_logits)

num_logits = Dense(32)(num_logits)

num_logits = LeakyReLU(0.1)(num_logits)

num_logits = Dropout(0.5)(num_logits)

all_logits = Concatenate()([num_logits, cat_logits])

logits = Dense(128, activation='relu')(all_logits)

logits = Dropout(0.5)(logits)

logits = Dense(64, activation='relu')(logits)

logits = Dropout(0.5)(logits)

out = Dense(1, activation='tanh')(logits)





model = Model(inputs = cat_inputs+[num_input],outputs=out)

#model.summary()

model.compile(optimizer='adam', loss=mae)
def get_input(market_train, indices):

    X_num = market_train.loc[indices, num_cols].values

    X = {'num':X_num}

    for cat in cat_cols:

        X[cat] = market_train.loc[indices, cat].values

    market_train.loc[indices, 'returnsOpenNextMktres10'] = market_train.loc[indices, 'returnsOpenNextMktres10'].apply(lambda x: 0 if x < -0.3 or x > 0.3else x)

    market_train.loc[indices, 'returnsOpenNextMktres10'] = market_train.loc[indices, 'returnsOpenNextMktres10'].apply(lambda x: -1 if x < 0 else x)

    market_train.loc[indices, 'returnsOpenNextMktres10'] = market_train.loc[indices, 'returnsOpenNextMktres10'].apply(lambda x: 1 if x > 0 else x)

    y = market_train.loc[indices, 'returnsOpenNextMktres10']

    #y = market_train.loc[indices, 'returnsOpenNextMktres10'].apply(lambda x: 0 if x > 0.3 or x < -0.3 else x)

    #y = y.apply(lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))/0.3

    y = y.values

    r = market_train.loc[indices,'returnsOpenNextMktres10'].values

    u = market_train.loc[indices, 'universe']

    d = market_train.loc[indices, 'time'].dt.date

    return X,y,r,u,d



# r, u and d are used to calculate the scoring metric

X_train,y_train,r_train,u_train,d_train = get_input(market_train, train_indices)

X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train, val_indices)
from keras.callbacks import EarlyStopping, ModelCheckpoint

check_point = ModelCheckpoint('model.hdf5', verbose=True, save_best_only=True)

early_stop= EarlyStopping(patience=2, verbose=True)

model.fit(X_train, y_train,

         validation_data = (X_valid, y_valid),

         epochs = 4,

         verbose = True,

         callbacks=[check_point, early_stop])
# Prediction

days = env.get_prediction_days()



n_days = 0

for (market_obs_df, news_obs_df, predictions_template_df) in days:

    n_days += 1

    if n_days % 10 == 0:

        print(n_days,end='\n')

    # num features

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)

    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])

    X_num_test = market_obs_df[num_cols].values

    X_test = {'num':X_num_test}

    test_cat_cols = []

    for i in range(len(cat_cols)):

        market_obs_df[cat_cols[i]+'_encoded'] = market_obs_df[cat_cols[i]].astype(str).apply(lambda x: encode(encoders[i], x))

        X_test[cat_cols[i]] = market_obs_df[cat_cols[i]+'_encoded']

    market_prediction = model.predict(X_test)

    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'], 'confidence':market_prediction.reshape(-1)})

    predictions_template_df = predictions_template_df.merge(preds, how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})

    env.predict(predictions_template_df)

env.write_submission_file()

print('Done!')