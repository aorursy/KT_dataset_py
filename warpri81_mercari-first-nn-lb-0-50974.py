import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.cross_validation import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.cluster import KMeans, MiniBatchKMeans
train = pd.read_table("../input/train.tsv")

test = pd.read_table("../input/test.tsv")
train.head()
def fill_null(df):

    df['category_name'].fillna('?', inplace=True)

    df['brand_name'].fillna('?', inplace=True)

    df['item_description'].fillna('?', inplace=True)

    return df

train = fill_null(train)

test = fill_null(test)
category_name_encoder = LabelEncoder()

category_name_encoder.fit(np.concatenate([train['category_name'], test['category_name']]))

train['category_id'] = category_name_encoder.transform(train['category_name'])

test['category_id'] = category_name_encoder.transform(test['category_name'])
brand_name_encoder = LabelEncoder()

brand_name_encoder.fit(np.concatenate([train['brand_name'], test['brand_name']]))

train['brand_id'] = brand_name_encoder.transform(train['brand_name'])

test['brand_id'] = brand_name_encoder.transform(test['brand_name'])
%%time



DESCRIPTION_FEATURES = 10000



description_vectorizer = TfidfVectorizer(max_df=0.5, max_features=DESCRIPTION_FEATURES,

                                         min_df=10, stop_words='english', lowercase=True,

                                         ngram_range=(1,3))

train_description_vectors = description_vectorizer.fit_transform(train['item_description'])

test_description_vectors = description_vectorizer.transform(test['item_description'])
print(len(description_vectorizer.get_feature_names()))

print(description_vectorizer.get_feature_names()[:500])
%%time



DESCRIPTION_EMBEDDING_SIZE = 200



description_svd = TruncatedSVD(DESCRIPTION_EMBEDDING_SIZE)

description_normalizer = Normalizer(copy=False)

description_lsa = make_pipeline(description_svd, description_normalizer)

train_description_embeddings = description_lsa.fit_transform(train_description_vectors)

test_description_embeddings = description_lsa.transform(test_description_vectors)
train['description_length'] = train['item_description'].apply(len)

test['description_length'] = test['item_description'].apply(len)

description_length_scaler = MinMaxScaler(feature_range=(-1, 1))

description_length_scaler.fit(np.concatenate([train['description_length'], test['description_length']]).reshape(-1, 1))

train['description_length_scaled'] = description_length_scaler.transform(train['description_length'].reshape(-1, 1))

test['description_length_scaled'] = description_length_scaler.transform(test['description_length'].reshape(-1, 1))
price_scaler = MinMaxScaler(feature_range=(-1, 1))

train_price_log = np.log(train['price'].reshape(-1, 1) + 1)

train['price_scaled'] = price_scaler.fit_transform(train_price_log.reshape(-1, 1))
ITEM_CONDITION_COUNT = len(np.unique(np.concatenate([train['item_condition_id'], test['item_condition_id']])))
(

    train_train, train_val,

    train_train_desc, train_val_desc,

) = train_test_split(

    train,

    train_description_embeddings,

    random_state=777, train_size=0.9)
from keras.models import Model

from keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout

from keras import backend as K
def rmsle(y, y_pred):

    y = np.array(y).reshape(-1)

    y_pred = np.array(y_pred).reshape(-1)

    assert len(y) == len(y_pred)

    to_sum = (np.log(y_pred + 1) - np.log(y + 1)) ** 2

    return to_sum.mean() ** 0.5

#Source: https://www.kaggle.com/marknagelberg/rmsle-function



def rmsle_cust(y_true, y_pred):

    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)

    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)

    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))
brand_id = Input(shape=(1,))

category_id = Input(shape=(1,))

item_condition = Input(shape=(1,))

shipping = Input(shape=(1,))

description_embedding = Input(shape=(DESCRIPTION_EMBEDDING_SIZE,))

description_length = Input(shape=(1,))



brand_embedding = Embedding(len(brand_name_encoder.classes_), 50)(brand_id)

category_embedding = Embedding(len(category_name_encoder.classes_), 20)(category_id)



all_inputs = Concatenate()([

    Flatten()(brand_embedding),

    Flatten()(category_embedding),

    item_condition,

    description_embedding,

    description_length,

    shipping,

])



layer_1 = Dense(512, activation='relu')(all_inputs)

layer_2 = Dense(512, activation='relu')(layer_1)

layer_3 = Dense(256, activation='relu')(layer_2)

output = Dense(1)(layer_3)



model = Model(

    [

        brand_id, category_id,

        item_condition, shipping,

        description_embedding, description_length,

    ],

    output

)



model.compile(loss='mse', optimizer='adam', metrics=['mse'])

#model.compile(loss='mse', optimizer='adam', metrics=['mse', rmsle_cust])
def prepare_input(df, description_embeddings):

    return [

        df['brand_id'],

        df['category_id'],

        (df['item_condition_id'] - 1) / 5 - 0.5,

        df['shipping'],

        description_embeddings, df['description_length_scaled'],

    ]
X_train = prepare_input(train_train, train_train_desc)

X_val = prepare_input(train_val, train_val_desc)
model.fit(X_train, train_train['price_scaled'].values,

         validation_data=(X_val, train_val['price_scaled'].values),

         epochs=5, batch_size=512)
y_val = model.predict(X_val)

pred_val = price_scaler.inverse_transform(y_val)

pred_val = np.exp(pred_val) - 1

pred_val = pred_val.clip(0, None)
rmsle(train_val['price'], pred_val)
X_test = prepare_input(test, test_description_embeddings)

y_test = model.predict(X_test)

pred_test = price_scaler.inverse_transform(y_test)

pred_test = np.exp(pred_test) - 1

pred_test = pred_test.clip(0, None)
submission = pd.DataFrame({'test_id': test['test_id'], 'price': pred_test.reshape(-1)})

submission.to_csv('mercare_simple_nn.csv', index=False)