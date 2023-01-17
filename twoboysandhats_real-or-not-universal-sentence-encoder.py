import random



import numpy as np

import pandas as pd



import tensorflow as tf

import tensorflow_hub as hub



import keras

from keras.layers import Input, Dense, LeakyReLU, Dropout, Softmax

from keras.models import Model

from keras.utils import to_categorical



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



random.seed(42)
df_train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head()
# What's the distribution of real tweets to fake tweets in the training data?

target_distribution = df_train['target'].value_counts()

sns.barplot(target_distribution.index, target_distribution)
# What are the 10 most popular keywords?

keywords_ranked = df_train['keyword'].value_counts()[:10]



g = sns.barplot(keywords_ranked.index, keywords_ranked)

g.set_xticklabels(g.get_xticklabels(), rotation=90)
# What are the 10 most popular locations?

locations_ranked = df_train['location'].value_counts()[:10]



g = sns.barplot(locations_ranked.index, locations_ranked)

g.set_xticklabels(g.get_xticklabels(), rotation=90)
# How many tweets actually contain keyword or location info?

has_keyword = df_train['keyword'].isna().value_counts()

has_location = df_train['location'].isna().value_counts()



fig, axs = plt.subplots(ncols=2, figsize = (12, 5))

sns.barplot(has_keyword.index, has_keyword, ax = axs[0])

sns.barplot(has_location.index, has_location, ax = axs[1])
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def preprocess_text(text):

    text = text.replace('#', '') # remove hashtag symbols

    return text



def preprocess_data(df):

    df = df.drop(['location', 'keyword'], axis = 1) # ignore location and keyword for now

    df['text'] = df['text'].apply(preprocess_text)

    

    return df



def embed_data(df, include_y = True):

    x = embed(df['text']).numpy()

    

    if include_y:

        y = to_categorical(df['target'].to_numpy(), num_classes = 2)

        return (x, y)

    

    return x

    

df_train_preprocessed = preprocess_data(df_train)

x, y = embed_data(df_train_preprocessed)
x_train, x_test, y_train, y_test = train_test_split(

    x,

    y,

    test_size=0.33,

    random_state=42

)



# apply StandardScaler and PCA.

# Only fit it on the train data though - the test data is meant to never

# have been seen before by our model



scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)



pca = PCA(n_components = 128)

x_train = pca.fit_transform(x_train)

x_test = pca.transform(x_test)
print (f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')

print (f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
DROPOUT = .2



in_1 = Input(shape = (128,))



dense_1 = Dense(500)(in_1)

act_1 = LeakyReLU()(dense_1)

drop_1 = Dropout(DROPOUT)(act_1)



dense_2 = Dense(100)(drop_1)

act_2 = LeakyReLU()(dense_2)

drop_2 = Dropout(DROPOUT)(act_2)



dense_3 = Dense(2)(drop_2)

act_3 = Softmax()(dense_3)



model = Model(in_1, act_3)



model.compile("adam", loss = "categorical_crossentropy", metrics = ["acc"])



model.summary()
model.fit(

    x_train,

    y_train,

    batch_size = 128,

    epochs = 16,

    validation_data = (x_test, y_test)

)
pred = model.predict(x_test).argmax(axis = 1)

print (classification_report(y_test.argmax(axis = 1), pred))
df_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df_test_preprocessed = preprocess_data(df_test)

x_test = embed_data(df_test_preprocessed, include_y = False)

x_test = pca.transform(scaler.transform(x_test))
pred = model.predict(x_test).argmax(axis = 1)

df_submission = pd.DataFrame({'id': df_test_preprocessed['id'], 'target': pd.Series(pred)})
df_submission.to_csv('submission.csv', index = False)