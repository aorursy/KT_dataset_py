# Installations

!pip install -q nltk

!pip install -q WordCloud

!pip install -q plotly

!pip install -q transformers

!pip install -q twython
import pandas as pd

import numpy as np

import os



#sklearn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



#NLP

import string

import re    #for regex

import nltk

from nltk.corpus import stopwords

from wordcloud  import WordCloud, STOPWORDS

from nltk import pos_tag

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

# Tweet tokenizer does not split at apostophes which is what we want

from nltk.tokenize import TweetTokenizer 

from nltk.sentiment.vader import SentimentIntensityAnalyzer





nltk.download('stopwords')

nltk.download('punkt')

nltk.download('vader_lexicon')



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



#Modelling



import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import TFAutoModel, AutoTokenizer

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors



from kaggle_datasets import KaggleDatasets
os.listdir('/kaggle/working')
# Importing all the required datasets

dir_path = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'



train_data_1 = pd.read_csv(dir_path + "/jigsaw-toxic-comment-train.csv")

train_data_2 = pd.read_csv(dir_path + "/jigsaw-unintended-bias-train.csv")

validation_data = pd.read_csv(dir_path + "/validation.csv")

test_data = pd.read_csv(dir_path + "/test.csv")
train_data_1.shape
train_data_1.head()
train_data_2.shape
train_data_2.head()
validation_data.shape
validation_data.head()
test_data.shape
test_data.head()
# In the second dataset, the toxicity is not 1 or 0 but instead a probability, we will round it to convert to a 1/0 column

train_data_2.toxic = train_data_2.toxic.round().astype(int)



# We combined the entire training set 1 with all the toxic comments of training set 2 and 200k non-toxic comments from set 2

train_data = pd.concat([

                train_data_1[['comment_text','toxic']]

                , train_data_2[['comment_text','toxic']].query('toxic == 1')

                , train_data_2[['comment_text', 'toxic']].query('toxic == 0').sample(n = 200000, random_state = 1993)

                ])

train_data.shape
sns.countplot(train_data.toxic)
sns.countplot(validation_data.toxic)
word_tokenize(train_data.comment_text[1])

#count_vectorizer = CountVectorizer(stop_words = 'english', ngram_range=(1,3))
#trial = count_vectorizer.fit_transform(train_data.comment_text)
#trial.shape
# Splitting up the comment into single words

text_words = word_tokenize(train_data.comment_text[1])

# Converting to lower case

text_words = [word.lower() for word in text_words]
modified_stopwords = stopwords.words('english')

modified_stopwords.remove('not')

#Removing stopwords and sumbols

text_words = [word for  word in text_words if not word in modified_stopwords and word.isalpha()]

len(text_words)
text_words
train_sample = train_data.sample(n = 10000, random_state = 1993)

train_sample = train_sample.reset_index(drop = True)
# Filtering comment text column, removing newline characters and filtering out unexpected data types from the column

def nan_filter(x):

    if type(x) == str:

        return (x.replace("\n", "")).lower()

    else:

        return ""



nontoxic_text = ' '.join([nan_filter(comment) for comment in train_sample.query('toxic==0')['comment_text']])

toxic_text = ' '.join([nan_filter(comment) for comment in train_sample.query('toxic == 1')['comment_text']])
wordcloud = WordCloud(max_font_size=300

                      , background_color='white'

                      , stopwords = modified_stopwords

                      , collocations=True

                      , max_words = 100

                      , width=1200

                      , height=1000).generate(nontoxic_text)



fig = px.imshow(wordcloud)



fig.update_layout(title_text='Non-Toxic Word Cloud(with bigrams)')
wordcloud = WordCloud(max_font_size=300

                      , background_color='white'

                      , stopwords = modified_stopwords

                      , collocations=False

                      , max_words = 100

                      , width=1200

                      , height=1000).generate(nontoxic_text)



fig = px.imshow(wordcloud)



fig.update_layout(title_text='Non-Toxic Word Cloud(unigrams)')
wordcloud = WordCloud(max_font_size=300

                      , background_color='white'

                      , stopwords = modified_stopwords

                      , collocations=True

                      , width=1200

                      , max_words = 100

                      , height=1000).generate(toxic_text)



fig = px.imshow(wordcloud)



fig.update_layout(title_text='Toxic Word Cloud(with bigrams)')
wordcloud = WordCloud(max_font_size=300

                      , background_color='white'

                      , stopwords = modified_stopwords

                      , collocations=False

                      , max_words = 100

                      , width=1200

                      , height=1000).generate(toxic_text)



fig = px.imshow(wordcloud)



fig.update_layout(title_text='Toxic Word Cloud(unigrams)')
train_sample.iloc[1,0]
# Comment size visualizations



def text_len(x):

    if type(x) is str:

        return len(x.split())

    else:

        return 0

    



train_sample['comment_size'] = train_sample.comment_text.apply(text_len)



toxic_text_lengths = train_sample.query('toxic == 1 and comment_size < 200') ['comment_size'].sample(frac = 1, random_state = 1993)

nontoxic_text_lengths = train_sample.query('toxic == 0 and comment_size < 200')['comment_size'].sample(frac = 1, random_state = 1993)

plt.figure(figsize=(13,5))

ax = sns.distplot(toxic_text_lengths)

plt.title('Toxic Comment Lengths')

plt.xlabel('Comment Length')

plt.xticks(np.arange(0,210,10))

plt.yticks(np.arange(0,0.025,0.0025));
plt.figure(figsize=(13,5))

ax = sns.distplot(nontoxic_text_lengths)

plt.title('Non-Toxic Comment Lengths')

plt.xlabel('Comment Length')

plt.xticks(np.arange(0,210,10))

plt.yticks(np.arange(0,0.025,0.0025));
def sentiment(x):

    if type(x) is str:

        return SIA.polarity_scores(x)

    else:

        return 1000



SIA = SentimentIntensityAnalyzer()

train_sample['polarity'] = train_sample.comment_text.apply(sentiment)

# Vader outputs 4 scores, Negative, Neutral, Positive and Compound

train_sample.query('toxic == 0').head(10)
train_sample.query('toxic==1').head(10)
# This comment has a negative score of 0 despite clearly being toxic.

train_sample.comment_text[22]
train_sample['negativity'] = train_sample.polarity.apply(lambda x: x['neg'])

train_sample['positivity'] = train_sample.polarity.apply(lambda x: x['pos'])
nontoxic_negativity = train_sample.query('toxic == 0').sample(frac = 1, random_state = 1993)['negativity']

toxic_negativity = train_sample.query('toxic == 1').sample(frac = 1, random_state = 1993)['negativity']



plot = ff.create_distplot([nontoxic_negativity, toxic_negativity]

                           , group_labels = ['Non-Toxic', 'Toxic']

                           , colors = ['Green', 'Red']

                           , show_hist= False)

plot.update_layout(title_text = 'Negativity vs Toxicity'

                   , xaxis_title = 'Negativity'

                   , xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 0.1))



plot.show()
nontoxic_positivity = train_sample.query('toxic == 0').sample(frac = 1, random_state = 1993)['positivity']

toxic_positivity = train_sample.query('toxic == 1').sample(frac = 1, random_state = 1993)['positivity']



plot = ff.create_distplot([nontoxic_positivity, toxic_positivity]

                          , group_labels=['Non-Toxic', 'Toxic']

                          , colors = ['Green', 'Red']

                          , show_hist= False)



plot.update_layout( title_text = 'Positivity vs Toxicity'

                    , xaxis_title = 'Positivity'

                    , xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 0.1))

plot.show()
roberta_string = 'jplu/tf-xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained(roberta_string)
print('Comment 1: - ' + '\n\n' + 

      train_sample.comment_text.values[0])
print('Comment 2: - ' + '\n\n' + 

      train_sample.comment_text.values[1])
sample_encoded = tokenizer.batch_encode_plus(train_sample.comment_text.values[0:2]

                                    , return_attention_masks=False

                                   , return_token_type_ids=False

                                   , pad_to_max_length=True

                                   , max_length = 512)

sample_encoded
def encode(text, max_len = 512):

    encoded_dict = tokenizer.batch_encode_plus(text

                               , return_attention_masks=False

                               , return_token_type_ids=False

                               , pad_to_max_length=True

                               , max_length = max_len)

    return np.array(encoded_dict['input_ids'])



MAX_LEN = 192
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 2

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



# We split up the datasets into X and y as we will train the model to predict target y's using feature sets X's

X_train = encode(train_data.comment_text.values, MAX_LEN)

X_valid = encode(validation_data.comment_text.values, MAX_LEN)

X_test = encode(test_data.content.values, MAX_LEN)



# target datasets don't need to be encoded since these are toxicity flag values of 0 and 1 for each comment

y_train = train_data.toxic.values

y_valid = validation_data.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((X_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(X_test)

    .batch(BATCH_SIZE)

)
%%time

with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained(roberta_string)

    

    input_word_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer_layer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

model.summary()
n_steps = X_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = X_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS

)
submission = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

submission['toxic'] = model.predict(test_dataset, verbose = 1)

submission.to_csv('/kaggle/working/submission.csv', index = False)