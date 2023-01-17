# Basic packages

import pandas as pd 

import numpy as np

import re

import collections

import matplotlib.pyplot as plt



# Modeling, selection, and evaluation

from fastai.text import *

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.utils.multiclass import unique_labels



%matplotlib inline 
# Read file into dataframe

pd.set_option('display.max_colwidth', -1)

df = pd.read_csv('../input/Tweets.csv')

df = df.reindex(np.random.permutation(df.index))  

df.head()
df['airline_sentiment'].value_counts().plot(kind='bar')
df['airline'].value_counts().plot(kind='bar')
df.groupby(['airline', 'airline_sentiment']).size().unstack().plot(kind='bar', stacked=True)
df['tweet_length'] = df['text'].apply(len)

df.groupby(['tweet_length', 'airline_sentiment']).size().unstack().plot(kind='line', stacked=False)
df[['tweet_length', 'airline_sentiment', 'airline_sentiment_confidence']].groupby(['tweet_length', 'airline_sentiment']).mean().unstack().plot(kind='line', stacked=False)
df[['tweet_length', 'airline_sentiment', 'airline_sentiment_confidence']].groupby(['tweet_length', 'airline_sentiment']).median().unstack().plot(kind='line', stacked=False)
test_percentage = 0.1

df.sort_index(inplace=True)

cutoff = int(test_percentage * df.shape[0])

df[['airline_sentiment', 'text']][:cutoff].to_csv('Tweets_filtered_test.csv', index=False, encoding='utf-8')

df[['airline_sentiment', 'text']][cutoff:].to_csv('Tweets_filtered_train.csv', index=False, encoding='utf-8')

df[['text']][cutoff:].to_csv('Tweets_text_only_train.csv', index=False, encoding='utf-8')
data = TextClasDataBunch.from_csv('.', 'Tweets_filtered_train.csv')

data.show_batch()
data.vocab.itos[:10]
print(data.train_ds[0][0])

print(data.train_ds[1][0])

print(data.train_ds[2][0])
print(data.train_ds[0][0].data[:10])

print(data.train_ds[1][0].data[:10])

print(data.train_ds[2][0].data[:10])
bs = 24

seed = 333
data_lm = (TextList.from_csv('.', 'Tweets_text_only_train.csv')

            .random_split_by_pct(0.1, seed = seed)

           #We randomly split and keep 10% for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=bs))

data_lm.save('data_lm.pkl')
# data_lm = load_data(path, 'data_lm.pkl', bs=bs)

data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=15)
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')

# learn.load('fit_head')
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')
learn.save_encoder('fine_tuned_enc')
data_clas = (TextList.from_csv('.', 'Tweets_filtered_train.csv', cols = 'text')               

             .random_split_by_pct(0.1, seed = seed)

             .label_from_df(cols=0)

             .databunch(bs=bs))

data_clas.save('data_clas.pkl')

data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)

learn.load_encoder('fine_tuned_enc')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.save('first')

# learn.load('first)
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.save('second')

# learn.load('second')
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
test_df = pd.read_csv("Tweets_filtered_test.csv", encoding="utf-8")

test_df['airline_sentiment'].value_counts().plot(kind='bar')
test_df['pred_sentiment'] = test_df['text'].apply(lambda row: str(learn.predict(row)[0]))

print("Test Accuracy: ", accuracy_score(test_df['airline_sentiment'], test_df['pred_sentiment']))
test_df[:20]
# Confusion matrix plotting adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    #classes = classes[unique_labels(y_true, y_pred)]



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
plot_confusion_matrix(test_df['airline_sentiment'], test_df['pred_sentiment'], classes=['negative', 'neutral', 'positive'], title='Airline sentiment confusion matrix')

# confusion_matrix(test_df['airline_sentiment'], test_df['pred_sentiment'], labels=['positive', 'neutral', 'negative'])

plt.show()
test_df.loc[(test_df['airline_sentiment'] == 'positive') & (test_df['pred_sentiment'] == 'negative')]