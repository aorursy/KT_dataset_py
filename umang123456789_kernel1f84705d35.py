# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk

import nltk

import  re

import string

import collections

import spacy

import tqdm

import random

from spacy.util import minibatch, compounding

import os

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')



        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split()))

df_train.head(10)
#df_train = df_train[df_train['Num_words_text']>=4]
#df_train.head(10)
def save_model(output_dir, nlp, new_model_name):

    ''' This Function Saves model to 

    given output directory'''

    

    output_dir = f'../working/{output_dir}'

    if output_dir is not None:        

        if not os.path.exists(output_dir):

            os.makedirs(output_dir)

        nlp.meta["name"] = new_model_name

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)
def train(train_data, output_dir, n_iter=20, model=None):

    """Load the model, set up the pipeline and train the entity recognizer."""

    ""

    if model is not None:

        nlp = spacy.load(output_dir)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")

    

    # create the built-in pipeline components and add them to the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")

    

    # add labels

    for _, annotations in train_data:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        # sizes = compounding(1.0, 4.0, 1.001)

        # batch up the examples using spaCy's minibatch

        if model is None:

            nlp.begin_training()

        else:

            nlp.resume_training()





        for itn in tqdm.tqdm(range(n_iter)):

            random.shuffle(train_data)

            batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts,  # batch of texts

                            annotations,  # batch of annotations

                            drop=0.5,   # dropout - make it harder to memorise data

                            losses=losses, 

                            )

            print("Losses", losses)

    save_model(output_dir, nlp, 'st_ner')
def get_training_data(sentiment):

    '''

    Returns Trainong data in the format needed to train spacy NER

    '''

    train_data = []

    for index, row in df_train.iterrows():

        if row.sentiment == sentiment:

            selected_text = row.selected_text

            text = row.text

            start = text.find(selected_text)

            end = start + len(selected_text)

            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return train_data
def get_model_out_path(sentiment):

    '''

    Returns Model output path

    '''

    model_out_path = None

    if sentiment == 'positive':

        model_out_path = 'models/model_pos'

    elif sentiment == 'negative':

        model_out_path = 'models/model_neg'

    return model_out_path
sentiment = 'positive'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)

# For DEmo Purposes I have taken 3 iterations you can train the model as you want

train(train_data, model_path, n_iter=4, model=None)
sentiment = 'negative'



train_data = get_training_data(sentiment)

model_path = get_model_out_path(sentiment)



train(train_data, model_path, n_iter=4, model=None)
def predict_entities(text, model):

    doc = model(text)

    ent_array = []

    for ent in doc.ents:

        start = text.find(ent.text)

        end = start + len(ent.text)

        new_int = [start, end, ent.label_]

        if new_int not in ent_array:

            ent_array.append([start, end, ent.label_])

    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text

    return selected_text
selected_texts = []

MODELS_BASE_PATH = '../input/tse-spacy-model/models/'



if MODELS_BASE_PATH is not None:

    print("Loading Models  from ", MODELS_BASE_PATH)

    model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')

    model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')

        

    for index, row in df_test.iterrows():

        text = row.text

        output_str = ""

        if row.sentiment == 'neutral' or len(text.split()) <= 2:

            selected_texts.append(text)

        elif row.sentiment == 'positive':

            selected_texts.append(predict_entities(text, model_pos))

        else:

            selected_texts.append(predict_entities(text, model_neg))

        

df_test['selected_text'] = selected_texts
df_submission = pd.DataFrame()

df_submission['selected_text'] = df_test['selected_text']
df_submission.head(2)


my_submission = pd.DataFrame({'textID': df_test.textID, 'selected_text': df_submission.selected_text})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
my_submission.head(1)