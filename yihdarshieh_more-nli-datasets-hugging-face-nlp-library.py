import os

os.environ["WANDB_API_KEY"] = "0" ## to silence warning



import numpy as np

import random

import sklearn

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

import plotly.express as px



!pip uninstall -y transformers

!pip install transformers



import transformers

import tokenizers



# Hugging Face new library for datasets (https://huggingface.co/nlp/)

!pip install nlp

import nlp



import datetime



strategy = None
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
original_train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")



original_train = sklearn.utils.shuffle(original_train)

original_train = sklearn.utils.shuffle(original_train)



validation_ratio = 0.2

nb_valid_examples = max(1, int(len(original_train) * validation_ratio))



original_valid = original_train[:nb_valid_examples]

original_train = original_train[nb_valid_examples:]
print(f"original - training: {len(original_train)} examples")

original_train.head(10)
print(f"original - validation: {len(original_valid)} examples")

original_valid.head(10)
original_test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")

print(f"original - test: {len(original_test)} examples")

original_test.head(10)
mnli = nlp.load_dataset(path='glue', name='mnli')
print(mnli, '\n')



print('The split names in MNLI dataset:')

for k in mnli:

    print('   ', k)

    

# Get the datasets

print("\nmnli['train'] is ", type(mnli['train']))



mnli['train']
print('The number of training examples in mnli dataset:', mnli['train'].num_rows)

print('The number of validation examples in mnli dataset - part 1:', mnli['validation_matched'].num_rows)

print('The number of validation examples in mnli dataset - part 2:', mnli['validation_mismatched'].num_rows, '\n')



print('The class names in mnli dataset:', mnli['train'].features['label'].names)

print('The feature names in mnli dataset:', list(mnli['train'].features.keys()), '\n')



for elt in mnli['train']:

    

    print('premise:', elt['premise'])

    print('hypothesis:', elt['hypothesis'])

    print('label:', elt['label'])

    print('label name:', mnli['train'].features['label'].names[elt['label']])

    print('idx', elt['idx'])

    print('-' * 80)

    

    if elt['idx'] >= 10:

        break
# convert to a dataframe and view

mnli_train_df = pd.DataFrame(mnli['train'])

mnli_valid_1_df = pd.DataFrame(mnli['validation_matched'])

mnli_valid_2_df = pd.DataFrame(mnli['validation_mismatched'])



mnli_train_df = mnli_train_df[['premise', 'hypothesis', 'label']]

mnli_valid_1_df = mnli_valid_1_df[['premise', 'hypothesis', 'label']]

mnli_valid_2_df = mnli_valid_2_df[['premise', 'hypothesis', 'label']]



mnli_train_df['lang_abv'] = 'en'

mnli_valid_1_df['lang_abv'] = 'en'

mnli_valid_2_df['lang_abv'] = 'en'
mnli_train_df.head(10)
mnli_valid_1_df.head(10)
mnli_valid_2_df.head(10)
snli = nlp.load_dataset(path='snli')



print('The number of training examples in snli dataset:', snli['train'].num_rows)

print('The number of validation examples in snli dataset:', snli['validation'].num_rows, '\n')



print('The class names in snli dataset:', snli['train'].features['label'].names)

print('The feature names in snli dataset:', list(snli['train'].features.keys()), '\n')



for idx, elt in enumerate(snli['train']):

    

    print('premise:', elt['premise'])

    print('hypothesis:', elt['hypothesis'])

    print('label:', elt['label'])

    print('label name:', snli['train'].features['label'].names[elt['label']])

    print('-' * 80)

    

    if idx >= 10:

        break
# convert to a dataframe and view

snli_train_df = pd.DataFrame(snli['train'])

snli_valid_df = pd.DataFrame(snli['validation'])



snli_train_df = snli_train_df[['premise', 'hypothesis', 'label']]

snli_valid_df = snli_valid_df[['premise', 'hypothesis', 'label']]



snli_train_df['lang_abv'] = 'en'

snli_valid_df['lang_abv'] = 'en'
snli_train_df.head(10)
snli_valid_df.head(10)
xnli = nlp.load_dataset(path='xnli')



print('The number of validation examples in xnli dataset:', xnli['validation'].num_rows, '\n')



print('The class names in xnli dataset:', xnli['validation'].features['label'].names)

print('The feature names in xnli dataset:', list(xnli['validation'].features.keys()), '\n')



for idx, elt in enumerate(xnli['validation']):

    

    print('premise:', elt['premise'])

    print('hypothesis:', elt['hypothesis'])

    print('label:', elt['label'])

    print('label name:', xnli['validation'].features['label'].names[elt['label']])

    print('-' * 80)

    

    if idx >= 3:

        break
# convert to a dataframe and view

buffer = {

    'premise': [],

    'hypothesis': [],

    'label': [],

    'lang_abv': []

}





for x in xnli['validation']:

    label = x['label']

    for idx, lang in enumerate(x['hypothesis']['language']):

        hypothesis = x['hypothesis']['translation'][idx]

        premise = x['premise'][lang]

        buffer['premise'].append(premise)

        buffer['hypothesis'].append(hypothesis)

        buffer['label'].append(label)

        buffer['lang_abv'].append(lang)

        

# convert to a dataframe and view

xnli_valid_df = pd.DataFrame(buffer)

xnli_valid_df = xnli_valid_df[['premise', 'hypothesis', 'label', 'lang_abv']]
xnli_valid_df.head(15 * 3)