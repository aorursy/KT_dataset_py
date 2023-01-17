

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

pd.options.display.max_colwidth = 100



# Todo: read data

train=pd.read_csv("../input/quora-insincere-questions-classification/train.csv")

train.head()

# Todo: show some basic information about the data

# example: what is the size of the data, how is the distribution of targe look like

train.shape



train.isnull().sum().sum()

# This shows that there are no missing values
train['target'].value_counts()

# the distribution of target ??? 
# Todo: try some basic text cleaning steps

# example:

# - remove numbers

def number_remove (string):

    result = ''.join([i for i in string if not i.isdigit()])

    return result

number_removed=train['question_text'].apply(number_remove)

number_removed

def space_remove(string):

    return string.replace(" ","") 

# - remove multiple spaces

number_space_removed=number_removed.apply(space_remove)

number_space_removed
def punc_remove(string):

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    no_punct = ""

    for char in string:

        if char not in punctuations:

            no_punct = no_punct + char

    return no_punct



punc_remove("aser2#$GF<sdf")
#cleaned text 

number_space_punct_removed=number_space_removed.apply(punc_remove)

number_space_punct_removed
train['cleaned_text']=number_space_punct_removed

train.head()
columns=['qid','cleaned_text','target']

cleaned_df=train[columns]

cleaned_df.head()
# Todo: vectorizing text data using bag-of-words (ngram) and Tfidf





# Todo: explore the arguments in CountVectorizer and TfidfVectorizer, compare the results





# Todo: save the vectorized data use `pickle`
