import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#!python -m spacy download en_core_web_lg

!python -m spacy download en_core_web_sm

!pip install sumy
import string

import pandas as pd

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from sumy.summarizers import luhn

from sumy.utils import get_stop_words

from sumy.nlp.stemmers import Stemmer

from sumy.summarizers.luhn import LuhnSummarizer 

from sumy.parsers.plaintext import PlaintextParser

from sumy.nlp.tokenizers import Tokenizer as sumytoken

from sumy.summarizers.lex_rank import LexRankSummarizer

from sumy.summarizers.lsa import LsaSummarizer as Summarizer
reviews_datasets = pd.read_csv(r'/kaggle/input/movies-metadata/movies_metadata.csv')

reviews_datasets.dropna()

reviews_datasets.head(5)
genre = "Action"

filtered_reviews = reviews_datasets[reviews_datasets['genres'].str.contains(genre)]

filtered_reviews.head(5)
text = filtered_reviews.sample(25)['overview'].str.cat(sep='. ')
LANGUAGE = "english"

SENTENCES_COUNT = 3

parser = PlaintextParser.from_string((text), sumytoken(LANGUAGE))

stemmer = Stemmer(LANGUAGE)

SEP = " "



def lexrank_summarizer():

    sentences = []

    summarizer_LexRank = LexRankSummarizer(stemmer)

    summarizer_LexRank.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer_LexRank(parser.document, SENTENCES_COUNT):

        sentences.append(str(sentence))

    return SEP.join(sentences)

        

def lsa_summarizer():

    sentences = []

    summarizer_lsa = Summarizer(stemmer)

    summarizer_lsa.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer_lsa(parser.document, SENTENCES_COUNT):

        sentences.append(str(sentence))

    return SEP.join(sentences)

        

def luhn_summarizer():

    sentences = []

    summarizer_luhn = LuhnSummarizer(stemmer)

    summarizer_luhn.stop_words = get_stop_words(LANGUAGE)

    for sentence in summarizer_luhn(parser.document, SENTENCES_COUNT):

        sentences.append(str(sentence))

    return SEP.join(sentences)



sum1 = lexrank_summarizer()

sum2 = lsa_summarizer()

sum3 = luhn_summarizer()



print(sum1)