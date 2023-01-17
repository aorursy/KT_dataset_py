!pip install textstat

!pip install syllables



# Import python packages

import os, sys

import pandas as pd,numpy as np

import re

import spacy

from os import walk

import shutil

from shutil import copytree, ignore_patterns

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

import xml.etree.cElementTree as ET

from collections import OrderedDict

import json

from __future__ import unicode_literals, print_function

import plac

import random

from pathlib import Path

from spacy.util import minibatch, compounding

from spacy.matcher import Matcher

#from word2number import w2n

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from datetime import date

import calendar

from sklearn.feature_extraction.text import CountVectorizer

from itertools import takewhile, tee

import itertools

import nltk, string

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.cluster.util import cosine_distance

import networkx as nx

from PIL import Image,ImageFilter

import textstat

from textstat.textstat import textstatistics, easy_word_set, legacy_round 

import syllables

from IPython.display import display, HTML, Javascript



bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

additional_data_dir = '../input/cityofla/CityofLA/Additional data'

STOP_WORDS = stopwords.words('english')



%matplotlib inline
feminine_coded_words = [

    "agree",

    "affectionate",

    "child",

    "cheer",

    "collab",

    "commit",

    "communal",

    "compassion",

    "connect",

    "considerate",

    "cooperat",

    "co-operat",

    "depend",

    "emotiona",

    "empath",

    "feel",

    "flatterable",

    "gentle",

    "honest",

    "interpersonal",

    "interdependen",

    "interpersona",

    "inter-personal",

    "inter-dependen",

    "inter-persona",

    "kind",

    "kinship",

    "loyal",

    "modesty",

    "nag",

    "nurtur",

    "pleasant",

    "polite",

    "quiet",

    "respon",

    "sensitiv",

    "submissive",

    "support",

    "sympath",

    "tender",

    "together",

    "trust",

    "understand",

    "warm",

    "whin",

    "enthusias",

    "inclusive",

    "yield",

    "share",

    "sharin"

]



masculine_coded_words = [

    "active",

    "adventurous",

    "aggress",

    "ambitio",

    "analy",

    "assert",

    "athlet",

    "autonom",

    "battle",

    "boast",

    "challeng",

    "champion",

    "compet",

    "confident",

    "courag",

    "decid",

    "decision",

    "decisive",

    "defend",

    "determin",

    "domina",

    "dominant",

    "driven",

    "fearless",

    "fight",

    "force",

    "greedy",

    "head-strong",

    "headstrong",

    "hierarch",

    "hostil",

    "impulsive",

    "independen",

    "individual",

    "intellect",

    "lead",

    "logic",

    "objective",

    "opinion",

    "outspoken",

    "persist",

    "principle",

    "reckless",

    "self-confiden",

    "self-relian",

    "self-sufficien",

    "selfconfiden",

    "selfrelian",

    "selfsufficien",

    "stubborn",

    "superior",

    "unreasonab"

]



hyphenated_coded_words = [

    "co-operat",

    "inter-personal",

    "inter-dependen",

    "inter-persona",

    "self-confiden",

    "self-relian",

    "self-sufficien"

]
def clean_up_word_list(ad_text):

    cleaner_text = ''.join([i if ord(i) < 128 else ' '

        for i in ad_text])

    cleaner_text = re.sub("[\\s]", " ", cleaner_text, 0, 0)

    cleaned_word_list = re.sub(u"[\.\t\,“”‘’<>\*\?\!\"\[\]\@\':;\(\)\./&]",

        " ", cleaner_text, 0, 0).split(" ")

    word_list = [word.lower() for word in cleaned_word_list if word != ""]

    return de_hyphen_non_coded_words(word_list)



def de_hyphen_non_coded_words(word_list):

    for word in word_list:

        if word.find("-"):

            is_coded_word = False

            for coded_word in hyphenated_coded_words:

                if word.startswith(coded_word):

                    is_coded_word = True

            if not is_coded_word:

                word_index = word_list.index(word)

                word_list.remove(word)

                split_words = word.split("-")

                word_list = (word_list[:word_index] + split_words +

                    word_list[word_index:])

    return word_list



def assess_coding(row):

    coding = ''

    coding_score = row["feminine_ad_word_count"] - row["masculine_ad_word_count"]

    if coding_score == 0:

        if row["feminine_ad_word_count"]>0:

            coding = "neutral"

        else:

            coding = "empty"

    elif coding_score > 3:

        coding = "strongly feminine"

    elif coding_score > 0:

        coding = "feminine"

    elif coding_score < -3:

        coding = "strongly masculine"

    else:

        coding = "masculine"

    return coding



def assess_coding_txt(fem_word_count, masc_word_count):

    coding = ''

    coding_score = fem_word_count - masc_word_count

    if coding_score == 0:

        if fem_word_count>0:

            coding = "neutral"

        else:

            coding = "empty"

    elif coding_score > 3:

        coding = "strongly feminine"

    elif coding_score > 0:

        coding = "feminine"

    elif coding_score < -3:

        coding = "strongly masculine"

    else:

        coding = "masculine"

    return coding



def find_and_count_coded_words(advert_word_list, gendered_word_list):

    gender_coded_words = [word for word in advert_word_list

        for coded_word in gendered_word_list

        if word.startswith(coded_word)]

    return (",").join(gender_coded_words), len(gender_coded_words)



def assessBias(txt):

    words = clean_up_word_list(txt)

    txt_masc_coded_words, masc_word_count = find_and_count_coded_words(words, masculine_coded_words)

    txt_fem_coded_words, fem_word_count = find_and_count_coded_words(words, feminine_coded_words)

    coding = assess_coding_txt(fem_word_count, masc_word_count)

#     print('List of masculine words found:')

#     print(txt_masc_coded_words)

#     print('\nList of feminine words found:')

#     print(txt_fem_coded_words)

    return coding, txt_masc_coded_words, txt_fem_coded_words



def getWordsFrame(words):

    words = pd.Series(words.split(','))

    words = words.value_counts()

    df = pd.DataFrame({'words':words.index, 'count':words.values})

    df.index = np.arange(1,len(df)+1)

    return df



def getResult(filename, bulletin_dir):

    CONTENT = ''

    jobs_list = []

    for file_name in os.listdir(bulletin_dir):

        if file_name == filename:

            with open(os.path.join(bulletin_dir,file_name), encoding = "ISO-8859-1") as f:

                CONTENT = f.read()

            break

    

    bias_code, masc_words, fem_words = assessBias(CONTENT)

    df_masc_words = getWordsFrame(masc_words)

    df_fem_words = getWordsFrame(fem_words)  



    dfScores = pd.DataFrame(["flesch_reading_ease","flesch_kincaid_grade",

                            "smog_index","coleman_liau_index","automated_readability_index",

                            "dale_chall_readability_score","gunning_fog"], columns = ['Label'])

    dfScores['Values'] = [

        textstat.flesch_reading_ease(CONTENT),

        textstat.flesch_kincaid_grade(CONTENT),

        textstat.smog_index(CONTENT),

        textstat.coleman_liau_index(CONTENT),

        textstat.automated_readability_index(CONTENT),

        textstat.dale_chall_readability_score(CONTENT),

        textstat.gunning_fog(CONTENT)

    ]

    dfScores.index = np.arange(1,len(dfScores)+1)

    if CONTENT!='':

        result = {

            "file_name": filename,

            "bias_code": bias_code,

            "df_masc_words": df_masc_words,

            "df_fem_words": df_fem_words,

            "dfScores": dfScores

        }

    else:

        result = {"FNF": "File not found in the directory."}

    return result



def viewOutput(response):

    if 'FNF' not in response.keys():

        display(HTML("<center><h1><b>File : " + response["file_name"] + "</b></h1></center>"))

        display(HTML("Content of the file is found to be : <b>"+response["bias_code"].title()+"</b> coded."))

        display(HTML("<b>List of masculine words in the file:</b>"))

        display(response["df_masc_words"])

        display(HTML("<b>List of feminine words in the file:</b>"))

        display(response["df_fem_words"])

        display(response["dfScores"])

    else:

        display(HTML(response["FNF"]))
response = getResult('AIRPORT ENGINEER 7256 070618.txt', bulletin_dir)

viewOutput(response)
response = getResult('AIRPORT MANAGER 7260 120216.txt', bulletin_dir)

viewOutput(response)
response = getResult('Incorrect File Name', bulletin_dir)

viewOutput(response)
response = getResult('BOILERMAKER SUPERVISOR 3737 101714.txt', bulletin_dir)

viewOutput(response)