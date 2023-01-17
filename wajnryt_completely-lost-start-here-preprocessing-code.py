# Usual imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from statistics import *
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import textstat
import warnings
warnings.filterwarnings('ignore')

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
%matplotlib inline
import os
print(os.listdir("../input"))
quora_train = pd.read_csv("../input/train.csv")
quora_train.head()
sentence="I love it, when David writes Great looking code"
parser = English() # Defines the parse sapcy will use
mytokens = parser(sentence)
print(mytokens)

mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
print(mytokens)
punctuations = string.punctuation  #gets a list of puctuations carachters from the string library
stopwords = list(STOP_WORDS) #gets a list of stop words - words that usually have little meaning in the phrase
mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
print(mytokens)
mytokens = " ".join([i for i in mytokens]) #go over each token in "mytokens" and add it to a string. Use a space (" ".) to separate the words in the new string.
print(mytokens)
# SpaCy Parser for questions
punctuations = string.punctuation
stopwords = list(STOP_WORDS)
parser = English()

def spacy_tokenizer(sentence): #Create a function called spacy_tokenizer that takes "sentence" as an argument and returns the processed sentence 
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
tqdm.pandas()
sincere_questions = quora_train["question_text"][quora_train["target"] == 0].progress_apply(spacy_tokenizer)
insincere_questions = quora_train["question_text"][quora_train["target"] == 1].progress_apply(spacy_tokenizer)