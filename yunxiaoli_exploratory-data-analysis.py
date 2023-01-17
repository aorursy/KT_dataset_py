!pip install -q pyicu

!pip install -q pycld2

!pip install -q polyglot

!pip install -q textstat

!pip install -q googletrans

!pip install morfessor
import warnings

warnings.filterwarnings("ignore")



import os

import gc

import re

import folium

import textstat

from scipy import stats

from colorama import Fore, Back, Style, init



import math

import numpy as np

import scipy as sp

import pandas as pd



import random

import networkx as nx

from pandas import Timestamp



from PIL import Image

from IPython.display import SVG

from keras.utils import model_to_dot



import requests

from IPython.display import HTML



import seaborn as sns

from tqdm import tqdm

import matplotlib.cm as cm

import matplotlib.pyplot as plt



tqdm.pandas()



import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer  



import nltk

from textblob import TextBlob



from nltk.corpus import wordnet

from nltk.corpus import stopwords

from googletrans import Translator

from nltk import WordNetLemmatizer

from polyglot.detect import Detector

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS

from nltk.sentiment.vader import SentimentIntensityAnalyzer



stopword=set(STOPWORDS)



lem = WordNetLemmatizer()

tokenizer=TweetTokenizer()



np.random.seed(0)
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

os.listdir(DATA_PATH)
TEST_PATH = DATA_PATH + "test.csv"

VAL_PATH = DATA_PATH + "validation.csv"

TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"



val_data = pd.read_csv(VAL_PATH)

test_data = pd.read_csv(TEST_PATH)

train_data = pd.read_csv(TRAIN_PATH)
train_data.head()
val_data.head()
test_data.head()
def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in train_data["comment_text"]])

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1000).generate(text)

fig = px.imshow(wordcloud)

fig.update_layout(title_text='Common words in comments')
def get_language(text):

    return Detector("".join(x for x in text if x.isprintable()), quiet=True).languages[0].name



train_data["lang"] = train_data["comment_text"].progress_apply(get_language)
lang_list = sorted(list(set(train_data["lang"])))

counts = [list(train_data["lang"]).count(cont) for cont in lang_list]

df = pd.DataFrame(np.transpose([lang_list, counts]))

df.columns = ["Language", "Count"]

df["Count"] = df["Count"].apply(int)



df_en = pd.DataFrame(np.transpose([["English", "Non-English"], [max(counts), sum(counts) - max(counts)]]))

df_en.columns = ["Language", "Count"]



fig = px.bar(df_en, x="Language", y="Count", title="Language of comments", color="Language", text="Count")

fig.update_layout(template="plotly_white")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.5

fig.data[1].marker.line.color = 'rgb(0, 0, 0)'

fig.data[1].marker.line.width = 0.5

fig.data[0].textfont.color = "black"

fig.data[0].textposition = "outside"

fig.data[1].textfont.color = "black"

fig.data[1].textposition = "outside"

fig
fig = px.bar(df.query("Language != 'en' and Language != 'un'").query("Count >= 50"),

             y="Language", x="Count", title="Language of non-English comments", template="plotly_white", color="Language", text="Count", orientation="h")

fig.update_traces(marker=dict(line=dict(width=0.75,

                                        color='black')),  textposition="outside")

fig.update_layout(showlegend=False)

fig
fig = go.Figure([go.Pie(labels=df.query("Language != 'en' and Language != 'un'").query("Count >= 50")["Language"],

           values=df.query("Language != 'en' and Language != 'un'").query("Count >= 50")["Count"])])

fig.update_layout(title_text="Pie chart of non-English languages", template="plotly_white")

fig.data[0].marker.colors = [px.colors.qualitative.Plotly[2:]]

fig.data[0].textfont.color = "black"

fig.data[0].textposition = "outside"

fig.show()
clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))

clean_mask=clean_mask[:,:,1]



subset = train_data.query("toxic == 0")

text = subset.comment_text.values

wc = WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.figure(figsize=(7.5, 7.5))

plt.axis("off")

plt.title("Words frequented in Clean Comments", fontsize=16)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()



clean_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))

clean_mask=clean_mask[:,:,1]



subset = train_data.query("toxic == 1")

text = subset.comment_text.values

wc = WordCloud(background_color="black",max_words=2000,mask=clean_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.figure(figsize=(7.5, 7.5))

plt.axis("off")

plt.title("Words frequented in Toxic Comments", fontsize=16)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()
toxic_mask=np.array(Image.open("../input/imagesforkernal/toxic-sign.png"))

toxic_mask=toxic_mask[:,:,1]

#wordcloud for clean comments

subset=train_data.query("obscene == 1")

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=4000,mask=toxic_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.figure(figsize=(20,20))

plt.subplot(221)

plt.axis("off")

plt.title("Words frequented in Obscene Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)



#Severely toxic comments

plt.subplot(222)

severe_toxic_mask=np.array(Image.open("../input/imagesforkernal/bomb.png"))

severe_toxic_mask=severe_toxic_mask[:,:,1]

subset=train_data[train_data.severe_toxic==1]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=severe_toxic_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Words frequented in Severe Toxic Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'Reds' , random_state=244), alpha=0.98)



#Threat comments

plt.subplot(223)

threat_mask=np.array(Image.open("../input/imagesforkernal/anger.png"))

threat_mask=threat_mask[:,:,1]

subset=train_data[train_data.threat==1]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=threat_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Words frequented in Threatening Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'summer' , random_state=2534), alpha=0.98)



#insult

plt.subplot(224)

insult_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))

insult_mask=insult_mask[:,:,1]

subset=train_data[train_data.insult==1]

text=subset.comment_text.values

wc= WordCloud(background_color="black",max_words=2000,mask=insult_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Words frequented in insult Comments", fontsize=20)

plt.imshow(wc.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)



plt.show()