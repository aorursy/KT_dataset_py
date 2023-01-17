!pip install texthero
import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='white')





import re                                  

import string                              

import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords        

from nltk.stem import PorterStemmer        

from nltk.tokenize import TweetTokenizer

from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer





from datetime import datetime

import os

import glob



from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
# reading and concating subtitle dataset

path = '../input/chai-time-data-science/Cleaned Subtitles' # use path

all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)



df_sub = pd.concat(li, axis=0, ignore_index=True)
# reading episodes and desription dataset

df_eps = pd.read_csv('../input/chai-time-data-science/Episodes.csv')

df_desc = pd.read_csv('../input/chai-time-data-science/Description.csv')
bold('**Preview of Sutitles Dataset**')

display(df_sub.head())

bold('**Preview of Episodes Dataset**')

display(df_eps.head())

bold('**Preview of Description Dataset**')

display(df_desc.head())
def clean_text(text):

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





def text_preprocessing(text):

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    nopunc = clean_text(text)

    tokenized_text = tokenizer.tokenize(nopunc)

    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(tokenized_text)

    return combined_text
# applying the fuction

df_sub['clean_text'] = df_sub['Text'].apply(str).apply(lambda x: text_preprocessing(x))

df_desc['clean_description'] = df_desc['description'].apply(str).apply(lambda x: text_preprocessing(x))
# extracting Kaggle speakers

[list(df_eps[df_eps['category']=='Kaggle']['heroes'])]
# creating new dataframe of kaggle speakers and subltite

df_sub.set_index('Speaker', inplace=True)

kg_heroes_df = df_sub.loc[[

  'Sanyam Bhutani',

  'Abhishek Thakur',

  'Ryan Chesler',

  'Shivam Bansal',

  'Andrew Lukyanenko',

  'Dr. Vlamidir Iglovikov',

  'Dr. Yury Kashnitsky',

  'Robbert Bracco',

  'Dr. Boris Dorado',

  'Andres Torrubia',

  'Philipp Singer',

  'CPMP',

  'Eugene Khvedchenya',

  'Dr. Olivier Grellier',

  'Gilberto Titericz',

  'Dmitry Gordeev',

  'Philipp Singer',

  'Rohan Rao',

  'Anthony Goldbloom',

  'Anokas',

  'Marios',

  'John Miller',

  'Christof',

  'Inversion',

  'Russ Wolfinger',

  'Dmytro',

  'Mark Landry',

  'Dmitry Danevskiy',

  'Yauhen Babakhin',

  'Martin Henze',

  'Max J',

  'Dmitry Larko'

  ],'clean_text'].to_frame('clean_text')



kg_heroes_df.reset_index(inplace=True)

kg_heroes_df
from wordcloud import WordCloud, STOPWORDS



plt.figure(figsize=(20,50), dpi=100)

font = '../input/boldfonts/ColorTube-Regular.otf'





plt.subplot(11,3,1)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Sanyam Bhutani']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, # Maximum numbers of words we want to see 

                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud

                       max_font_size=150, min_font_size=20,  # Font size range

                       background_color="white").generate(" ".join(txt))



plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words Used By Sanyam Bhutani")



plt.subplot(11,3,2)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Abhishek Thakur']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500,

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20,  

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Abhishek Thakur")



plt.subplot(11,3,3)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Ryan Chesler']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Ryan Chesler")





plt.subplot(11,3,4)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Shivam Bansal']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Shivam Bansal")



plt.subplot(11,3,5)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Andrew Lukyanenko']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Andrew Lukyanenko")



plt.subplot(11,3,6)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dr. Vlamidir Iglovikov']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dr. Vlamidir Iglovikov")



plt.subplot(11,3,7)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dr. Yury Kashnitsky']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dr. Yury Kashnitsky")





plt.subplot(11,3,8)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Robbert Bracco']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Robbert Bracco")





plt.subplot(11,3,9)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dr. Boris Dorado']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dr. Boris Dorado")



plt.subplot(11,3,10)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Andres Torrubia']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Andres Torrubia")



plt.subplot(11,3,11)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Philipp Singer']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Philipp Singer")



plt.subplot(11,3,12)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='CPMP']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By CPMP")



plt.subplot(11,3,13)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Eugene Khvedchenya']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Eugene Khvedchenya")



plt.subplot(11,3,14)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dr. Olivier Grellier']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dr. Olivier Grellier")



plt.subplot(11,3,15)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Gilberto Titericz']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Gilberto Titericz")



plt.subplot(11,3,16)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dmitry Gordeev']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dmitry Gordeev")



plt.subplot(11,3,17)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Philipp Singer']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Philipp Singer")



plt.subplot(11,3,18)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Rohan Rao']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Rohan Rao")



plt.subplot(11,3,19)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Anthony Goldbloom']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Anthony Goldbloom")



plt.subplot(11,3,20)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Anokas']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Anokas")



plt.subplot(11,3,21)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Marios']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Marios")



plt.subplot(11,3,22)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='John Miller']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By John Miller")



plt.subplot(11,3,23)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Christof']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Christof")



plt.subplot(11,3,24)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Inversion']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Inversion")



plt.subplot(11,3,25)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Russ Wolfinger']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Russ Wolfinger")



plt.subplot(11,3,26)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dmytro']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dmytro")



plt.subplot(11,3,27)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dmitry Danevskiy']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dmitry Danevskiy")



plt.subplot(11,3,28)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Mark Landry']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Mark Landry")



plt.subplot(11,3,29)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Yauhen Babakhin']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Yauhen Babakhin")



plt.subplot(11,3,30)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Martin Henze']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Martin Henze")



plt.subplot(11,3,31)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Max J']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Max J")



plt.subplot(11,3,32)

txt = kg_heroes_df[kg_heroes_df['Speaker'] =='Dmitry Larko']['clean_text']

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='GnBu', 

                       margin=0,

                       stopwords=STOPWORDS,

                       max_words=500, 

                       min_word_length=3, 

                       max_font_size=150, min_font_size=20, 

                       background_color="white").generate(" ".join(txt))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words  Used By Dmitry Larko")



plt.show()
font = '../input/boldfonts/FFF_Tusj.ttf'

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='cividis', 

                       margin=0,

                       max_words=500, # Maximum numbers of words we want to see 

                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud

                       max_font_size=150, min_font_size=20,  # Font size range

                       background_color="white").generate(" ".join(df_desc['clean_description']))



plt.figure(figsize=(10, 16))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words in Decription", fontsize=15)

plt.show()
font = '../input/boldfonts/FFF_Tusj.ttf'

word_cloud = WordCloud(font_path=font,

                       width=1600,

                       height=800,

                       colormap='cividis', 

                       margin=0,

                       max_words=500, # Maximum numbers of words we want to see 

                       min_word_length=3, # Minimum numbers of letters of each word to be part of the cloud

                       max_font_size=150, min_font_size=20,  # Font size range

                       background_color="white").generate(" ".join(df_eps['episode_name']))



plt.figure(figsize=(10, 16))

plt.imshow(word_cloud, interpolation="gaussian")

plt.axis("off")

plt.title("Frequent Words in Episode Name", fontsize=15)

plt.show()
# Extracting polarity, text length and Word Count

kg_heroes_df['polarity'] = kg_heroes_df['clean_text'].map(lambda text: TextBlob(text).sentiment.polarity)

kg_heroes_df['text_len'] = kg_heroes_df['clean_text'].astype(str).apply(len)

kg_heroes_df['word_count'] = kg_heroes_df['clean_text'].apply(lambda x: len(str(x).split()))
kg_heroes_df['polarity'].iplot(

    kind='hist',

    bins=50,

    xTitle='polarity',

    linecolor='black',

    color='dodgerblue',

    yTitle='count',

    title='Sentiment Polarity Distribution')
kg_heroes_df['text_len'].iplot(

    kind='hist',

    bins=100,

    xTitle='review length',

    linecolor='black',

    color='dodgerblue',

    yTitle='count',

    title='Text Length Distribution')
kg_heroes_df['word_count'].iplot(

    kind='hist',

    bins=100,

    xTitle='word count',

    linecolor='black',

    color='dodgerblue',

    yTitle='count',

    title='Text Word Count Distribution')
def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

common_words = get_top_n_bigram(kg_heroes_df['clean_text'], 20)



df_bi = pd.DataFrame(common_words, columns = ['Text' , 'count'])

df_bi.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', color='dodgerblue', title='Top 20 bigrams in Text')
def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]

common_words = get_top_n_trigram(kg_heroes_df['clean_text'], 20)



df_tri = pd.DataFrame(common_words, columns = ['Text' , 'count'])

df_tri.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', color='dodgerblue', title='Top 20 trigrams in Text')
import texthero as hero



temp = kg_heroes_df.set_index('Speaker')

temp = temp.drop(index='Sanyam Bhutani')

temp.reset_index(inplace=True)



temp['pca'] = (

   temp['clean_text']

   .pipe(hero.tfidf)

   .pipe(hero.pca)

)

hero.scatterplot(temp, 'pca', color='Speaker', title="PCA Speaker's Subtitle")
# Let's see the episode data

df_eps.info()
# plot bar plot

fig = go.Figure(data=[

    go.Bar(x=df_eps.episode_id, 

           y=df_eps.youtube_impressions,

           name='Youtube Impressions',

           marker_color='#000000'),

    go.Bar(x=df_eps.episode_id, 

           y=df_eps.youtube_impression_views,

           name='Youtube Impression Views', 

           marker_color='#FF0000')

])

# Change the bar mode

fig.update_layout(barmode='stack', template = 'plotly_white', width=700, height=700, title_text = 'Total Youtube Impressions And Impression Views',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()





# plot of growth rate of confirmed cases

fig1 = px.scatter(df_eps, 

                 x='episode_id', 

                  y="youtube_ctr", 

                  text='youtube_ctr',)

fig1.update_traces(marker=dict(size=3,

                              line=dict(width=2,

                                        color='DarkSlateGrey')),

                  marker_color='#4169e1',

                  mode='text+lines+markers',textposition='top center', )



fig1.update_layout(template = 'plotly_white', width=700, height=700, title_text = 'Click Through Rate',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig1.show()
# sorting 

cat_ctr = df_eps.sort_values(by = 'youtube_ctr',ascending = False)



# plot

fig = px.bar(cat_ctr, 

                 x='episode_id', 

                  y="youtube_ctr", 

                 color='category')



fig.update_layout(template = 'plotly_white',width=700, height=500, title_text = '<b>Category Wise Click Through Rate',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
# plot bar plot

fig = go.Figure(data=[

    go.Bar(x=df_eps.episode_id, 

           y=df_eps.youtube_views,

           name='Views',

           marker_color='#008080'),

    go.Bar(x=df_eps.episode_id, 

           y=df_eps.youtube_likes,

           name='Likes', 

           marker_color='#ADFF2F'),

    go.Bar(x=df_eps.episode_id, 

           y=df_eps.youtube_comments,

           name='Comment', 

           marker_color='#FF4500')

    

])

# Change the bar mode

fig.update_layout(barmode='stack', template = 'plotly_white', width=700, height=700, title_text = 'Total Youtube View, Like, Comment',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
# sorting 

cat_views = df_eps.sort_values(by = 'youtube_views',ascending = False)



# plot

fig = px.bar(cat_views, 

            x='episode_id', 

            y='youtube_views', 

            color='category')



fig.update_layout(template = 'plotly_white',width=700, height=500, title_text = '<b>Category Wise Views Distribution',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
df_eps['recording_date'] = df_eps['recording_date'].apply(pd.to_datetime)

fig = px.scatter(df_eps, 

            x='recording_date', 

            y='youtube_views')

fig.update_traces(marker=dict(size=4.5),

                  mode='markers',

                  marker_color='#800080')

fig.update_layout(template = 'plotly_white',width=700, height=500, title_text = '<b>Trend of Views over the year',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
fig = go.Figure(data=[

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.episode_duration,

           name='Episode Duration',

           mode='lines', 

           line_color='indigo',

           fill='tonexty'),

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.youtube_avg_watch_duration,

           name='Avg Watch Duration', 

           mode='lines', 

           line_color='blue',

               fill='tonexty'),



])





fig.update_layout(template = 'plotly_white', width=700, height=700, title_text = 'Total Episodes Duration Vs Avg Watch Duration(In Sec)',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()



df_eps['youtube_watch_hours'].iplot(kind='area',

                                        fill=True,

                                        opacity=1,

                                        color = 'blue',

                                        xTitle='Episode',

                                        yTitle='Duration(Hrs)',

                                        title='Total watch hours on YouTube')
fig = px.bar(df_eps.sort_values('youtube_subscribers', ascending= False).sort_values('youtube_subscribers', ascending=True), 

             x="youtube_subscribers", y="episode_id", 

             title='New subscribers to YouTube channel', 

             text='youtube_subscribers', 

             orientation='h', 

             width=700, height=2000)

fig.update_traces(marker_color='#FFA500', opacity=0.8, textposition='inside')



fig.update_layout(template = 'plotly_white')

fig.show()
# let's see the episode 27 video

print(df_eps[df_eps['episode_id'] == 'E27']['episode_name'])

from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('205j37G1cxw',width=400, height=200)
# fit the regression

from sklearn.linear_model import LinearRegression

X = df_eps['youtube_watch_hours']

Y = df_eps['youtube_subscribers']



temp_df = pd.DataFrame({'youtube_watch_hours': X, 'youtube_subscribers':Y})

reg = LinearRegression().fit(np.vstack(temp_df['youtube_watch_hours']), Y)

temp_df['bestfit'] = reg.predict(np.vstack(temp_df['youtube_watch_hours']))



# plot

fig = go.Figure(data=[

    go.Scatter(x=temp_df['youtube_watch_hours'], 

               y=temp_df['youtube_subscribers'].values, 

               mode='markers',

               name='Watch hours vs New subscribers', 

               marker_color='black'),

    

    go.Scatter(x=X, 

               y=temp_df['bestfit'],

               name='line of best fit', 

               mode='lines', 

               marker_color='red'),

])

    

fig.update_layout(template = 'plotly_white', width=700, height=500, title_text = 'Relationship Between Watch Hours and New Subcribers',

                  xaxis_title = 'youtube_watch_hours', yaxis_title = 'youtube_subscribers',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
from sklearn.linear_model import LinearRegression

X = df_eps['youtube_views']

Y = df_eps['youtube_likes']



temp_df = pd.DataFrame({'youtube_views': X, 'youtube_likes':Y})

reg = LinearRegression().fit(np.vstack(temp_df['youtube_views']), Y)

temp_df['bestfit'] = reg.predict(np.vstack(temp_df['youtube_likes']))



# plot

fig = go.Figure(data=[

    go.Scatter(x=temp_df['youtube_views'], 

               y=temp_df['youtube_likes'].values, 

               mode='markers',

               name='Watch hours vs New subscribers', 

               marker_color='black'),

    

    go.Scatter(x=X, 

               y=temp_df['bestfit'],

               name='line of best fit', 

               mode='lines', 

               marker_color='red'),

])

    

fig.update_layout(template = 'plotly_white', width=700, height=500, title_text = 'Relationship Between Watch Hours and New Subcribers',

                  xaxis_title = 'youtube_views', yaxis_title = 'youtube_likes',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
trace1 = go.Scatter(

                x=df_eps['episode_id'],

                y=df_eps['anchor_plays'],

                name="Total Anchor Play",

                mode='lines+markers',

                line_color='orange')

trace2 = go.Scatter(

                x=df_eps['episode_id'],

                y=df_eps['spotify_listeners'],

                name="Spotify listeners",

                mode='lines+markers',

                line_color='red')



trace3 = go.Scatter(

                x=df_eps['episode_id'],

                y=df_eps['apple_listeners'],

                name="Apple listeners",

                mode='lines+markers',

                line_color='green')





layout = go.Layout(template="plotly_white", width=700, height=500, title_text = 'Number of unique listeners on spotify and Apple podcasts.</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2,trace3], layout = layout)

fig.show()
fig = go.Figure(data=[

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.spotify_starts,

           name='Spotify Starts(>= sec)',

           mode='lines', 

           line_color='red',

           fill='tonexty'),

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.spotify_streams,

           name='Spotify Streams(>=60 sec)', 

           mode='lines', 

           line_color='deeppink',

               fill='tonexty'),



])





fig.update_layout(template = 'plotly_white', width=700, height=700, title_text = 'Spotify Starts(>= sec) Vs Spotify Streams(>=60 sec)',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()
fig = go.Figure(data=[

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.apple_listened_hours * 3600,

           name='Total Lintened',

           mode='lines', 

           line_color='orange',

           fill='tonexty'),

    go.Scatter(x=df_eps.episode_id, 

           y=df_eps.apple_avg_listen_duration,

           name='Total Avg Listened', 

           mode='lines', 

           line_color='green',

               fill='tonexty'),



])





fig.update_layout(template = 'plotly_white', width=700, height=700, title_text = 'Total Listen Vs Avg Listen',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig.show()