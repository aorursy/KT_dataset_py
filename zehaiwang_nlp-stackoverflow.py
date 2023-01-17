import numpy as np

import os

import glob

import pandas as pd

import nltk



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import time

import datetime

import gc

%matplotlib inline
!ls ../input
new_users_df = pd.read_csv('../input/stackoverflow-preliminary-eda/new_users.csv').drop('Unnamed: 0',axis=1)
new_users_df.head()
new_users_df.info()
post_df = pd.read_csv('../input/fork-of-stackoverflow-preliminary-eda/posts_2017.csv').drop('Unnamed: 0',axis=1)
post_df.head()
post_df.info()
# rate of post with accepted answers

post_df['accepted_answer_id'].isnull().value_counts(normalize = True).plot.bar()

plt.title('Post get accepted answers')

plt.ylabel('accpted answer rate')
# new feature whether the post has accepted answer

post_df['Get an accepted answer?'] = post_df['accepted_answer_id'].notnull()
# create corpor
def print_log_stats(col,ax,log_flag = False):

    #print(post_df[col].describe()

    if log_flag:

        sns.boxplot(post_df[col].apply(np.log), ax = ax)

        #ax.set_xlabel('log(%s)'%col)

    else:

        sns.boxplot(post_df[col], ax = ax)

        #ax.set_xlabel('%s'%col)

    #ax.set_title('%s'%col, fontsize = 14)
fig, axes = plt.subplots(5,1,figsize=(13,15))

cols = ['answer_count', 'comment_count', 'favorite_count', 'score', 'view_count']

for i in range(5):

        print_log_stats(cols[i],axes[i])

        

print(post_df['score'].describe())

sns.boxplot((post_df['score']-post_df['score'].min()).apply(np.log))

plt.xlabel('log(score)')

plt.title('score distributions')
fig, ax = plt.subplots(figsize=(8,7))

sns.heatmap(post_df.drop(['post_type_id','accepted_answer_id'],axis =1).corr(), annot=True, linewidths = 0.5, cmap = "YlGnBu", ax=ax)

fig, axes = plt.subplots(2,2,figsize=(7,6))





post_df.groupby('Get an accepted answer?')['answer_count'].mean().plot.bar(ax= axes[0,0])

axes[0,0].set_title('mean(Answer_count)')



post_df.groupby('Get an accepted answer?')['score'].mean().plot.bar(ax= axes[0,1])

axes[0,1].set_title('mean(Score)')



post_df.groupby('Get an accepted answer?')['comment_count'].mean().plot.bar(ax= axes[1,1])

axes[1,1].set_title('mean(Comments_count)')



post_df.groupby('Get an accepted answer?')['view_count'].mean().plot.bar(ax= axes[1,0])

axes[1,0].set_title('mean(View_count)')

plt.subplots_adjust(hspace=0.6)

bad_score_df = post_df[post_df['score']<0]

print(len(bad_score_df))

sns.boxplot(bad_score_df.score)
post_df['score'].describe()
good_score_df = post_df[post_df['score']> 2.0]

print(len(good_score_df))

sns.boxplot(good_score_df.score)
pd.DataFrame(data = {'bad_questions':[len(bad_score_df)],'good_questions':[len(good_score_df)]}).plot.bar()

plt.xlabel(" ")
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, background_color = 'black', mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=background_color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=600, 

                    height=300,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
plot_wordcloud(bad_score_df["title"], title="Word Cloud of bad titles")
plot_wordcloud(good_score_df["title"], title="Word Cloud of good titles", background_color='white')
from collections import defaultdict

from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



train0_df = bad_score_df

train1_df = good_score_df

## English stop word

from nltk.corpus import stopwords

stopwords_corp = set(STOPWORDS)

nltk_stopwords = set(stopwords.words('english'))

more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown','-',':'}

stopwords_corp = stopwords_corp.union(nltk_stopwords)

stopwords_corp = stopwords_corp.union(more_stopwords)





## custom function for ngram generation ##

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in stopwords_corp]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]



## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace



## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'blue')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'blue')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of bad questions", 

                                          "Frequent words of good questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=700, width=700, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')



#plt.figure(figsize=(10,16))

#sns.barplot(x="ngram_count", y="ngram", data=fd_sorted.loc[:50,:], color="b")

#plt.title("Frequent words for Insincere Questions", fontsize=16)

#plt.show()
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'green')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'green')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.1,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent bigrams of bad questions", 

                                          "Frequent bigrams of good questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'orange')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent trigrams of bad questions", 

                                          "Frequent trigrams of good questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
print(post_df['view_count'].describe())

sns.boxplot(post_df['view_count'].apply(np.log))

plt.xlabel('log(view_counts)')

plt.title('View counts distributions')
post_df['log_view_count'] = post_df['view_count'].apply(np.log)

post_df['log_view_count'].describe()
attetion_questions = post_df[post_df['log_view_count'] > post_df['log_view_count'].quantile(0.75)]

ignored_questions = post_df[post_df['log_view_count'] < post_df['log_view_count'].quantile(0.25)]

print (len(attetion_questions))

print(len(ignored_questions))
plot_wordcloud(ignored_questions["title"], title="Word Cloud of ignored titles")
plot_wordcloud(attetion_questions["title"], title="Word Cloud of alluring titles", background_color='white')
train0_df = ignored_questions

train1_df = attetion_questions
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'green')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'green')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.1,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent bigrams of ignored questions", 

                                          "Frequent bigrams of alluring questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'orange')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent trigrams of ignored questions", 

                                          "Frequent trigrams of good questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
sns.boxplot(post_df['view_count'].apply(np.log))
sns.boxplot(post_df['answer_count'])
# Rescal

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

post_df['target'] = (scaler.fit_transform(post_df[['answer_count']])+ scaler.fit_transform(post_df[['log_view_count']]))/2

print (post_df['target'].describe())

sns.boxplot(post_df['target'])
train0_df = post_df[post_df['target'] < post_df['target'].quantile(0.25)]

train1_df = post_df[post_df['target'] > post_df['target'].quantile(0.75)]
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,2):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'green')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'green')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.1,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent bigrams of ignored questions", 

                                          "Frequent bigrams of alluring questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")

py.iplot(fig, filename='word-plots')
## Get the bar chart from sincere questions ##

freq_dict0 = defaultdict(int)

for sent in train0_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict0[word] += 1

fd_sorted0 = pd.DataFrame(sorted(freq_dict0.items(), key=lambda x: x[1])[::-1])

fd_sorted0.columns = ["word", "wordcount"]



trace0_set = set(fd_sorted0['word'].head(50).tolist())



## Get the bar chart from insincere questions ##

freq_dict1 = defaultdict(int)

for sent in train1_df["title"]:

    for word in generate_ngrams(sent,3):

        freq_dict1[word] += 1

fd_sorted1 = pd.DataFrame(sorted(freq_dict1.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]



trace1_set = set(fd_sorted1['word'].head(50).tolist())



inter_words = trace0_set.intersection(trace1_set)



trace0 = horizontal_bar_chart(fd_sorted0.head(50)[fd_sorted0.head(50)['word'].isin(trace0_set - inter_words)], 'orange')

trace1 = horizontal_bar_chart(fd_sorted1.head(50)[fd_sorted1.head(50)['word'].isin(trace1_set - inter_words)], 'orange')



# Creating two subplots

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.2,specs=[[{'l':0.1},{'l':0.1}]],

                          subplot_titles=["Frequent trigrams of ignored questions", 

                                          "Frequent trigrams of good questions"])

fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")

py.iplot(fig, filename='word-plots')
post_df['tags'].head().apply(lambda x: len(x.split('|')))
post_df['num_tags'] = post_df['tags'].apply(lambda x: len(str(x).split('|')))
sns.distplot(post_df['num_tags'])
sns.boxplot(x = 'num_tags', y = 'target', data = post_df)
sns.barplot(x = 'num_tags', y = 'target', data = post_df)
df = post_df[['id','title','body','creation_date','num_tags']]

target = post_df['target']
post_df['body'].iloc[1]
from bs4 import BeautifulSoup

import re



from html.parser import HTMLParser



class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()



# cleantext = BeautifulSoup(post_df['body'].iloc[1], 'lxml').text

# cleantext



text = strip_tags(post_df['body'].iloc[1])

str(text).replace('\n',' ')



print(len(re.findall('\n', text)))



print(df['body'].head(1))

df['body'].head(5).apply(lambda x: len(re.findall('\n', x)))
# remove the html feature from body

df['body'] = df['body'].apply(strip_tags)
df['body_cleaned'] = df['body'].apply(lambda x: x.replace('\n',' '))



df['body_lines'] = df['body'].apply(lambda x: len(re.findall('\n', x)))
df = df.drop(['body'],axis = 1)
df.head()
print(df[df['body_lines']==1304]['title'])
print(df.body_lines.describe())

sns.boxplot(df.body_lines)
## Number of words in the text ##

import string

df["num_title_words"] = df["title"].apply(lambda x: len(str(x).split()))

df["num_body_words"] = df["body_cleaned"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

df["num_title_unique_words"] = df["title"].apply(lambda x: len(set(str(x).split())))

df["num_body_unique_words"] = df["body_cleaned"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

df["num_title_chars"] = df["title"].apply(lambda x: len(str(x)))

df["num_body_chars"] = df["body_cleaned"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

df["num_title_stopwords"] = df["title"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

df["num_body_stopwords"] = df["body_cleaned"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))



## Number of punctuations in the text ##

df["num_title_punctuations"] =df['title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

df["num_body_punctuations"] =df['body_cleaned'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Average length of the words in the text ##

df["mean_title_word_len"] = df["title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

df["mean_body_word_len"] = df["body_cleaned"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



# ## Number of title case words in the text ##

# df["num_words_upper"] = df["title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# df["num_words_upper"] = df["title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



# ## Number of title case words in the text ##

# df["num_words_title"] = df["title"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



df['creation_date'] = pd.to_datetime(df['creation_date'])
# Time features

df['weekofyear'] = df['creation_date'].dt.weekofyear

df['month'] = df['creation_date'].dt.month

df['dayofweek'] = df['creation_date'].dt.dayofweek

df['weekend'] = (df.creation_date.dt.weekday >=5).astype(int)

df['hour'] = df['creation_date'].dt.hour
df.head()
from sklearn.model_selection import train_test_split



train_df, test_df, y_train, y_test = train_test_split(df, target, test_size= 0.3, random_state = 2019)
fig, axes = plt.subplots(1,2,figsize=(10,4))

sns.distplot(y_train, ax = axes[0])

axes[0].set_title('training set')

sns.distplot(y_test, ax=axes[1])

axes[1].set_title('test set')
# save_csv

train_df.to_csv('train.csv')

test_df.to_csv('test.csv')

y_train.to_csv('train_target.csv')

y_test.to_csv('test_target.csv')
first = y_train.quantile(0.33)
second = y_train.quantile(0.66)
def map_target_value(x):

    if x < first:

        return 0

    elif x < second:

        return 1

    else:

        return 2
train_df['target_class'] = y_train.apply(map_target_value)
train_df.columns.tolist()
import warnings

warnings.filterwarnings('ignore')
sns.barplot(x = 'target_class', y = 'body_lines', data = train_df)
sns.barplot(x = 'target_class', y = 'num_title_words', data = train_df)
sns.barplot(x = 'target_class', y = 'num_body_words', data = train_df)
sns.barplot(x = 'target_class', y =  'num_title_chars', data = train_df)
sns.barplot(x = 'target_class', y = 'num_body_chars', data = train_df)
sns.barplot(x = 'target_class', y = 'num_title_stopwords', data = train_df)
sns.barplot(x = 'target_class', y = 'num_body_stopwords', data = train_df)
sns.barplot(x = 'target_class', y = 'num_title_punctuations', data = train_df)

plt.title('num_title_punctuations')
sns.boxplot(x = 'target_class', y = 'num_title_punctuations', data = train_df)

plt.title('num_title_punctuations')
sns.barplot(x = 'target_class', y = 'num_body_punctuations', data = train_df)
sns.barplot(x = 'target_class', y = 'weekend', data = train_df)
sns.barplot(x = 'target_class', y =  'dayofweek', data = train_df)
sns.barplot(x = 'target_class', y = 'hour', data = train_df)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb