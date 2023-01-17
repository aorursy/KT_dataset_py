import matplotlib.pyplot as plt

import plotly.figure_factory as ff

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os

import re

from tqdm import tqdm

tqdm.pandas()





from wordcloud import WordCloud, STOPWORDS

from PIL import Image

from kaggle_datasets import KaggleDatasets

from colorama import Fore, Back, Style, init

import plotly.graph_objects as go



import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
dir = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification'



train_set1 = pd.read_csv(os.path.join(dir, 'jigsaw-toxic-comment-train.csv'))

train_set2 = pd.read_csv(os.path.join(dir, 'jigsaw-unintended-bias-train.csv'))

train_set2.toxic = train_set2.toxic.round().astype(int)



valid = pd.read_csv(os.path.join(dir, 'validation.csv'))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
# Combine train1 with a subset of train2

train = pd.concat([

    train_set1[['comment_text', 'toxic']],

    train_set2[['comment_text', 'toxic']].query('toxic==1'),

    train_set2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)

])
print(train.shape)

train.head()
print(valid.shape)

valid.head()
print(test.shape)

test.head()
print("Check for missing values in Train dataset")

null_check=train.isnull().sum()

print(null_check)

print("Check for missing values in Validation dataset")

null_check=valid.isnull().sum()

print(null_check)

print("Check for missing values in Test dataset")

null_check=test.isnull().sum()

print(null_check)

print("filling NA with \"unknown\"")

train["comment_text"].fillna("unknown", inplace=True)

valid["comment_text"].fillna("unknown", inplace=True)
for i in range(3):

    print(f'[Comment {i+1}]\n', train['comment_text'][i])

    print()
print("Toxic comments:")

print(train[train.toxic==1].iloc[:10,0])
#print(train.toxic.value_counts())

#print(valid.toxic.value_counts())



print("Train set")

print("Toxic comments = ",len(train[train['toxic']==1]))

print("Non-toxic comments = ",len(train[train['toxic']==0]))



print("\nValidation set")

print("Toxic comments = ",len(valid[valid['toxic']==1]))

print("Non-toxic comments = ",len(valid[valid['toxic']==0]))
sns.set(style="darkgrid")



f = plt.figure(figsize=(20,5))

f.add_subplot(1,2,1)

sns.countplot(train_set1.toxic)

plt.title('Toxic Comment Distribution in Train Set 1')

f.add_subplot(1,2,2)

sns.countplot(train_set2.toxic)

plt.title('Toxic Comment Distribution in Train Set 2')
f = plt.figure(figsize=(20,5))

f.add_subplot(1,2,1)

sns.countplot(train.toxic)

plt.title('Toxic Comment Distribution in Train Set')

f.add_subplot(1,2,2)

sns.countplot(valid.toxic)

plt.title('Toxic Comment Distribution in Validation Set')
print(valid.lang.value_counts())

print(test.lang.value_counts())
f = plt.figure(figsize=(20,5))

f.add_subplot(1,2,1)

sns.countplot(valid.lang)

plt.title('Langauages in Validation Set')

f.add_subplot(1,2,2)

sns.countplot(test.lang)

plt.title('Languages in Final Test Set')
stopword=set(STOPWORDS)



#wordcloud of all comments

plt.figure(figsize=(10,10))

text = train.comment_text.values

wc = WordCloud(background_color="black",max_words=2000,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Common words in All Comments", fontsize=16)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)
#non-toxic wordcloud

clean_mask=np.array(Image.open("../input/imagesforkernal/safe-zone.png"))

clean_mask=clean_mask[:,:,1]



plt.figure(figsize=(20,20))

plt.subplot(121)

subset = train.query("toxic == 0")

text = subset.comment_text.values

wc = WordCloud(background_color="black",max_words=1000,mask=clean_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Common words in non-Toxic Comments", fontsize=16)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)



#toxic wordcloud

clean_mask=np.array(Image.open("../input/imagesforkernal/swords.png"))

clean_mask=clean_mask[:,:,1]



plt.subplot(122)

subset = train.query("toxic == 1")

text = subset.comment_text.values

wc = WordCloud(background_color="black",max_words=1000,mask=clean_mask,stopwords=stopword)

wc.generate(" ".join(text))

plt.axis("off")

plt.title("Common words in Toxic Comments", fontsize=16)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)



plt.show()
nums_1 = train[train['toxic']==1]['comment_text'].sample(frac=0.1).str.len()

nums_2 = train[train['toxic']==0]['comment_text'].sample(frac=0.1).str.len()



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Number of characters per comment vs. Toxicity", xaxis_title="No of characters per comment", 

                  yaxis_title="Distribution of observations (%)", template="simple_white")

fig.show()
nums_1 = train[train['toxic']==1]['comment_text'].sample(frac=0.1).str.split().str.len()

nums_2 = train[train['toxic']==0]['comment_text'].sample(frac=0.1).str.split().str.len()



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Number of words per comment vs. Toxicity", xaxis_title="No of words per comment", 

                  yaxis_title="Distribution of observations (%)", template="simple_white")

fig.show()
SIA = SentimentIntensityAnalyzer()



def polarity(x):

    if type(x) == str:

        return SIA.polarity_scores(x)

    else:

        return 1000

    

train["polarity"] = train["comment_text"].progress_apply(polarity)
print(train[train.toxic==1].iloc[4,0])



polarity(train[train.toxic==1].iloc[4,0])
fig = go.Figure(go.Histogram(x=[pols["neg"] for pols in train["polarity"] if pols["neg"] != 0], marker=dict(color='red')))

fig.update_layout(xaxis_title="Negative sentiment", title_text="Negative sentiment", 

                  yaxis_title="Number of comments", template="simple_white")
train["negativity"] = train["polarity"].apply(lambda x: x["neg"])



nums_1 = train.sample(frac=0.1).query("toxic == 1")["negativity"]

nums_2 = train.sample(frac=0.1).query("toxic == 0")["negativity"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-Toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Negative Sentiment vs. Toxicity", xaxis_title="Negative Sentiment", 

                  yaxis_title="Number of comments", template="simple_white")

fig.show()
fig = go.Figure(go.Histogram(x=[pols["pos"] for pols in train["polarity"] if pols["pos"] != 0], marker=dict(color='green')))

fig.update_layout(xaxis_title="Positive sentiment", title_text="Positive sentiment", 

                  yaxis_title="Number of comments", template="simple_white")
train["positivity"] = train["polarity"].apply(lambda x: x["pos"])



nums_1 = train.sample(frac=0.1).query("toxic == 1")["positivity"]

nums_2 = train.sample(frac=0.1).query("toxic == 0")["positivity"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-Toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Positive Sentiment vs. Toxicity", xaxis_title="Positive Sentiment", 

                  yaxis_title="Number of comments", template="simple_white")

fig.show()
fig = go.Figure(go.Histogram(x=[pols["neu"] for pols in train["polarity"] if pols["neu"] != 1], marker=dict(color='grey')))

fig.update_layout(xaxis_title="Neutral sentiment", title_text="Neutral sentiment", 

                  yaxis_title="Number of comments", template="simple_white")
train["neutral"] = train["polarity"].apply(lambda x: x["neu"])



nums_1 = train.sample(frac=0.1).query("toxic == 1")["neutral"]

nums_2 = train.sample(frac=0.1).query("toxic == 0")["neutral"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-Toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Neutral Sentiment vs. Toxicity", xaxis_title="Neutral Sentiment", 

                  yaxis_title="Number of comments", template="simple_white")

fig.show()
fig = go.Figure(go.Histogram(x=[pols["compound"] for pols in train["polarity"] if pols["compound"] != 0], marker=dict(color='yellow')))

fig.update_layout(xaxis_title="Compound sentiment", title_text="Compound sentiment", 

                  yaxis_title="Number of comments", template="simple_white")
train["compound"] = train["polarity"].apply(lambda x: x["compound"])



nums_1 = train.sample(frac=0.1).query("toxic == 1")["compound"]

nums_2 = train.sample(frac=0.1).query("toxic == 0")["compound"]



fig = ff.create_distplot(hist_data=[nums_1, nums_2],

                         group_labels=["Toxic", "Non-Toxic"],

                         colors=["red", "green"], show_hist=False)



fig.update_layout(title_text="Compound Sentiment vs. Toxicity", xaxis_title="Compound Sentiment", 

                  yaxis_title="Number of comments", template="simple_white")

fig.show()