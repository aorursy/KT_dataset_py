# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory     
# Any results you write to the current directory are saved as output.

import os

# files is the dataframe that contains file name and full path of all json formatted articles.
files=pd.DataFrame(columns=["name","path"])
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        files=files.append({"name":filename,"path":os.path.join(dirname, filename)}, ignore_index=True)

# df is the dataframe of the metadata file given in the covid 19 open research database
df=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
print(df.shape)
df.head()
files.head(10)
files.shape
indices=files[files.name.str.contains(".json")==False].index 
files.drop(index=indices, inplace=True)
files.shape
files.reset_index(inplace=True)
files.drop("index",axis=1, inplace=True)
files.head(3) # We have name of the all files in json format and the path (including the name of the file).
# necessary modules for keyword search and visualization
import json
from pandas.io.json import json_normalize
import collections
#!pip install wordcloud 
# if not installed install wordcloud by uncommeting the above line
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS) #stopwords are the words like "the", "he'll", "i", etc that won't be used in the search.
key_words=["pseudoknots"]
# This function searches the title and the body of the article for the keywords.
def article_search(path, key_words):
    with open(path) as f:
      article_json = json.load(f)
    title = json_normalize(article_json['metadata'])
    article = title.title[0].lower()+"\n"
    article_title = title.title[0].lower()
    text = json_normalize(article_json['body_text']) 
    for j in range(text.shape[0]):
        article=article+"\n"+text.text[j].lower()
    
    # Check whether article contains all keywords
    key_word_count = len(key_words)
    article_chosen=False
    for key_word in (key_words):
        if key_word in article:
            key_word_count -= 1
    if key_word_count == 0:
        article_chosen=True
        return article_chosen, article_title, article
    else:
        return article_chosen, "", ""
wordcount={} # defining dictionary
article_titles=[] # will be used for listing the selected articles at the end of the notebook
for i in range(files.shape[0]):
    path=files.loc[i,"path"]
    contains_keywords, article_title, article = article_search(path, key_words)
    
    if contains_keywords:
        article_titles.append(article_title)
        for word in article.split():
            word = word.replace(".","")
            word = word.replace(",","")
            word = word.replace("\"","")
            word = word.replace("“","")
            word = word.replace("(","")
            word = word.replace(")","")
            word = word.replace("<","")
            word = word.replace(">","")
            if word not in stopwords:
                if word not in wordcount:
                    wordcount[word] = 1
                else:
                    wordcount[word] += 1
            
word_dict = collections.Counter(wordcount) # final dictionary
len(word_dict) # number of words in the dictionary
start = 0 #starting number of word. 0 is most common. Negative values can be given to search from the rarest words.
end = 100 #ending number of word. It should be greater than start

for word, count in word_dict.most_common()[start:end]:
    print(word, ": ", count)
# instantiate a word cloud object
word_cloud_source=""
for word, count in word_dict.most_common()[start:end]:
    word_cloud_source=word_cloud_source+(word+" ")*count

word_cloud = WordCloud(
    background_color='white',
    max_words=abs(end-start),
    stopwords=stopwords,
    collocations=False
)

# generate the word cloud
word_cloud.generate(word_cloud_source)
# display the cloud
fig = plt.figure()
fig.set_figwidth(30) # set width
fig.set_figheight(25) # set height

plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
indices = df[df.title.str.lower().isin(article_titles)].index
indices
df.loc[indices]
remaining_titles=pd.Series(article_titles)
remaining_indices= remaining_titles[~remaining_titles.isin(df.loc[indices,"title"].str.lower())]
remaining_titles.loc[remaining_indices]
