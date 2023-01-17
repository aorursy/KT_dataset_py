import numpy as np 

import pandas as pd 



espn = pd.read_csv("../input/youtube-video-statistics/ESPN.csv")

espn.head()
from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer

import re



class LemmaTokenizer: # Keeping terms composed of only alphabets

  def __init__(self):

   self.wnl = WordNetLemmatizer()

  def __call__(self, doc):

   return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if re.match(r'(?u)\b[A-Za-z]+\b',t)] 



from sklearn.feature_extraction.text import TfidfVectorizer



def warn(*args, **kwargs): # Turning off the warnings

    pass

import warnings

warnings.warn = warn



vectorizer = TfidfVectorizer(lowercase = False, tokenizer=LemmaTokenizer(), stop_words='english', min_df=0.001, max_df=0.99)

corpus = vectorizer.fit_transform(espn['title'])
from sklearn.decomposition import NMF 

model = NMF(n_components=20, init='random', random_state=0)

corpus_by_topics = model.fit_transform(corpus) 
max_index_in_row = np.argmax(corpus_by_topics, 1)

max_value_in_row = np.amax(corpus_by_topics, 1)



print(max_index_in_row)

print(max_value_in_row)
from scipy import stats

stats.describe(max_value_in_row)
topic = pd.Series(max_index_in_row)

relevance = pd.Series(max_value_in_row)

frame = {'topic':topic, 'relevance':relevance}

df = pd.DataFrame(frame)



espn_derived = pd.concat([df, espn[['likes', 'dislikes', 'comments']]], axis=1, sort=False)

espn_derived.head()
#getting rid of negative values

stats.describe(espn_derived['comments'])

espn_derived['comments'] = np.where(espn_derived['comments'] < 0, 0, espn_derived['comments'] )



#removing outliers

z_scores = stats.zscore(espn_derived)

abs_z_scores = np.abs(z_scores)

filtered_entries = (abs_z_scores < 3).all(axis=1)

espn_derived = espn_derived[filtered_entries]
import seaborn as sns

sns.set(style="white")

import matplotlib.pyplot as plt



#used https://mokole.com/palette.html to generate 20 visually distinct colors

colors=[ "#696969", "#ffe4c4", "#2e8b57", "#8b0000", "#808000", "#000080", "#ff0000", "#ff8c00", "#ffd700", "#ba55d3", "#00ff7f", "#00bfff", "#0000ff", "#adff2f", "#ff00ff", "#f0e68c", "#fa8072", "#dda0dd", "#ff1493", "#7fffd4"]



plt.figure(figsize=(15,8))



# it is not possible to parametrize alpha in Seaborn; so I define three levels of alphas and draw them seperately.

espn_derived["alpha"] = np.where(espn_derived['relevance'] < 0.22, 0.1, np.where(espn_derived['relevance'] < 0.4, 0.4, 0.5))



ax = sns.scatterplot(x="likes", y="dislikes", hue="topic", size="comments", sizes=(1,500), size_norm=(100,4000), alpha=0.1, palette=sns.color_palette(colors), data=espn_derived[espn_derived['alpha']==0.1])



sns.scatterplot(legend=False, ax=ax, x="likes", y="dislikes", hue="topic", size="comments",sizes=(1,500), size_norm=(100,4000), alpha=0.4, palette=sns.color_palette(colors), data=espn_derived[espn_derived['alpha']==0.4])



sns.scatterplot(legend=False, ax=ax, x="likes", y="dislikes", hue="topic", size="comments", sizes=(1,500), size_norm=(100,4000),alpha=0.5, palette=sns.color_palette(colors), data=espn_derived[espn_derived['alpha']==0.5])



plt.plot([0,3000], [0,3000], color='r')



plt.plot([0,14000], [0,1800], color='b')



plt.legend(bbox_to_anchor=(1, 1), loc=2)



ax.legend(ncol=2)



corpus_by_topics_model = model.fit(corpus) # the wieght of each term in every topic       

weight_dict = dict(zip(vectorizer.get_feature_names(), corpus_by_topics_model.components_[4])) # associating the actual terms with their weight



from wordcloud import WordCloud



wc = WordCloud(width=1600, height=800)

wc.generate_from_frequencies(weight_dict)

plt.figure(figsize=(15,8))

plt.imshow(wc) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
weight_dict = dict(zip(vectorizer.get_feature_names(), corpus_by_topics_model.components_[17]))

 

from wordcloud import WordCloud



wc = WordCloud(width=1600, height=800)

wc.generate_from_frequencies(weight_dict)

plt.figure(figsize=(15,8))

plt.imshow(wc) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
weight_dict = dict(zip(vectorizer.get_feature_names(), corpus_by_topics_model.components_[3]))



from wordcloud import WordCloud



wc = WordCloud(width=1600, height=800)

wc.generate_from_frequencies(weight_dict)

plt.figure(figsize=(15,8))

plt.imshow(wc) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
weight_dict = dict(zip(vectorizer.get_feature_names(), corpus_by_topics_model.components_[0]))



from wordcloud import WordCloud



wc = WordCloud(width=1600, height=800)

wc.generate_from_frequencies(weight_dict)

plt.figure(figsize=(15,8))

plt.imshow(wc) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 