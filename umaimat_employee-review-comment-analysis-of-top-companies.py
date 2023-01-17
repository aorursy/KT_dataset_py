import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/employee_reviews.csv")
df.shape
df.info()
df.isnull().sum()
df.head()
df.groupby('company').size()
df.groupby(['overall-ratings','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
df.groupby(['work-balance-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
df.groupby(['culture-values-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
df.groupby(['carrer-opportunities-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
df.groupby(['comp-benefit-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
df.groupby(['senior-mangemnet-stars','company']).size().unstack().plot.bar(subplots=True, rot=0, figsize=(15,20), legend=True)
def plot_heatmap(dataCol1, dataCol2, heading):

    grpby = df.groupby([dataCol1, dataCol2]).size()

    pct = grpby.groupby(level=1).apply(lambda x:100 * x / float(x.sum()))

    unstk_data = pct.unstack()

    fig, ax = plt.subplots()

    sns.heatmap(unstk_data, annot=True, linewidths=.5, ax=ax, cmap='YlGn')

    ax.set_title(heading)

    fig.tight_layout()

    plt.show()
plot_heatmap('overall-ratings','company', 'Overall-ratings in Companies in %' )
plot_heatmap('work-balance-stars','company', 'Work-Life-Balance in Companies in %' )
plot_heatmap('culture-values-stars','company', 'Culture Values in Companies in %' )
plot_heatmap('carrer-opportunities-stars','company', 'Career Opportunities in Companies in %' )
plot_heatmap('comp-benefit-stars','company', 'Compensation Benefits in Companies in %' )
plot_heatmap('senior-mangemnet-stars','company', 'Senior-Management Ratings in Companies in %' )
#Define a function to get rid of stopwords present in the messages

from nltk.corpus import stopwords

import string



def message_text_process(mess):

    # Check characters to see if there are punctuations

    no_punctuation = [char for char in mess if char not in string.punctuation]

    # now form the sentence.

    no_punctuation = ''.join(no_punctuation)

    # Now eliminate any stopwords

    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]    
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation



vectorizer = TfidfVectorizer(analyzer=message_text_process)
n_top_words = 20

lda = LatentDirichletAllocation()



def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

    

def find_top_comments(corpus):

    tfidf_transformer = vectorizer.fit_transform(corpus)

    tf_feature_names = vectorizer.get_feature_names()    

    lda.fit_transform(tfidf_transformer)

    print_top_words(lda, tf_feature_names, n_top_words)
find_top_comments(df['pros'])
find_top_comments(df['cons'])
find_top_comments(df[df.summary.notnull()].summary)
df[df.summary.notnull()].summary.head(25)
from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 



def generate_word_cloud(text):  

    comment_words = ' '

    stopwords = set(STOPWORDS) 



    # iterate through the csv file 

    for val in text: 



        # typecaste each val to string 

        val = str(val) 



        # split the value 

        tokens = val.split() 



        # Converts each token into lowercase 

        for i in range(len(tokens)): 

            tokens[i] = tokens[i].lower() 



        for words in tokens: 

            comment_words = comment_words + words + ' '

        

    wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

    

    # plot the WordCloud image                        

    plt.figure(figsize = (8, 8), facecolor = None) 

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show() 
grouped = df.groupby('company')



for name,group in grouped:

    print (name)

    generate_word_cloud(group['summary'])

    print('cons')

    generate_word_cloud(group['cons'])

    print('pros')

    generate_word_cloud(group['pros'])

    print('Advice to Management')

    generate_word_cloud(group['advice-to-mgmt'])

    