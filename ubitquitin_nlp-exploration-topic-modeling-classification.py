# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from tqdm import tqdm_notebook as tqdm

import re
genre_df = pd.read_csv("../input/myanimelist-comment-dataset/animeListGenres.csv",error_bad_lines=False, warn_bad_lines=False, sep=",")

review_df = pd.read_csv("../input/myanimelist-comment-dataset/animeReviewsOrderByTime.csv",error_bad_lines=False, warn_bad_lines=False, sep=",")
nRowsRead = 'None' # specify 'None' if want to read whole file

# animeReviewsOrderByTime.csv has 135201 rows in reality, but we are only loading/previewing the first 1000 rows





with open('../input/myanimelist-comment-dataset/animeReviewsOrderByTime.csv', 'r', encoding='utf-8') as f:

    headers = f.readline().replace('"','').replace('\n','').split(',')

    print(headers)

    print('The number of column: ', len(headers))

    dataFormat = dict()

    for header in headers:

        dataFormat[header] = list()



    for idx, line in enumerate(tqdm(f.readlines(), desc='Now parsing... ')):

        

        if idx == 67:

            yee = line

        

        if line != '':

            line = line.replace('\n','')

            indices = [i for i, x in enumerate(line) if x == ',']

            idxStart = 0

            for i in range(len(headers)):

                if i < len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:indices[i] - 1])

                    idxStart = indices[i] + 1

                elif i == len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:-1])

                else:

                    break

        if nRowsRead is not None and nRowsRead == idx + 1:

            print('We read only', nRowsRead, 'lines.')

            break
review_df = pd.DataFrame(dataFormat)
review_df.head()
df = review_df.drop(columns = ["id", "workId", "reviewId", "postTime", "author"])

df.head()
df = df[df.overallRating != '0']

df = df[df.overallRating != '11']
stoplist = set("""for a of what can about don't these them any much each well any these

               doesn't that's when show series character characters how o y e 

               get up out then do only we it's which there because even neither 

               nor my were la the and to in is that i you was it this her with but 

               their its not while they are like very as who be an his her she he him just 

               really on it\'s de que no or are anime have all so has at from by more 

               some one me if would other also into it's will being your than most many

               few none where does while through way such think had good story make say me

               our own why know time off both first around may through things something thing give 

               want many""".split())



print(stoplist)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(

    df['review'], 

    df['overallRating'], 

    test_size=0.1, random_state=2019

)
print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
print(np.unique(y_train))
df2 = df[['workName','overallRating']]
df2['overallRating'] = df2['overallRating'].astype(float)

df2.head()
df3 = df2.groupby(['workName']).mean().reset_index()

df4 = df2.groupby(['workName']).count().reset_index()

print(df3.columns)

print(df4.head())
df4[df4['overallRating']==df4['overallRating'].max()] #max number of ratings 

perfect_anime = df3[df3['overallRating']==df3['overallRating'].max()] #max average rating



perfect_anime.head()

df5 = pd.merge(df4, perfect_anime, on='workName')

df5.head()

frequency_for_10_scorers = list(df5['overallRating_x'])

import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



num_bins = 5

n, bins, patches = plt.hist(frequency_for_10_scorers, num_bins, facecolor='blue', alpha=0.5)

plt.show()
anime_df = df3

anime_df['freq'] = df4['overallRating']

anime_df['averageRating'] = anime_df['overallRating']

anime_df = anime_df.drop(columns = 'overallRating')

anime_df.head()



v = anime_df['freq'] #frequency

m = 100 #minum number of ratings to be taken into consideration 

R = anime_df['averageRating'] #average

C = np.mean(anime_df['averageRating'])#mean score across all anime



anime_df['score'] = (v/(v+m))*R + (m/(v+m))*C



#(v ÷ (v+m)) × R + (m ÷ (v+m)) × C

# R = average for the movie (mean) = (Rating)

# v = number of votes for the movie = (votes)

# m = minimum votes required to be listed in the Top 250 (currently 3000)

# C = the mean vote across the whole report (currently 6.9)
anime_df.sort_values(by=['score'],ascending=False)[0:10]
import gensim
print(df['review'][0:5])
corpus = []

for i in df['review']:

    corpus.append(i)
print(len(corpus))
# remove common words and tokenize

texts = [

    [word for word in document.lower().split() if word not in stoplist]

    for document in corpus

]



texts = [

    [token for token in text if token.isalnum()]

        for text in texts

]



from collections import defaultdict

# remove words that appear only once

frequency = defaultdict(int)

for text in texts:

    for token in text:

        frequency[token] += 1



texts = [

    [token for token in text if frequency[token] > 1]

    for text in texts

]



from gensim import corpora

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

print(texts[0:2])
print(corpus[0])
from gensim import corpora, models, similarities

tfidf = models.TfidfModel(corpus) 

corpus_tfidf = tfidf[corpus]
NUM_TOPICS = 5

lda = models.ldamodel.LdaModel(corpus,num_topics = NUM_TOPICS, id2word=dictionary)

corpus_lda = lda[corpus_tfidf]



lda.show_topics(NUM_TOPICS,5)
import pyLDAvis.gensim

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline





pyLDAvis.enable_notebook()

panel = pyLDAvis.gensim.prepare(lda, corpus_lda, dictionary, mds='mmds')#

panel
pyLDAvis.save_html(panel,'vis.html')
from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 

import matplotlib.colors as mcolors



cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stoplist,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=20,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda.show_topics(formatted=False)



fig, axes = plt.subplots(1, 5, figsize=(15,15), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
df.head()
!pip install BeautifulSoup4

from bs4 import BeautifulSoup  

import re
def review_to_words( raw_review ):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stoplist)                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   
target = df['overallRating']

df = df.drop(columns=['overallRating','storyRating','animationRating','soundRating','enjoymentRating','characterRating'])

df.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(

    df, 

    target, 

    test_size=0.2, random_state=2019

)
# Get the number of reviews based on the dataframe column size

num_reviews = x_train["review"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in x_train["review"]:

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_train_reviews.append( review_to_words( i) )
# Get the number of reviews based on the dataframe column size

num_reviews = x_test["review"].size



# Initialize an empty list to hold the clean reviews

clean_test_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in x_test["review"]:

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_test_reviews.append( review_to_words( i) )
print(clean_train_reviews[0:2])
from keras.preprocessing import text, sequence

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α}•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(clean_train_reviews)
rtok = tokenizer.texts_to_sequences(clean_train_reviews)

rtok = sequence.pad_sequences(rtok, maxlen=256)
from keras.preprocessing import text, sequence

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α}•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(clean_test_reviews)
rtok2 = tokenizer.texts_to_sequences(clean_test_reviews)

rtok2 = sequence.pad_sequences(rtok2, maxlen=256)
print(rtok.shape)

print(rtok[0])
len(rtok)

len(y_train)
y_train = [ int(x) for x in y_train ]

print(type(y_train[0]))

y_test = [ int(x) for x in y_test ]
#convert y_train to a binary classification problem

#1 for good review, 0 for bad review

for i in range(len(y_train)):

    if y_train[i] >=6:

        y_train[i] = 1

    elif y_train[i] <6:

        y_train[i] = 0
#convert y_test to a binary classification problem

#1 for good review, 0 for bad review

for i in range(len(y_test)):

    if y_test[i] >=6:

        y_test[i] = 1

    elif y_test[i] <6:

        y_test[i] = 0
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(rtok,y_train)
print(len(rtok2))
print(len(y_test))
y_pred = clf.predict(rtok2)



match = (y_pred == y_test)

acc = sum(match)/len(match)
print("accuracy is: ", acc)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

import seaborn as sns 

sns.heatmap(cm,cmap="YlGnBu",linewidths=.5, annot=True)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(

    df, 

    target, 

    test_size=0.2, random_state=2019

)
# Get the number of reviews based on the dataframe column size

num_reviews = x_train["review"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in x_train["review"]:

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_train_reviews.append( review_to_words( i) )
# Get the number of reviews based on the dataframe column size

num_reviews = x_test["review"].size



# Initialize an empty list to hold the clean reviews

clean_test_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in x_test["review"]:

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_test_reviews.append( review_to_words( i) )
from keras.preprocessing import text, sequence

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α}•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(clean_train_reviews)
rtok = tokenizer.texts_to_sequences(clean_train_reviews)

rtok = sequence.pad_sequences(rtok, maxlen=256)
from keras.preprocessing import text, sequence

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α}•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

tokenizer.fit_on_texts(clean_test_reviews)

rtok2 = tokenizer.texts_to_sequences(clean_test_reviews)

rtok2 = sequence.pad_sequences(rtok2, maxlen=256)
y_train = [ int(x) for x in y_train ]

print(type(y_train[0]))

y_test = [ int(x) for x in y_test ]
import numpy as np

import math

from matplotlib import pyplot as plt



bins = np.linspace(math.ceil(min(y_train)), 

                   math.floor(max(y_train)),

                   10) # fixed number of bins



plt.xlim([min(y_train), max(y_train)])



plt.hist(y_train, bins=bins, alpha=0.5)

plt.title('Y_train (fixed number of bins)')

plt.xlabel('variable X (10 evenly spaced bins)')

plt.ylabel('count')



plt.show()
import numpy as np

import math

from matplotlib import pyplot as plt



bins = np.linspace(math.ceil(min(y_test)), 

                   math.floor(max(y_test)),

                   10) # fixed number of bins



plt.xlim([min(y_test), max(y_test)])



plt.hist(y_test, bins=bins, alpha=0.5)

plt.title('Y_test (fixed number of bins)')

plt.xlabel('variable X (10 evenly spaced bins)')

plt.ylabel('count')



plt.show()
#convert y_train to a binary classification problem

#1 for good review, 0 for bad review

for i in range(len(y_train)):

    if y_train[i] <=5:

        y_train[i] = 0 #very bad

    elif y_train[i] == 6 or y_train[i] == 7 or y_train[i] == 8 or y_train[i] == 9:

        y_train[i] = 1 #average

    elif y_train[i] == 10:

        y_train[i] = 2 #excellent
#convert y_train to a binary classification problem

#1 for good review, 0 for bad review

for i in range(len(y_test)):

    if y_test[i] <=5:

        y_test[i] = 0 #very bad

    elif y_test[i] == 6 or y_test[i] == 7 or y_test[i] == 8 or y_test[i] == 9:

        y_test[i] = 1 #average

    elif y_test[i] == 10:

        y_test[i] = 2 #excellent
uni = np.unique(y_train)

for i in uni:

    print("unique ", i, ": ", y_train.count(i))
print(type(y_train))
uni = np.unique(y_test)

for i in uni:

    print("unique ", i, ": ", y_test.count(i))
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(rtok,y_train)
y_pred = clf.predict(rtok2)



match = (y_pred == y_test)

acc = sum(match)/len(match)

print("accuracy is: ", acc)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,cmap="YlGnBu",linewidths=.5, annot=True)
importances = clf.feature_importances_





print("Top 10 Features:")



for f in range(10):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    print([key for key, value in tokenizer.word_index.items() if value == indices[f]])