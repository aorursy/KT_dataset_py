# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv', parse_dates=["date"])

df_test = pd.read_csv('../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv', parse_dates=["date"])
print(df.shape)

print(df_test.shape)
df['month'] = df.date.apply(lambda i: i.month)

df['day'] = df.date.apply(lambda i: i.day)

df['year'] = df.date.apply(lambda i: i.year)



df_test['month'] = df_test.date.apply(lambda i: i.month)

df_test['day'] = df_test.date.apply(lambda i: i.day)

df_test['year'] = df_test.date.apply(lambda i: i.year)



df['reviewLength'] = df.review.apply(lambda x: len(x.split()))

df_test['reviewLength'] = df_test.review.apply(lambda x: len(x.split()))
df.dtypes
fig, ax = plt.subplots(3,1, figsize=(15,15))



plt.style.use('seaborn')



# Score by year

ax[0].plot(df.groupby('year').rating.mean())

ax[0].set_ylabel('Average Rating')

ax[0].set_xlabel('Year')

ax[0].set_title('Average Rating vs. Year')

# plt.show()



ax[1].plot(df.groupby('month').rating.mean())

ax[1].set_ylabel('Average Rating')

ax[1].set_xlabel('Month')

ax[1].set_xticks(range(1,13))

ax[1].set_title('Average Rating vs. Month')



ax[2].plot(df.groupby('day').rating.mean())

ax[2].set_ylabel('Average Rating')

ax[2].set_xlabel('Day')

ax[2].set_xticks(range(1,32))

ax[2].set_title('Average Rating vs. Month')



plt.show()
# Create some mock data



# plt.figure(figsize=(12,8))

plt.style.use('fivethirtyeight')



fig, ax1 = plt.subplots(figsize=(9,6))

t = range(1,11)



color = 'tab:red'

ax1.set_xlabel('Rating')

ax1.set_ylabel('Average usefulCount', color=color)

ax1.plot(t, df.groupby('rating').usefulCount.mean(), color=color, alpha=0.8)

ax1.set_xticks(range(1,11))

ax1.tick_params(axis='y', labelcolor=color)

ax1.set_title('Average usefulCount vs. Rating\n Average Review Length vs. Rating')



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('Average Review Length', color=color)  # we already handled the x-label with ax1

ax2.plot(t, df.groupby('rating').reviewLength.mean(), color=color, alpha=0.8)

ax2.tick_params(axis='y', labelcolor=color)



fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
# Make side-by-side barplot

plt.figure(figsize=(9,6))

plt.style.use('fivethirtyeight')





def make_hist(data1, data2, density=True) -> None:

    """make hist plot of useful counts given data, plot fraction if density is True"""

#     print(data.shape)

    total_counts1 = data1.usefulCount.sum()

    total_counts2 = data2.usefulCount.sum()

#     print(total_counts)

    

    sum_counts1 = data1.groupby('rating').usefulCount.sum()

    frac_counts1 = sum_counts1/total_counts1

    frac1_6_10 = data1.groupby('rating').usefulCount.sum()[5:].sum()/total_counts1

    sum_counts2 = data2.groupby('rating').usefulCount.sum()

    frac_counts2 = sum_counts2/total_counts2

    frac2_6_10 = data2.groupby('rating').usefulCount.sum()[5:].sum()/total_counts2

#     print(sum_counts)

#     print(sum_counts.sum())



    plt.bar([i-0.2 for i in range(1,11)],frac_counts1,width=0.4,  label='Total Average', color = 'tab:red', alpha=0.85)

    plt.bar([i+0.2 for i in range(1,11)],frac_counts2,width=0.4,  label='Depression Average', color = 'tab:blue', alpha=0.85)

    

    plt.annotate(str(int(frac2_6_10*100))+'%',

            xy=(9, 0.43), xycoords='data',

            xytext=(5.75, 0.1), textcoords='data',

            size=20, va="center", ha="center",

            arrowprops=dict(arrowstyle="simple",

                            connectionstyle="arc3,rad=0.2",

                            color='tab:blue', alpha=0.85),

            )

    plt.annotate(str(int(frac1_6_10*100))+'%',

            xy=(8, 0.43), xycoords='data',

            xytext=(5.75, 0.17), textcoords='data',

            size=20, va="center", ha="center",

            arrowprops=dict(arrowstyle="simple",

                            connectionstyle="arc3,rad=0.2",

                            color='tab:red', alpha=0.85),

            )

    

    

    plt.xticks(range(1,11))

    plt.xlabel('Rating')

    plt.legend(['All Drugs','Depression Drugs'])

    plt.ylabel('Fraction of usefulCounts')

    plt.title('Distribution of usefulCounts\n Total vs. Depression')

    plt.tight_layout() 

    plt.show()



make_hist(df, df.loc[df.condition=='Depression',:])
# Before

print(df.review[5])
##### Step 1. deal with html special symbols

# https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python



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



df['review'] = df.review.apply(lambda x: strip_tags(x))

df_test['review'] = df_test.review.apply(lambda x: strip_tags(x))
# After

print(df.review[5])
# Step 2. Remove contractions



!pip install textsearch

!pip install contractions



import contractions # need textsearch as well



df['review'] = df['review'].map(lambda x: contractions.fix(x))

df_test['review'] = df_test['review'].map(lambda x: contractions.fix(x))



print(df.review[5])
# Step 3. Convert to lowercase



df['review'] = df['review'].map(lambda x: x.lower())

df_test['review'] = df_test['review'].map(lambda x: x.lower())



# Step 4. Remove numbers

import re



df['review'] = df['review'].map(lambda x: re.sub(r'\d+', '', x))

df_test['review'] = df_test['review'].map(lambda x: re.sub(r'\d+', '', x))



# Step 5. Remove punctuation

df['review'] = df['review'].map(lambda x: re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', '', x))

df_test['review'] = df_test['review'].map(lambda x: re.sub('[!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', '', x))



df['review'] = df['review'].map(lambda x: x.split('"')[1])

df_test['review'] = df_test['review'].map(lambda x: x.split('"')[1])





print(df.review[5])
# Step 6. Lemmatize

## This step is time-consuming



from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

lemmatizer=WordNetLemmatizer()



df['review'] = df['review'].map(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="v") for word in word_tokenize(x)))

df_test['review'] = df_test['review'].map(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="v") for word in word_tokenize(x)))



df['review'] = df['review'].map(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="n") for word in word_tokenize(x)))

df_test['review'] = df_test['review'].map(lambda x: ' '.join(lemmatizer.lemmatize(word, pos="n") for word in word_tokenize(x)))



print(df.review[5])
# Step 8. Removing stop words

from nltk.corpus import stopwords

stopword = stopwords.words('english')



# text = “This is a Demo Text for NLP using NLTK. Full form of NLTK is Natural Language Toolkit”

# word_tokens = word_tokenize(text)

# removing_stopwords = [word for word in word_tokens if word not in stopword]

# print (removing_stopwords)



df['review'] = df['review'].map(lambda x: ' '.join(word for word in word_tokenize(x) if word not in stopword))

df_test['review'] = df_test['review'].map(lambda x: ' '.join(word for word in word_tokenize(x) if word not in stopword))



print(df.review[5])
# Convert rating to 1-5

df.rating = (df.rating.values+1)//2

df_test.rating = (df_test.rating.values+1)//2
from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report



log_losses = []

y_test = pd.get_dummies(df_test.rating)

y_pred = y_test.copy()
# Random guessing

y_pred.iloc[:,:] = 0.2

log_losses.append(log_loss(y_test, y_pred))
clf = OneVsRestClassifier(LogisticRegression(solver = 'liblinear'))

X_train = df.loc[:,['year','month','reviewLength']]

y_train = df.rating

X_test = df_test.loc[:,['year','month','reviewLength']]

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)

log_losses.append(log_loss(y_test, y_pred))
X_train = df.review

y_train = df.rating



# create the pipeline object

pl1 = Pipeline([

        ('vectorizer', CountVectorizer()),

        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])





# fit the pipeline to our training data

%time pl1.fit(X_train, y_train)
y_pred = pl1.predict_proba(df_test.review)

log_losses.append(log_loss(y_test, y_pred))
from sklearn.metrics import classification_report
# set a reasonable number of features before adding interactions

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

# create the pipeline object

pl2 = Pipeline([

        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),

        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))])



# fit the pipeline to our training data

%time pl2.fit(X_train, y_train)



y_pred = pl2.predict_proba(df_test.review)

log_losses.append((log_loss(y_test, y_pred)))

print(log_losses)
# create the pipeline object

pl3 = Pipeline([

        ('vectorizer', CountVectorizer(ngram_range=(2, 4))),

        ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced')))])



%time pl3.fit(X_train, y_train)



y_pred = pl3.predict_proba(df_test.review)

log_losses.append((log_loss(y_test, y_pred)))

print(log_losses)
print(classification_report(pl3.predict(df_test.review), df_test.rating))
labels = ['Random Guess', 'LogisticRegression \nwith numeric variables', 

          'LogisticRegression \nwith text', 'LogisticRegression \nwith text(ngram=(1,2))', 'LogisticRegression \nwith text(ngram=(2,4))' ]
plt.style.use('fivethirtyeight')

# plt.figure()

fig, ax = plt.subplots(figsize=(10,10))

ind = np.arange(len(labels))

width = 0.75



ax.barh(ind,log_losses[::-1], width)

ax.set_yticks(ind)

ax.set_yticklabels(labels[::-1])

plt.xlabel('Log loss values')



plt.annotate('-65%',

            xy=(0.7, 0.3), xycoords='data',

            xytext=(1.6, 3.2), textcoords='data',

            size=20, va="center", ha="center",

            arrowprops=dict(arrowstyle="simple",

                            connectionstyle="arc3,rad=-0.2",

                            color='tab:blue', alpha=0.85),

            )



# plt.ylabel('Methods')

plt.title('Log loss values under different methods')

# plt.tight_layout() 

plt.show()
import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



all = pd.concat([df.iloc[:,0:11], df_test])
text = " ".join(review for review in all.review)



print ("There are {} words in the combination of all review.".format(len(text)))



# Create and generate a word cloud image:

wordcloud = WordCloud(width=800, height=300,background_color="white").generate(text)



plt.figure(figsize=(20,5))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Birth Control reviews



text = " ".join(review for review in all[all.condition=='Birth Control'].review)



print ("There are {} words in the combination of all review.".format(len(text)))



wordcloud = WordCloud(width=800, height=300,background_color="white").generate(text)



plt.figure(figsize=(20,5))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Birth Control reviews



text = " ".join(review for review in all[all.condition=='Depression'].review)



print ("There are {} words in the combination of all review.".format(len(text)))



wordcloud = WordCloud(width=800, height=300,background_color="white").generate(text)



plt.figure(figsize=(20, 5))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
verbs = ['take', 'get', 'go', 'work', 'start', 'day', 'days', 'feel', 'time', 'year', 'month', 'years', 'months', 'mg', 'like', 'would', 'use',\

         'first', 'try', 'doctor', 'pill', 'week', 'dose', 'lb', 'hour', 'one', 'never']



# remove drug names

new_set = set(all.drugName.tolist())

new_set = {i.lower() for i in new_set}



new_list=[]

for i in new_set:

    good = i.split()

    new_list.extend(good)

new_set=set(new_list)



all = all.copy()

all['review'] = all.review.map(lambda x: ' '.join(word for word in word_tokenize(x) if word not in verbs))

all['review'] = all.review.map(lambda x: ' '.join(word for word in word_tokenize(x) if word not in new_set))
%%time



# import warnings

# warnings.simplefilter("ignore", DeprecationWarning)



# Load the LDA model from sk-learn

from sklearn.decomposition import LatentDirichletAllocation as LDA

 

# Helper function

def print_topics(model, count_vectorizer, n_top_words):

    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        print("\nTopic #%d:" % topic_idx)

        print(" ".join([words[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
count_vectorizer = CountVectorizer()

count_data = count_vectorizer.fit_transform(all[all.condition=='Birth Control'].review)



number_topics = 3

number_words = 8



lda = LDA(n_components=number_topics, n_jobs=-1, random_state=1)

lda.fit(count_data)



# Print the topics found by the LDA model

print("Topics found via LDA:")

print_topics(lda, count_vectorizer, number_words)
count_data = count_vectorizer.fit_transform(all[all.condition=='Depression'].review)



number_topics = 3

number_words = 8



lda = LDA(n_components=number_topics, n_jobs=-1, random_state=55566)

lda.fit(count_data)



# Print the topics found by the LDA model

print("Topics found via LDA:")

print_topics(lda, count_vectorizer, number_words)