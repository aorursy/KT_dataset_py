import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re
from PIL import Image
import requests
from io import BytesIO
%matplotlib inline
data = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
data.head()
data.describe()
data.isnull().sum()
reviewer = data.dropna(subset = ['Title', 'Review Text'])
plt.figure(figsize = (16, 9))
sns.countplot(x = 'Age', data = reviewer)
plt.xticks(rotation = 60)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Number of reviews by age')
plt.show()
tmp_reviewer = reviewer['Age'].value_counts()

print(tmp_reviewer.idxmax(), 'is the age group who wrote the most reviews with', tmp_reviewer.max(), 'reviews')
print(tmp_reviewer.idxmin(), 'is the age group who wrote the least reviews with', tmp_reviewer.min(), 'reviews')
plt.figure(figsize = (16, 9))
sns.countplot(x = 'Rating', data = data)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Distribution of rating')
plt.show()
rating_deparment = data.groupby('Department Name', as_index = False).mean()
sns.barplot(x = 'Department Name', y = 'Rating', data = rating_deparment)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title('Average rating by deparment')
plt.show()
tmp_data = (data[data['Rating']==1])['Department Name'].value_counts()
print(tmp_data.idxmax(), 'is the deparment with the most 1-star rating with', tmp_data.max())
# prepare text data

# firstly, we concatenate all review text in to a string and also lower all of them. They are separated by ' '
all_words = data['Review Text'].str.lower().str.cat(sep = ' ')

# removes punctuation, numbers and returns list of words
all_words = re.sub('[^A-Za-z]+', ' ', all_words)

# remove all stopwords and numeric from the text
stop_words = set(stopwords.words('english'))
tokens = nltk.word_tokenize(all_words)
no_stop_words = []
for w in tokens:
    if (w not in stop_words) and (w.isdigit() is False):
        no_stop_words.append(w)
# count the frequency of word
word_list = nltk.FreqDist(no_stop_words)
word_df = pd.DataFrame(word_list.most_common(100), columns=['Word', 'Frequency'])

plt.figure(figsize = (10, 5))
sns.barplot(x = 'Word', y = 'Frequency', data = word_df[:10])
plt.title('Most 10 common words')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.show()
wordcloud = WordCloud(max_words=200, background_color="white").generate(all_words)
# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()