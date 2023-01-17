# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
def print_word_cloud_for_topic(topic):
    print('getting wc for topic %s' % topic)
    train_pos = train[ train[topic] >= 0.75]
#     train_pos = train_pos['comment_text']
    print('positives shape %s' % str(train_pos.shape))
    train_neg = train[ train[topic] <= 0.25]
#     train_neg = train_neg['comment_text']
    print('negatives shape %s' % str(train_neg.shape))

    def wordcloud_draw(data, color = 'black'):
        words = ' '.join(data)
        cleaned_words = " ".join([word for word in words.split()
                                if not hasNumbers(word)
                                ])
        wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color=color,
                          width=2500,
                          height=2000,
                          max_words=100
                         ).generate(cleaned_words)
        
        plt.figure(1,figsize=(13, 13))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
        return set(wordcloud.words_)

    print("Positive words for topic %s" % topic)
    pos_words = wordcloud_draw(train_pos['comment_text'],'white')
    print("Negative words for topic %s" % topic)
    neg_words = wordcloud_draw(train_neg['comment_text'])
    
    return pos_words, neg_words, train_pos, train_neg
toxic_pos_words, toxic_neg_words, train_pos, train_neg = print_word_cloud_for_topic('identity_hate')
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def get_sentence_grams(sen):
    sen = sen.split()
    grams1 = find_ngrams(sen, 1)
    grams2 = find_ngrams(sen, 2)
    return ([(g[0]) for g in grams1 if len(g) == 1]) + ([(g[0] + ' ' + g[1]) for g in grams2 if len(g) == 2])

print('total pos words: %s' %len(toxic_pos_words))
print('total neg words: %s' %len(toxic_neg_words))

toxic_pos_words = [w.lower() for w in toxic_pos_words]

# get filtered negatives examples
df_train_neg = pd.DataFrame(train_neg)
print('shape before filter (neg): %s '% df_train_neg.shape[0])
train_neg_filtered = df_train_neg[df_train_neg['comment_text'].apply(lambda x: len(set(get_sentence_grams(x.lower())).intersection(toxic_pos_words)) > 0)]
print('shape after filter (neg): %s '% train_neg_filtered.shape[0])

# get filtered positive examples
df_train_pos = pd.DataFrame(train_pos)
print('shape before filter (pos): %s '% df_train_pos.shape[0])
train_pos_filtered = df_train_pos[df_train_pos['comment_text'].apply(lambda x: len(set(get_sentence_grams(x.lower())).intersection(toxic_pos_words)) > 0)]
print('shape after filter (pos): %s '% train_pos_filtered.shape[0])


pd.set_option('display.max_colwidth', -1)

train_neg_filtered.to_csv('train_filtered_identity_hate_negatives.csv', index=False)
train_pos_filtered.to_csv('train_filtered_identity_hate_positives.csv', index=False)

