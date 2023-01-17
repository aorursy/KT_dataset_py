import json
import re
from random import seed, randrange
from math import log
from sklearn.utils import shuffle

import nltk
nltk.download("stopwords")
data_df = pd.read_csv('data/bgg-13m-reviews.csv',index_col=0)
data_df.head()
data_df = data_df.dropna()
data_df.head()
reviews_df = data_df[['rating','comment']]
reviews_df.head()
reviews_df = shuffle(reviews_df)
reviews_df.head()
def segmentation(str):
    words = re.sub('[^a-zA-Z]',' ', str).lower().split() # Remove non-alphabetic characters
    stop_words = (nltk.corpus.stopwords.words('english')) # Remove stopwords
    words =  [x for x in words if x not in stop_words]
    return words
x = [segmentation(review) for review in reviews_df['comment']]
y = [round(r) for r in reviews_df['rating']]
x_train, y_train, x_test, y_test = x, y, [], []
test_size = int(len(x)*0.01)

seed(1)

for _ in range(test_size):
    random_index = randrange(len(x_train))
    x_test.append(x_train.pop(random_index))
    y_test.append(y_train.pop(random_index))

print('Size of Train Set: ', len(x_train))
print('Size of Test Set: ', len(x_test))
# Get all the words in the training set non-repeatedly and record the index of each word
words_index_dict = dict()
index = 0
for rating in x_train:
    for word in rating:
        if word not in words_index_dict:
            words_index_dict[word]=index
            index+=1
tf={}
idf = [0 for _ in range(len(words_index_dict))]
for review_index, review in enumerate(x_train):
    review_counts = pd.value_counts(review)
    for word_index, word in enumerate(review):
        if word in words_index_dict:
            tf[(review_index,words_index_dict[word])] = review_counts[word]/len(review)
            idf[words_index_dict[word]]+=1

idf = [log(len(x_train)/(cont+1)) for cont in idf]
for key in tf:
    tf[key]*=idf[key[1]]
tfidf=dict()
for rating in range(11):
    tfidf[rating]=[0 for _ in range(len(words_index_dict))]
for key, value in tf.items():
    label = y_train[key[0]]
    word_index = key[1]
    tfidf[label][word_index]+=value
for i in range(len(tfidf)):
    row_sum = sum(tfidf[i])
    tfidf[i]=[x/row_sum for x in tfidf[i]]
label_count = [0 for _ in range(11)] + [len(x_train)]
for rating in y_train:
    label_count[rating]+=1
def count_value(list):
    value_count=dict()
    for x in list:
        if x not in value_count:
            value_count[x]=0
        value_count[x]+=1
    return value_count

def predict(review):
    probability = []
    words_in_review_set = set(review)
    words_counts = count_value(review)
    for label in range(11):
        prob = 0
        for word in words_in_review_set:
            if word not in words_index_dict:
                continue
            prob+=log(tfidf[label][words_index_dict[word]]*words_counts[word]+1)
        prob *= label_count[label]/label_count[-1]
        probability.append(prob)
    return probability.index(max(probability))
correct = 0
for i in range(len(x_test)):
    if predict(x_test[i]) == y_test[i]:
        correct+=1
accuracy = correct/len(x_test)
print("Accuracy = ", accuracy)