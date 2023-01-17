# Removing warnings
import warnings
warnings.filterwarnings('ignore')

# Importing files
import json
from pandas.io.json import json_normalize

import numpy as np
import pandas as pd
from scipy.sparse import hstack

# Visualisation
import matplotlib.pyplot as plt

# Train/Test split
from sklearn.model_selection import train_test_split

# Preprocessing
import re
import string
import gensim

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Models
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV

# Evaluation
from sklearn.metrics import roc_auc_score
train = json.load(open('../input/random-acts-of-pizza/train.json'))
train = json_normalize(train)
test = json.load(open('../input/random-acts-of-pizza/test.json'))
test = json_normalize(test)

print("Train :", train.shape[0], "rows", train.shape[1], "columns")
print("Test :", test.shape[0], "rows", test.shape[1], "columns")
in_train_not_in_test = set(train.columns.values)-set(test.columns.values)
in_test_but_not_in_train = set(test.columns.values)-set(train.columns.values)
shared = set([c for c in train.columns.values if c in test.columns.values])

print("In train but not in test")
print(in_train_not_in_test, "\n")
print("In test but not in train")
print(in_test_but_not_in_train, "\n")
print("Shared")
print(shared)
shared.add('requester_received_pizza')
shared = list(shared)
train = train[shared]
dev_data, valid_data, dev_labels, valid_labels = \
    train_test_split(train, train['requester_received_pizza'], test_size=0.2)
dev_data.describe()
success = dev_data[dev_data.requester_received_pizza == True]
failure = dev_data[dev_data.requester_received_pizza == False]

print(round(len(success)/len(dev_data)*100, 2), "% successfull requests")
dev_data['datetime'] = pd.to_datetime(dev_data.unix_timestamp_of_request_utc, unit='s')
dev_data['day'] = dev_data.datetime.dt.dayofweek
dev_data['hour'] = dev_data.datetime.dt.hour
dev_data['month'] = dev_data.datetime.dt.month

plt.figure()
plt.title("Distribution of requests accross the year")
plt.xlabel("Month of the year")
plt.ylabel("Percentage")
plt.xticks(list(range(12)), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.plot(list(range(12)), [len(dev_data[dev_data.month == i])/len(dev_data)*100 for i in range(12)], 'bo', label='All requests')
plt.plot(list(range(12)), [len(dev_data[dev_data.requester_received_pizza == True][dev_data.month == i])/len(dev_data[dev_data.month == i])*100 if len(dev_data[dev_data.month == i]) > 0 else 0 for i in range(12)], 'rx', label='Successful requests')
plt.legend()

plt.figure()
plt.title("Distribution of requests accross the week")
plt.xlabel("Day of the week")
plt.ylabel("Percentage")
plt.xticks(list(range(7)), ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.plot(list(range(7)), [len(dev_data[dev_data.day == i])/len(dev_data)*100 for i in range(7)], 'bo', label='All requests')
plt.plot(list(range(7)), [len(dev_data[dev_data.requester_received_pizza == True][dev_data.day == i])/len(dev_data[dev_data.day == i])*100 for i in range(7)], 'rx', label='Successful requests')
plt.legend()

plt.figure()
plt.title("Distribution of requests accross the clock")
plt.xlabel("Time of the day")
plt.ylabel("Percentage")
plt.plot(list(range(24)), [len(dev_data[dev_data.hour == i])/len(dev_data)*100 for i in range(24)], 'bo', label='All requests')
plt.plot(list(range(24)), [len(dev_data[dev_data.requester_received_pizza == True][dev_data.hour == i])/len(dev_data[dev_data.hour == i])*100 for i in range(24)], 'rx', label='Successful requests')
plt.legend()
def compare_criterion(data, column):
    data.boxplot(column=[column], by='requester_received_pizza')
# Age
compare_criterion(dev_data, 'requester_account_age_in_days_at_request')
# Number of days since first post
compare_criterion(dev_data, 'requester_days_since_first_post_on_raop_at_request')
# Subreddits
#compare_criterion(dev_data, 'requester_subreddits_at_request')
compare_criterion(dev_data, 'requester_number_of_subreddits_at_request')
# Comments
compare_criterion(dev_data, 'requester_number_of_comments_at_request')
compare_criterion(dev_data, 'requester_number_of_comments_in_raop_at_request')
# Posts
compare_criterion(dev_data, 'requester_number_of_posts_at_request')
compare_criterion(dev_data, 'requester_number_of_posts_on_raop_at_request')
# Upvotes
compare_criterion(dev_data, 'requester_upvotes_plus_downvotes_at_request')
compare_criterion(dev_data, 'requester_upvotes_minus_downvotes_at_request')
def make_non_textual_features(data):
    mat = pd.DataFrame()
    
    # Time
    mat['datetime'] = pd.to_datetime(data['unix_timestamp_of_request_utc'], unit='s')
    mat['month'] = mat['datetime'].dt.month
    mat['day_of_week'] = mat['datetime'].dt.dayofweek
    mat['day_of_month'] = mat['datetime'].dt.day
    mat['hour'] = mat['datetime'].dt.hour
    del mat['datetime']
    
    # Age
    mat['age'] = data['requester_account_age_in_days_at_request']
    mat['community_age'] = (pd.to_datetime(data['unix_timestamp_of_request_utc'], unit = 's') - pd.to_datetime('2010-12-8', format='%Y-%m-%d')).astype('timedelta64[D]')
    
    # Popularity and activity
    mat['first_post']= data['requester_days_since_first_post_on_raop_at_request']
    mat['subreddits'] = data['requester_number_of_subreddits_at_request']
    mat['posts'] = data['requester_number_of_posts_at_request']
    mat['posts_pizza'] = data['requester_number_of_posts_on_raop_at_request']
    mat['comments'] = data['requester_number_of_comments_at_request']
    mat['comments_pizza'] = data['requester_number_of_comments_in_raop_at_request']
    mat['giver'] = data['giver_username_if_known'].apply(lambda x: 0 if x=='N/A' else 1)
    
    # Votes
    mat['upvotes_plus_downvotes'] = data['requester_upvotes_plus_downvotes_at_request']
    mat['upvotes_minus_downvotes'] = data['requester_upvotes_minus_downvotes_at_request']
    upvotes = mat.apply(lambda row : (row['upvotes_plus_downvotes'] + row['upvotes_minus_downvotes'])/2, axis=1)
    downvotes = mat.apply(lambda row : (row['upvotes_plus_downvotes'] - row['upvotes_minus_downvotes'])/2, axis=1)
    mat['upvotes'] = upvotes
    mat['downvotes'] = downvotes
    mat['votes_ratio'] = upvotes / (upvotes + downvotes + 1)

    return mat.as_matrix()
dev_non_textual = make_non_textual_features(dev_data)
valid_non_textual = make_non_textual_features(valid_data)
en_stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
blacklist = set.union(en_stopwords, punctuation)
w2v = gensim.models.KeyedVectors.load_word2vec_format('../input/google-word-to-vec/GoogleNews-vectors-negative300.bin', binary=True)  
def tokenize(s):
    tokens = ""
    sentences = sent_tokenize(s.lower())
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            if len(word) and word not in blacklist and not(word.isdigit()):
                tokens += word + " "
    return tokens[:-1]
def representation(s):
    vector = np.zeros(w2v.wv['the'].shape)
    count = 0
    for word in s.split():
        if word in w2v:
            count += 1
            vector = vector +  w2v[word]
    if count:
        vector /= count
    return vector
dev_request_tokens = dev_data['request_text_edit_aware'].apply(tokenize)
dev_request_len = dev_request_tokens.apply(lambda x : x.split()).apply(len)
dev_title_tokens = dev_data['request_title'].apply(tokenize)
dev_title_len = dev_title_tokens.apply(lambda x : x.split()).apply(len)
dev_all_tokens = dev_title_tokens.map(str) + ' ' + dev_request_tokens

valid_request_tokens = valid_data['request_text_edit_aware'].apply(tokenize)
valid_request_len = valid_request_tokens.apply(lambda x : x.split()).apply(len)
valid_title_tokens = valid_data['request_title'].apply(tokenize)
valid_all_tokens = valid_title_tokens.map(str) + ' ' + valid_request_tokens
vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,2), norm='l2', sublinear_tf=True)
dev_bow = vectorizer.fit_transform(dev_all_tokens)
valid_bow = vectorizer.transform(valid_all_tokens)

lr = LogisticRegression(C=1,penalty='l1').fit(dev_bow, dev_labels)
model = SelectFromModel(lr, prefit=True)
dev_pruned_bow = model.transform(dev_bow)
valid_pruned_bow = model.transform(valid_bow)
cv = CountVectorizer(min_df=5,ngram_range=(1,1))
dev_cv = cv.fit_transform(dev_all_tokens)
valid_cv = cv.transform(valid_all_tokens)

lda = LDA(n_components = 5, learning_method="batch", max_iter=30, learning_decay=.7)
dev_topics = lda.fit_transform(dev_cv)
valid_topics = lda.transform(valid_cv)
dev_representation = dev_all_tokens.apply(representation)
valid_representation = valid_all_tokens.apply(representation)
dev_representation = np.array(dev_representation.tolist())
valid_representation = np.array(valid_representation.tolist())
del dev_request_tokens, dev_title_tokens, dev_all_tokens
del valid_request_tokens, valid_title_tokens, valid_all_tokens
dev_inputs = np.zeros((dev_non_textual.shape[0], dev_non_textual.shape[1] + 2))
dev_inputs[:,:-2] = dev_non_textual
dev_inputs[:,-2] = dev_request_len
dev_inputs[:,-1] = dev_title_len

valid_inputs = np.zeros((valid_non_textual.shape[0], valid_non_textual.shape[1] + 2))
valid_inputs[:,:-2] = valid_non_textual
valid_inputs[:,-2] = valid_request_len
valid_inputs[:,-1] = valid_title_len
dev_input = hstack([dev_inputs, dev_pruned_bow, dev_topics, dev_representation])
valid_input = hstack([valid_inputs, valid_pruned_bow, valid_topics, valid_representation])
lr = LogisticRegression()
parameters = {'C':np.linspace(0.005, 0.1, 100)}
gs = GridSearchCV(lr, parameters, cv=5)
gs.fit(dev_input, dev_data['requester_received_pizza'])
pred_valid_prob = gs.predict_proba(valid_input)[:,1]
pred_valid_labels = gs.predict(valid_input)

print(gs.best_params_)
roc_auc_score(valid_data['requester_received_pizza'], pred_valid_prob, average='micro')
valid_labelss = valid_data['requester_received_pizza'].as_matrix()

success_count = 0
failure_count = 0

success_correct = 0
failure_correct = 0
for i in range(len(valid_labelss)):
    if valid_labelss[i]:
        success_count += 1
        if pred_valid_labels[i]:
            success_correct += 1
    else:
        failure_count += 1
        if not(pred_valid_labels[i]):
            failure_correct += 1
print(round(success_correct / success_count*100, 2), "% accurate prediction on success")
print(round(failure_correct / failure_count*100, 2), "% accurate prediction on failure")
print(round((success_correct + failure_correct) / (success_count + failure_count)*100, 2), "% accurate prediction in total")
