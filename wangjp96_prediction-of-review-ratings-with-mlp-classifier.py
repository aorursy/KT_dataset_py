import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import sklearn
import re

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def get_data():
    df = pd.read_csv('../input/GrammarandProductReviews.csv')
    return df
rawdata = get_data()
rawdata.head(5)
useful_list = ['reviews.rating', 'reviews.text']
data = rawdata.dropna(subset=['reviews.text'])[useful_list]
data.shape
plt.hist(data['reviews.rating'], range=(1, 6), align='left', color='y', edgecolor='black')
for x, y in zip(range(1, 6), data['reviews.rating'].value_counts(sort=False)):
    plt.text(x, y, str(y), ha='center', va='bottom', fontsize=8)
plt.title('Rating distribution')
plt.xlabel('ratings')  
plt.ylabel('frequency')
plt.show()
def split_text(text):
    text = re.sub("[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    text = text.lower()
    return text.split()

def get_text_len(data):
    len_dict = {'text_length' : {}}
    for i in range(len(data)):
        text = str(data.iloc[i]['reviews.text'])
        len_dict['text_length'][i] = len(split_text(text))
    len_df = pd.DataFrame.from_dict(len_dict)
    return pd.concat([data.reset_index(drop=True), len_df], axis=1)

text_len_df = get_text_len(data)
print("The average text length is " + str(text_len_df['text_length'].mean()))
print("The correlation coefficient between text length and rating is " + str(text_len_df.corr().iloc[0][1]))
print("The longest review text consists of " + str(max(text_len_df['text_length'])) + " words.")
len(text_len_df[text_len_df['text_length'] > 200])
plt.figure(figsize=(12, 8))
plt.hist(text_len_df['text_length'], bins=40, range=(0, 200), align='left', color='y', edgecolor='black')
plt.title('Distribution of text length (<= 200)')
plt.xlabel('text length')  
plt.ylabel('frequency')
plt.show()
def split_data(data, frac=0.7):
    train = data.sample(frac=frac)
    test = data[~data.index.isin(train.index)]
    return train, test

train_data, test_data = split_data(data)
def get_sentence_list(data, column):
    sentence_list = []
    stop = [re.sub("[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", word) for word in stopwords.words('english')]
    for i in range(len(data)):
        text = data.iloc[i][column]
        word_list = [word for word in split_text(text) if word not in stop]
        sentence_list.append(word_list)
    return sentence_list

from gensim.models import Word2Vec

def get_word_vec(data, column, dims=100):
    s_list = get_sentence_list(data, column)
    model = Word2Vec(s_list, size=dims, min_count=5)
    wv = model.wv
    del model
    return wv
len(text_len_df[text_len_df['text_length'] > 50]) / len(data)
def vectorize_data(data, column='reviews.text', with_wv=False, return_wv=True, init_wv=None, dims=100, max_sent_len=50):
    if with_wv:
        wv = init_wv
    else:
        wv = get_word_vec(data, column, dims)
    df  = {'word_vec' : {}}
    for i in range(len(data)):
        text = data.iloc[i][column]
        word_list = split_text(text)
        sentence_mat = []
        j = 0
        while j < max_sent_len:
            if j < len(word_list):
                if word_list[j] in wv:
                    sentence_mat.append(list(wv[word_list[j]]))
                else:
                    sentence_mat.append([0] * dims)
            else:
                sentence_mat.append([0] * dims)
            j += 1
        df['word_vec'][i] = np.array(sentence_mat).flatten()
    result = pd.DataFrame.from_dict(df)
    if return_wv:
        return pd.concat([data.reset_index(drop=True), result], axis=1), wv
    else:
        return pd.concat([data.reset_index(drop=True), result], axis=1)

def get_matrix(data):
    result = []
    for i in range(len(data)):
        result.append(list(data.iloc[i]['word_vec']))
    return np.array(result)

def get_score(data, col_1='reviews.rating', col_2='predicted_rating'):
    count = 0
    for i in range(len(data)):
        if data.iloc[i][col_1] == data.iloc[i][col_2]:
            count += 1
    return count / len(data)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
def use_model(train_data, test_data, model, x_col='reviews.text', y_col='reviews.rating'):
    train, wv = vectorize_data(train_data, x_col)
    x = get_matrix(train)
    y = np.array(train[y_col])
    model.fit(x, y)
    test= vectorize_data(test_data, with_wv=True, return_wv=False, init_wv=wv)
    test_x = get_matrix(test)
    y_predict = model.predict(test_x)
    y_pre_df = pd.DataFrame(y_predict)
    y_pre_df.columns = ['predicted_rating']
    prediction = pd.concat([test_data.reset_index(drop=True), y_pre_df], axis=1)
    return prediction, get_score(prediction, col_1=y_col)
prediction, score = use_model(train_data, test_data, model)
score
def get_binary(data, column='reviews.rating'):
    df = {'binary_rating' : {}}
    for i in range(len(data)):
        if data.iloc[i][column] >= 4:
            df['binary_rating'][i] = 1
        elif data.iloc[i][column] <= 3:
            df['binary_rating'][i] = 0
    result = pd.DataFrame.from_dict(df)
    return pd.concat([data.reset_index(drop=True), result], axis=1)
data_bin = get_binary(data)
train_data_bin, test_data_bin = split_data(data_bin)
pred_bin, score_bin = use_model(train_data_bin, test_data_bin, model, y_col='binary_rating')
score_bin
def get_key_words(data, column='reviews.text'):
    s_list = get_sentence_list(data, column)
    word_dict = {}
    for sentence in s_list:
        for word in sentence:
            if word not in word_dict.keys():
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    result = {'words' : {}, 'frequency' : {}}
    count = 0
    for word in word_dict.keys():
        result['words'][count] = word
        result['frequency'][count] = word_dict[word]
        count += 1
    return pd.DataFrame.from_dict(result)
def get_pos_and_neg_words(data):
    data_pos = data[data['reviews.rating'] >= 4]
    data_neg = data[data['reviews.rating'] <= 3]
    pos_words = list(get_key_words(data_pos).nlargest(100, 'frequency')['words'])
    neg_words = list(get_key_words(data_neg).nlargest(100, 'frequency')['words'])
    commendatory = [word for word in pos_words if word not in neg_words]
    derogatory = [word for word in neg_words if word not in pos_words]
    return commendatory, derogatory
def encode_text_by_word_frequency(data, column='reviews.text'):
    commendatory, derogatory = get_pos_and_neg_words(data)
    result = {'pos_word_frequency' : {}, 'neg_word_frequency' : {}}
    for i in range(len(data)):
        text = data.iloc[i][column]
        word_list = split_text(text)
        pos_count = 0
        neg_count = 0
        for word in word_list:
            if word in commendatory:
                pos_count += 1
            if word in derogatory:
                neg_count += 1
        rating = data.iloc[i]['reviews.rating']
        if rating >= 4:
            result['pos_word_frequency'][i] = pos_count
            result['neg_word_frequency'][i] = -neg_count
        elif rating <= 3:
            result['pos_word_frequency'][i] = -pos_count
            result['neg_word_frequency'][i] = neg_count
    result_df = pd.DataFrame.from_dict(result)
    return pd.concat([data.reset_index(drop=True), result_df], axis=1)
data_alt = encode_text_by_word_frequency(data_bin, column='reviews.text')
train_alt, test_alt = split_data(data_alt)
train_x = np.array(train_alt[['pos_word_frequency', 'neg_word_frequency']])
train_y = np.array(train_alt['binary_rating'])

model.fit(train_x, train_y)

test_x = np.array(test_alt[['pos_word_frequency', 'neg_word_frequency']])
y_predict = model.predict(test_x)
y_pred_df = pd.DataFrame(y_predict)
y_pred_df.columns = ['predicted_rating']
pred_alt = pd.concat([test_alt.reset_index(drop=True), y_pred_df], axis=1)
score_alt = get_score(pred_alt, col_1='binary_rating')
score_alt
commendatory, derogatory = get_pos_and_neg_words(data)
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=30, background_color='gray').generate(re.sub("'", "", str(commendatory)))
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.title("Commendatory Terms")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=30, background_color='gray').generate(re.sub("'", "", str(derogatory)))
plt.figure(figsize=(8, 8))
plt.imshow(wordcloud)
plt.title("Derogatory Terms")
plt.axis("off")
plt.show()