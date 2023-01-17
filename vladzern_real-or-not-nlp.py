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
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import re #работа с регулярными выражениями
def cleanhtml(raw_html): #удаление html
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def removeurl(raw_text): #удаление url
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*#', '', raw_text, flags=re.MULTILINE)
    return clean_text
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
data = pd.read_csv('../input/nlp-getting-started/train.csv',index_col='id')
data_test = pd.read_csv('../input/nlp-getting-started/test.csv',index_col='id')
data.head()
data = standardize_text(data, "text")
# Приведите тексты к нижнему регистру (text.lower())
data['text'] = data['text'].map(lambda x: x.lower())
data_test['text'] = data_test['text'].map(lambda x: x.lower())
# Замените все, кроме букв и цифр, на пробелы
data['text'] = data['text'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['text'] = data_test['text'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test.head(2)
from sklearn.feature_extraction.text import TfidfVectorizer
TF_IDF = TfidfVectorizer()
X_transform = TF_IDF.fit_transform(data['text']) # X - тексты
X_test_transform = TF_IDF.transform(data_test['text']) # X - тексты
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_transform = vectorizer.fit_transform(data['text'])
X_test_transform = vectorizer.transform(data_test['text']) # X - тексты
print (X_transform.shape)
print (X_test_transform.shape)
y = data['target']
y.shape
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=30.0, class_weight='balanced', multi_class='multinomial', n_jobs=-1, random_state=40) #penalty - вид регуляризации, class_weight - балансировка классов
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
rang = KFold(n_splits=10)
score = cross_val_score(estimator=model, X=X_transform, y=y, cv=rang,scoring='f1')
print (score.mean()) #средняя оценка
print (score.std())  #разброс по оценкам
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v:k for k,v in vectorizer.vocabulary_.items()}
    
    # loop for each class
    classes ={}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops':tops,
            'bottom':bottom
        }
    return classes
model.fit(X_transform, y)

importance = get_most_important_features(TF_IDF, model, 20)
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    
    bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    
    fig = plt.figure(figsize=(10, 10))  

    plt.subplot(121)
    plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplot(122)
    plt.barh(y_pos,top_scores, align='center', alpha=0.5)
    plt.title('Disaster', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    
    plt.subplots_adjust(wspace=0.8)
    plt.show()

top_scores = [a[0] for a in importance[0]['tops']]
top_words = [a[1] for a in importance[0]['tops']]
bottom_scores = [a[0] for a in importance[0]['bottom']]
bottom_words = [a[1] for a in importance[0]['bottom']]

plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
# Fill empty and NaNs values with NaN
data = data.fillna(np.nan)
# Check for Null values
data.isnull().sum()
# Fill empty and NaNs values with NaN
data_test = data_test.fillna(np.nan)
# Check for Null values
data_test.isnull().sum()
data['location'] = data['location'].fillna(0) #Замена пропусков на 0
print (np.any(data['location'].isnull()))
location = list()
for ob in data['location']:
    if (ob == 'USA') | (ob == 'United States') | (ob == 'New York'):
        location.append('USA')
    elif (ob == 'London'):
        location.append('London')
    elif ob=='Canada':
        location.append('Canada')
    elif ob== 0:
        location.append('0')
    else:
        location.append('other')
print(np.array(location).shape)
print (data['location'].shape)
print (np.any(data['location'].isnull()))
data['location'] = pd.Series(location, index=data.index)
print (data['location'].value_counts())
print (np.any(data['location'].isnull()))
from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer(sparse = False)
X_location = enc.fit_transform(data[['location']].to_dict('records'))
X_location.shape
data['keyword'] = data['keyword'].fillna('n_word') #Замена пропусков на 0
print (data['keyword'].value_counts())
enc = DictVectorizer(sparse = False)
X_keyword = enc.fit_transform(data[['keyword']].to_dict('records'))
print (np.any(pd.DataFrame(X_keyword).isnull()))
print (np.any(pd.DataFrame(X_location).isnull()))
from scipy.sparse import hstack
X_train = hstack([X_keyword, X_transform])
X_train.shape
from sklearn.svm import SVC
model = SVC(kernel='rbf', C = 10000)
score = cross_val_score(estimator=model, X=X_transform, y=y, cv=rang,scoring='f1')
print (score.mean()) #средняя оценка
print (score.std())  #разброс по оценкам
import xgboost as xgb
model = xgb.XGBClassifier(learning_rate =0.7, n_estimators=1000,  objective= 'binary:logistic', scale_pos_weight=1)
score = cross_val_score(estimator=model, X=X_transform, y=y, cv=rang,scoring='f1')
print (score.mean()) #средняя оценка
print (score.std())  #разброс по оценкам