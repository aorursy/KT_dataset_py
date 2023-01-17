import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
def get_data():

    train = pd.read_csv("../input/steam-game-reviews/train.csv")

    test = pd.read_csv("../input/steam-game-reviews/test.csv")

    game = pd.read_csv("../input/steam-game-reviews/game_overview.csv")

    sub = pd.read_csv("../input/steam-game-reviews/sample_submission.csv")

    

    print("Train Shape : \t{}\nTest Shape : \t{}\nOverview Shape :{}\n".format(train.shape, test.shape, game.shape))



    return train, test, game, sub
train, test, game, sub = get_data()
train.head(2)
train_unique_titles = set(train['title'].unique())

test_unique_titles = set(test['title'].unique())



common_titles = set.intersection(train_unique_titles, test_unique_titles)



print("Number of Common Titles between Train and Test : {}".format(len(common_titles)))
train.drop(['title'], axis=1, inplace=True)

test.drop(['title'], axis=1, inplace=True)
plt.figure(figsize=(15, 5))

sns.countplot(train['year'], hue=train['user_suggestion'])

plt.show()
train.drop(['year'], axis=1, inplace=True)

test.drop(['year'], axis=1, inplace=True)
# Sample Review : 



train['user_review'].iloc[0]
# user_review distribution

lens = train['user_review'].str.len()

print("Mean : \t{}\nSTD : \t{}\nMAX: \t{}".format(lens.mean(), lens.std(), lens.max()))



lens.hist()

plt.title("User Review Distribution")

plt.show()
train['user_suggestion'].value_counts()
train[train['user_suggestion'] == 0]['user_review'].iloc[1]
train[train['user_suggestion'] == 1]['user_review'].iloc[1]
# Sample non-english reviews

train['user_review'].iloc[331]
!pip install langdetect

from langdetect import detect

from tqdm import tqdm_notebook
# Sample usage : 

train['user_review'].iloc[1][: 20], detect(train['user_review'].iloc[1])
def get_language(X):

    try:

        return detect(X)

    except:

        return "Error"
%%time



train['user_review_language'] = train['user_review'].apply(lambda x: get_language(x))
print("Original Shape : {}".format(train.shape))



new_lang_df = train[train['user_review_language'] != 'en'].copy()

train = train[train['user_review_language'] == 'en'].copy()



print("New Shape : {}".format(train.shape))
import re

from re import finditer



def remove_EAR(X):

    """

    Removing 'Early Access Review'

    """

    X = X.replace("Early Access Review", "")

    

    return X



def split_number_and_text(X):

    x = re.split('(\d+)', X)

    x = " ".join(x)

    x = x.strip()

    

    return x



def handle_camelcase(X):

    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', X)

    return " ".join([m.group(0) for m in matches])



def handling_whitespaces(X):

    X = " ".join(X.split())

    X = X.strip()

    

    return X



waste_symbols = "人̳⣟⣦̪⠓▒͎¸⠟⣅>⡾ ⠻⣀⣛„ͭ⣮⡻⠦⡀͐‘̨⣆̤⣿<／丶⣞͇⣵͞⠹ͩ⢒̯⢸⣤̗̫ͯ͆̔͠⠛⢻⠏-́☐̺͛̋⠸⣥⠄̷＼͟·⌒͗⠁́｀⢹\\⢄͈̌ͨ⢤彡~¯/⠶⠲ˆ⡥̮̻͔☉⣻̣ゝ⡞̧͙̿̒̊̑ノ⠭ͤ_⠐⣇҉̚–⡄´̓█▄☑⣧̴͖̍｜⣷̭͘͝｡⠴̜̄ʖ¨̵̏͢⢂͋;͒:⢉つ̾＿̈⣴⣌ͫ⢛⡹⣈へ⢯,̅⣭̩̬̕⡈ム͡⣼ͦ)̛͜ヽ̝̥⣠⢟̶⠤̡͉⠘̹̈́⡴̠⢀）⠇⣾͊⢰̞ͮ̇`⠑⡿\u3000⠃⣸⠾͍̆ͅ￣⢚̓⠂⡵─⢬ー⠿(⠆⠉̦*͕ﾉ⣹⡟⣬⠙▓⡐7͏̟̲⢿⢦（̰♥̸̢⣙͓̂▀くﾌ⠀.⠰⡒°̖̎､⣒⣰̼⢅⣁⠒͑⢾⡂͌̀ͧ…̃▐ﾚ、丿⢌|̱⢴⡠⣩▌⣉͚ͪ'⢆⢠⡇⡛⣏⡶⣜⣄⡸⠈̘ͣ⣽̉̽̐ͥ⡏ͬ⣗⣶░⠋⠔̙͂^"



def remove_waste_symbols(X):

    for item in waste_symbols:

        X = X.replace(item, " ")

        

    return X
def clean_review(X):

    X = remove_EAR(X)

    X = remove_waste_symbols(X)

    X = handle_camelcase(X)

    X = split_number_and_text(X)

    X = handling_whitespaces(X)

    

    return X
%%time



train['user_review_clean'] = train['user_review'].apply(lambda x: clean_review(x))

test['user_review_clean'] = test['user_review'].apply(lambda x: clean_review(x))
train.reset_index(drop=True, inplace=True)
from sklearn.feature_extraction.text import CountVectorizer



sample_texts = ['Hello this is review number 1, Bye Bye', 'I am not a review']



vect = CountVectorizer()

vect.fit(sample_texts)



vect.vocabulary_
for item in sample_texts:

    

    print("Text : {}\nEncoded Format : {}".format(item, vect.transform([item]).toarray()))
X = pd.DataFrame(sample_texts, columns=['text'])

enc_texts = vect.transform(X['text'].values)

enc_texts = pd.DataFrame(enc_texts.toarray(), columns=vect.get_feature_names())



X = pd.concat([X, enc_texts], axis=1)

X.head(2)
%%time



total_reviews = pd.concat([train['user_review_clean'], test['user_review_clean']], axis=0)

total_reviews.reset_index(drop=True, inplace=True)



vect = CountVectorizer()



vect.fit(total_reviews.values)



train_count_vect = vect.transform(train['user_review_clean'].values)

test_count_vect = vect.transform(test['user_review_clean'].values)



print("Number of features / words in vocab : {}".format(len(vect.get_feature_names())))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score
X_train, X_valid, y_train, y_valid = train_test_split(train_count_vect, train['user_suggestion'], test_size=0.15, random_state=13)
model = LogisticRegression()



model.fit(X_train, y_train)



train_score = accuracy_score(y_train, model.predict(X_train))

valid_score = accuracy_score(y_valid, model.predict(X_valid))



print("Train Score : {}\nValid Score : {}".format(train_score, valid_score))
results = cross_val_score(model, train_count_vect, train['user_suggestion'].values, cv=3, scoring='accuracy')



print("Accuracy Mean : \t{}\n3-Fold Scores : \t{}".format(results.mean(), results))
from sklearn.feature_extraction.text import TfidfVectorizer
%%time



vect = TfidfVectorizer()



vect.fit(total_reviews.values)



train_tfidf_vect = vect.transform(train['user_review_clean'].values)

test_tfidf_vect = vect.transform(test['user_review_clean'].values)



print("Number of features / words in vocab : {}".format(len(vect.get_feature_names())))
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')



print(english_stopwords[: 5])



vect = TfidfVectorizer(stop_words=english_stopwords)
X_train, X_valid, y_train, y_valid = train_test_split(train_tfidf_vect, train['user_suggestion'], test_size=0.15, random_state=13)
model = LogisticRegression()



model.fit(X_train, y_train)



train_score = accuracy_score(y_train, model.predict(X_train))

valid_score = accuracy_score(y_valid, model.predict(X_valid))



print("Train Score : {}\nValid Score : {}".format(train_score, valid_score))
results = cross_val_score(model, train_tfidf_vect, train['user_suggestion'].values, cv=3, scoring='accuracy')



print("Accuracy Mean : \t{}\n3-Fold Scores : \t{}".format(results.mean(), results))