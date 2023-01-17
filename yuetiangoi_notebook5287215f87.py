# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv')

df_train.head()
df_test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')

df_test.head()
df_train.drop('review_id', axis=1, inplace=True) #axies=1 is column



print('Train shape:', df_train.shape)

print('Test shape:', df_test.shape)
df_train.head()
df_train.rating.value_counts()
import re

def review_cleaning(text):

    

    # delete lowercase and newline

    text = text.lower()

    text = re.sub(r'\n', '', text)

    text = re.sub('([.,!?()])', r' \1 ', text)

    text = re.sub('\s{2,}', ' ', text)

    

    # change emoticon to text

    text = re.sub(r':\(', 'dislike', text)

    text = re.sub(r': \(\(', 'dislike', text)

    text = re.sub(r':, \(', 'dislike', text)

    text = re.sub(r':\)', 'smile', text)

    text = re.sub(r';\)', 'smile', text)

    text = re.sub(r':\)\)\)', 'smile', text)

    text = re.sub(r':\)\)\)\)\)\)', 'smile', text)

    text = re.sub(r'=\)\)\)\)', 'smile', text)

    

    # We decide to include punctuation in the model so we comment this line out!

#     text = re.sub('[^a-z0-9! ]', ' ', text)

    

    tokenizer = text.split()

    

    return ' '.join([text for text in tokenizer])
df_train['review'] = df_train['review'].apply(review_cleaning)

df_test['review'] = df_test['review'].apply(review_cleaning)
repeated_rows_train = []

repeated_rows_test = []



for idx, review in enumerate(df_train['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_train.append(idx)

        

for idx, review in enumerate(df_test['review']):

    if re.match(r'\w*(\w)\1+', review):

        repeated_rows_test.append(idx)
def delete_repeated_char(text):

    

    text = re.sub(r'(\w)\1{2,}', r'\1', text)

    

    return text
df_train.loc[repeated_rows_train, 'review'] = df_train.loc[repeated_rows_train, 'review'].apply(delete_repeated_char)

df_test.loc[repeated_rows_test, 'review'] = df_test.loc[repeated_rows_test, 'review'].apply(delete_repeated_char)
#print('Before: ', train_df_original.loc[92129, 'review'])

print('After: ', df_train.loc[92129, 'review'])



#print('\nBefore: ', train_df_original.loc[56938, 'review'])

print('After: ', df_train.loc[56938, 'review'])



#print('\nBefore: ', train_df_original.loc[72677, 'review'])

print('After: ', df_train.loc[72677, 'review'])



#print('\nBefore: ', train_df_original.loc[36558, 'review'])

print('After: ', df_train.loc[36558, 'review'])
import emoji

def emoji_cleaning(text):

    

    # Change emoji to text

    text = emoji.demojize(text).replace(":", " ")

    

    # Delete repeated emoji

    tokenizer = text.split()

    repeated_list = []

    

    for word in tokenizer:

        if word not in repeated_list:

            repeated_list.append(word)

    

    text = ' '.join(text for text in repeated_list)

    text = text.replace("_", " ").replace("-", " ")

    return text
have_emoji_train_idx = []

have_emoji_test_idx = []



for idx, review in enumerate(df_train['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_train_idx.append(idx)

        

for idx, review in enumerate(df_test['review']):

    if any(char in emoji.UNICODE_EMOJI for char in review):

        have_emoji_test_idx.append(idx)
train_emoji_percentage = round(len(have_emoji_train_idx) / df_train.shape[0] * 100, 2)

print(f'Train data has {len(have_emoji_train_idx)} rows that used emoji, that means {train_emoji_percentage} percent of the total')



test_emoji_percentage = round(len(have_emoji_test_idx) / df_test.shape[0] * 100, 2)

print(f'Test data has {len(have_emoji_test_idx)} rows that used emoji, that means {test_emoji_percentage} percent of the total')
train_df_original = df_train.copy()

test_df_original = df_test.copy()



# emoji_cleaning

df_train.loc[have_emoji_train_idx, 'review'] = df_train.loc[have_emoji_train_idx, 'review'].apply(emoji_cleaning)

df_test.loc[have_emoji_test_idx, 'review'] = df_test.loc[have_emoji_test_idx, 'review'].apply(emoji_cleaning)
#before clear

train_df_original.loc[have_emoji_train_idx, 'review'].tail()
# after cleaning

df_train.loc[have_emoji_train_idx, 'review'].tail()
def recover_shortened_words(text):

    

    # put \b (boundary) for avoid the characters in the word to be replaced

    # I only make a few examples here, you can add if you're interested :)

    

    text = re.sub(r'\bapaa\b', 'apa', text)

    

    text = re.sub(r'\bbsk\b', 'besok', text)

    text = re.sub(r'\bbrngnya\b', 'barangnya', text)

    text = re.sub(r'\bbrp\b', 'berapa', text)

    text = re.sub(r'\bbgt\b', 'banget', text)

    text = re.sub(r'\bbngt\b', 'banget', text)

    text = re.sub(r'\bgini\b', 'begini', text)

    text = re.sub(r'\bbrg\b', 'barang', text)

    

    text = re.sub(r'\bdtg\b', 'datang', text)

    text = re.sub(r'\bd\b', 'di', text)

    text = re.sub(r'\bsdh\b', 'sudah', text)

    text = re.sub(r'\bdri\b', 'dari', text)

    text = re.sub(r'\bdsni\b', 'disini', text)

    

    text = re.sub(r'\bgk\b', 'gak', text)

    

    text = re.sub(r'\bhrs\b', 'harus', text)

    

    text = re.sub(r'\bjd\b', 'jadi', text)

    text = re.sub(r'\bjg\b', 'juga', text)

    text = re.sub(r'\bjgn\b', 'jangan', text)

    

    text = re.sub(r'\blg\b', 'lagi', text)

    text = re.sub(r'\blgi\b', 'lagi', text)

    text = re.sub(r'\blbh\b', 'lebih', text)

    text = re.sub(r'\blbih\b', 'lebih', text)

    

    text = re.sub(r'\bmksh\b', 'makasih', text)

    text = re.sub(r'\bmna\b', 'mana', text)

    

    text = re.sub(r'\borg\b', 'orang', text)

    

    text = re.sub(r'\bpjg\b', 'panjang', text)

    

    text = re.sub(r'\bka\b', 'kakak', text)

    text = re.sub(r'\bkk\b', 'kakak', text)

    text = re.sub(r'\bklo\b', 'kalau', text)

    text = re.sub(r'\bkmrn\b', 'kemarin', text)

    text = re.sub(r'\bkmrin\b', 'kemarin', text)

    text = re.sub(r'\bknp\b', 'kenapa', text)

    text = re.sub(r'\bkcil\b', 'kecil', text)

    

    text = re.sub(r'\bgmn\b', 'gimana', text)

    text = re.sub(r'\bgmna\b', 'gimana', text)

    

    text = re.sub(r'\btp\b', 'tapi', text)

    text = re.sub(r'\btq\b', 'thanks', text)

    text = re.sub(r'\btks\b', 'thanks', text)

    text = re.sub(r'\btlg\b', 'tolong', text)

    text = re.sub(r'\bgk\b', 'tidak', text)

    text = re.sub(r'\bgak\b', 'tidak', text)

    text = re.sub(r'\bgpp\b', 'tidak apa apa', text)

    text = re.sub(r'\bgapapa\b', 'tidak apa apa', text)

    text = re.sub(r'\bga\b', 'tidak', text)

    text = re.sub(r'\btgl\b', 'tanggal', text)

    text = re.sub(r'\btggl\b', 'tanggal', text)

    text = re.sub(r'\bgamau\b', 'tidak mau', text)

    

    text = re.sub(r'\bsy\b', 'saya', text)

    text = re.sub(r'\bsis\b', 'sister', text)

    text = re.sub(r'\bsdgkan\b', 'sedangkan', text)

    text = re.sub(r'\bmdh2n\b', 'semoga', text)

    text = re.sub(r'\bsmoga\b', 'semoga', text)

    text = re.sub(r'\bsmpai\b', 'sampai', text)

    text = re.sub(r'\bnympe\b', 'sampai', text)

    text = re.sub(r'\bdah\b', 'sudah', text)

    

    text = re.sub(r'\bberkali2\b', 'repeated', text)

    

    text = re.sub(r'\byg\b', 'yang', text)

    

    return text
%%time

df_train['review'] = df_train['review'].apply(recover_shortened_words)
rating_mapper_encode = {1: 0,

                        2: 1,

                        3: 2,

                        4: 3,

                        5: 4}



# convert back to original rating after prediction later(dont forget!!)

rating_mapper_decode = {0: 1,

                        1: 2,

                        2: 3,

                        3: 4,

                        4: 5}



df_train['rating'] = df_train['rating'].map(rating_mapper_encode)
from sklearn.model_selection import train_test_split

X1_train, X1_test, Y_train, Y_test = train_test_split(df_train['review'], df_train['rating'], test_size=0.25, random_state=5)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(min_df=10)

X_train = vectorizer.fit_transform(X1_train)

X_test = vectorizer.transform(X1_test)
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

MNB.fit(X_train, Y_train)
from sklearn import metrics

predicted = MNB.predict(X_test)

accuracy_score = metrics.accuracy_score(predicted, Y_test)
print(str('{:04.2f}'.format(accuracy_score*100))+'%')
from sklearn.model_selection import cross_val_score



k_fold_scores = cross_val_score(MNB, X_train, Y_train, scoring='accuracy', cv=5)

print(k_fold_scores)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train,Y_train)
from sklearn.metrics import mean_squared_error



y_pred = lin_reg.predict(X_train)

lin_mse = mean_squared_error(y_pred, Y_train)

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
from sklearn.model_selection import cross_val_score



k_fold_scores = cross_val_score(lin_reg, X_train, Y_train, scoring='neg_mean_squared_error', cv=5)

linreg_rmse_scores = np.sqrt(-k_fold_scores)

print(linreg_rmse_scores)
df_train.rating.mean()
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(max_iter = 500)

log_reg.fit(X_train, Y_train)
y_pred=log_reg.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
from sklearn.model_selection import cross_val_score



k_fold_scores = cross_val_score(log_reg, X_train, Y_train, scoring='accuracy', cv=5)

print(k_fold_scores)