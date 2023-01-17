# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, re, string

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

sample = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")



train_pos=train_data.loc[train_data['sentiment'] == 'positive']

train_neg=train_data.loc[train_data['sentiment'] == 'negative']

train_nut=train_data.loc[train_data['sentiment'] == 'neutral']

train_pos.shape[0]+train_neg.shape[0] + train_nut.shape[0],train_data.shape[0]
def clean(text):

    for i in range(2):

        #Make text lowercase, remove text in square brackets,remove links,remove punctuation

        #and remove words containing numbers.

        text = str(text).lower()

        text = re.sub('\[.*?\]', '', text)

        text = re.sub('https?://\S+|www\.\S+', '', text)

        text = re.sub('<.*?>+', '', text)

        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

        text = re.sub('\n', '', text)

        text = re.sub('\w*\d\w*', '', text)

        return text
grouped_cleaned=[clean(train_pos['text'].str.cat(sep=' ')),

          clean(train_neg['text'].str.cat(sep=' ')),

          clean(train_nut['text'].str.cat(sep=' '))]
# Initialize CountVectorize

cv=CountVectorizer(ngram_range=(1, 3),stop_words='english')

cv
# CountVectorize fit (apply CountVectorize to the documents) assign a number to each word in alphabetical order

fitted_words=cv.fit(grouped_cleaned)

fitted_words.vocabulary_
# count the number of instances of each word in each document

counted_fitted_words=cv.transform(grouped_cleaned)

pd.DataFrame(counted_fitted_words.todense(),columns=cv.get_feature_names())
# Initialise the tfidfTransformer

tfidf=TfidfTransformer()

tfidf
#find idf values for each word

idf_words=tfidf.fit(counted_fitted_words)

pd.DataFrame(idf_words.idf_,index=cv.get_feature_names())
# Find the TF IDF for each word in all the documents

TFidf_words=tfidf.transform(counted_fitted_words)

df_TFidf_words=pd.DataFrame(TFidf_words.todense(),columns=cv.get_feature_names())

df_TFidf_words
type(clean(test_data['text']))

cleaned_test_data=test_data.copy()

for row in range(len(test_data)):

    cleaned_test_data['text'][row]=clean(test_data['text'][row])

cleaned_test_data
submission=cleaned_test_data.copy()

submission.insert(2,'selected_text',['']*len(test_data))

submission.head()



for row in range(len(submission)):

    cv_row=CountVectorizer(ngram_range=(1, 3))

    try:

        cv_row_count=cv_row.fit([submission['text'][row]]) #or use .iloc[row,col] col 0 = ID, 1 = text, 2 = sentiment

        ngrams_row = list(cv_row_count.vocabulary_)

        tfidf_word_score=df_TFidf_words.reindex(columns=ngrams_row,fill_value=0)

        if submission['sentiment'][row]=='neutral':    

            submission['selected_text'][row]=submission['text'][row]  #neutral is 3rd (2) row in grouped



        elif submission['sentiment'][row]=='positive':

            submission['selected_text'][row]=tfidf_word_score.loc[0,:].idxmax(axis=0) #positive is 1st row (0) in grouped

        elif submission['sentiment'][row]=='negative':

            submission['selected_text'][row]=tfidf_word_score.loc[1,:].idxmax(axis=0) #negative is 2nd row (1) in grouped

    except:

        print('row ',row,' broken')

submission

submission.to_csv('submission.csv', columns=['textID','selected_text'], index = False)