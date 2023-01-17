import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



raw_data = pd.read_csv('../input/All-seasons.csv')

raw_data.head()
#little summary, not all lines are unique. Moreover alot of characters

raw_data.describe()
#count lines for character. Data have big "tale" of more or less unique characters

#It seems reasonable to take only those caracters, who spoke at least 100 times (sligtly more than once per episode)

raw_data.groupby(['Character']).size().sort_values(ascending=False)
#select lines for top speakres

top_speakers = raw_data.groupby(['Character']).size().loc[raw_data.groupby(['Character']).size() > 2000]

#print(top_speakers.index.values)

main_char_lines = raw_data.loc[raw_data['Character'].isin(top_speakers.index.values)]

main_char_lines.describe()

#main_char_lines
from sklearn.model_selection import train_test_split



main_char_lines['Line'] = [line.replace('\n','') for line in main_char_lines['Line']]

train, test = train_test_split(main_char_lines, test_size=0.3, random_state=14)
#preprocess data, vectorizing lines

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

import nltk

from nltk.stem.lancaster import LancasterStemmer



st = LancasterStemmer()

def token(text):

    txt = nltk.word_tokenize(text.lower())

    return [st.stem(word) for word in txt]





stop = set(stopwords.words("english"))

cv = CountVectorizer(#lowercase=True, 

                     tokenizer=token, #stop_words=stop,# token_pattern=u'(?u)\b\w\w+\b',

                     analyzer=u'word', min_df=4)

#print(train['Line'].tolist())



vec_train = cv.fit_transform(train['Line'].tolist())

vec_test = cv.transform(test['Line'].tolist())



#print(vec_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, f1_score, accuracy_score



rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

rf.fit(X = vec_train, y = train['Character'])



accuracy_score(rf.predict(vec_test), test['Character'])
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()#multi_class='multinomial')

lr.fit(X = vec_train, y = train['Character'])



accuracy_score(lr.predict(vec_test), test['Character'])