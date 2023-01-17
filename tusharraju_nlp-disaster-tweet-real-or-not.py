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
import os

import re

import numpy as np

import pandas as pd

from nltk.stem import PorterStemmer

from matplotlib import pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from wordcloud import WordCloud, STOPWORDS

from nltk.stem import WordNetLemmatizer



stopwords = set(STOPWORDS)
stopwords.add('will')

stopwords.add('ve')

stopwords.add('now')

stopwords.add('gonna')

stopwords.add('wanna')

stopwords.add('lol')

stopwords.add('via')

            
test = pd.read_csv('/kaggle/input/nlp-disaster-tweet/test.csv')

train = pd.read_csv('/kaggle/input/nlp-disaster-tweet/train.csv')
# Data Overview

train.head()
test.head()
print('Null values in Train data set%')

train.isnull().sum()/len(train)*100
print('Null values in Test dataset %')

test.isnull().sum()/len(test)*100
# Combining all the text data 

tweets_data = pd.concat([train, test], axis=0, sort=False, ignore_index=True)

tweets_data.head()
# Data cleaning steps

def clean_data(df, text_col, new_col='cleaned_text', stemming=False, lemmatization=True):

    

    '''

    It will remove the noise from the text data(@user, characters not able to encode/decode properly)    

    ----Arguments----

    df : Data Frame

    col : column name (string)

    stemming : boolean

    lemmatization : boolean

    '''

    tweets_data = df.copy() # deep copying the data in order to avoid any change in the main data col  

    

    # Creating one more new column for new text transformation steps

    tweets_data[new_col] = tweets_data[text_col]

    

    # removing @<userid>, as it is very common in the twitter data

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub('@[A-Za-z0-9_]+', '', x)) 

    

    # Removing &amp 

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub('&amp',' ', str(x)))

    

    # Removing URLs from the data

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub('https?:\/\/[a-zA-z0-9\.\/]+','',str(x)))

   

    # Changing into lower case

    tweets_data[new_col] = tweets_data[new_col].str.lower()

    

    #removing some common patterns

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’", "\'", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\s\'", " ", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"won\'t", "will not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"can\'t", "can not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"don\'t", "do not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"dont", "do not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\’t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"n\'t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'re", " are", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub("\'s", " is", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\’d", " would", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ll", " will", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'t", " not", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'ve", " have", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'m", " am", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\n", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\r", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\'", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r"\"", "", str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))

    tweets_data[new_col] = tweets_data[new_col].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))

   

    # Trimming the sentences

    tweets_data[new_col] = tweets_data[new_col].str.strip() 

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.findall("[A-Za-z0-9]+", x))

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))

    

    # Remove stopwords

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : ['' if word in stopwords else word for word in x.split()])

    

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))

        

    # Removing extra spaces

    tweets_data[new_col] = tweets_data[new_col].apply(lambda x : re.sub("\s+", " ", x))

     

        

        # lemmatization (lemmatization groups simmilar words according tothere meaning and context)

    if lemmatization:

        

        lemma = WordNetLemmatizer()

        

        tweets_data[new_col] = tweets_data[new_col].apply(lambda sentence :[lemma.lemmatize(word,'v') for word in sentence.split(" ")])

        

        tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))

    

    

    # Stemming code (stemming is used to words to change into there origin word like cowardly, easily becomes coward and easy)

    if stemming:

        stemming = PorterStemmer()

        

        tweets_data[new_col] = tweets_data[new_col].apply(lambda sentence : [stemming.stem(x) for x in sentence.split(" ")])

        

        tweets_data[new_col] = tweets_data[new_col].apply(lambda x : " ".join(x))



    return tweets_data
tweets_data = clean_data(tweets_data, "text", 

                         'cleaned_text', 

                         lemmatization=True,

                         stemming=False)



pd.set_option('display.max_colwidth', -1)



print('----- Text Before And After Cleaning -----')



tweets_data[['text', 'cleaned_text']].head(10)
stemming = PorterStemmer()

print(f"Runs converted to {stemming.stem('runs')}")

print(f"Stemming converted to {stemming.stem('stemming')}")
wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10)

wordcloud.generate(" ".join(tweets_data['cleaned_text']))

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None, dpi=80) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  
vec = TfidfVectorizer(ngram_range=(1,5),

#                       max_features=10000,

                      min_df=3,

                      stop_words='english')



tfidf_matrix = vec.fit_transform(tweets_data['cleaned_text'])



tfidf_matrix.shape
tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray(),

                            columns = vec.get_feature_names(),

                            dtype='float32')



print("Shape of the dataframe ",tfidf_matrix.shape)

print('Data Frame Info')

tfidf_matrix.info()
# Prepare the data set for model training



X = tfidf_matrix.iloc[range(0, train.shape[0]), :]



test_dataset = tfidf_matrix.iloc[train.shape[0]:, :] 

                           

y = tweets_data.loc[0:train.shape[0]-1, 'target']



x_train, x_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    random_state=123, 

                                                    test_size = 0.3)
clf = LogisticRegression(max_iter=1500,

                        solver='lbfgs')



clf.fit(x_train, y_train)
print("F1 Score is ", f1_score(y_test, clf.predict(x_test)))

confusion_matrix(y_test, clf.predict(x_test))
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
accuracy_score(y_test,clf.predict(x_test))
clf.fit(X, y)



act_pred = clf.predict(test_dataset)

act_pred = act_pred.astype('int')



submission_file = pd.DataFrame({'id' : test['id'],

                               'target' : act_pred})



submission_file.to_csv('subLR.csv', index = False)
nv = GaussianNB()

nv.fit(x_train, y_train)
print("F1 Score is ", f1_score(y_test, nv.predict(x_test)))

print('Confusion Matrix')

confusion_matrix(y_test, nv.predict(x_test))
rf = RandomForestClassifier(n_estimators=1500,

                            max_depth=6,

                            oob_score=True)

rf.fit(x_train, y_train)
print("F1 Score is ", f1_score(y_test, rf.predict(x_test)))

print('--------Confusion Matrix---------')

confusion_matrix(y_test, rf.predict(x_test))
accuracy_score(y_test, rf.predict(x_test))
rf.fit(X, y)



act_pred = rf.predict(test_dataset)

act_pred = act_pred.astype('int')



submission_file = pd.DataFrame({'id' : test['id'],

                               'target' : act_pred})



submission_file.to_csv('subrf.csv', index = False)