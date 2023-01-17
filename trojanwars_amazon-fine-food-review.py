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
%matplotlib inline



import sqlite3

import pandas as pd

import numpy as np

import nltk

import os

from IPython.core.display import display, HTML





if not os.path.isfile('../input/database.sqlite'):

    display(HTML("""

        <h3 style='color: red'>Dataset database missing!</h3><h3> Please download it

        <a href='https://www.kaggle.com/snap/amazon-fine-food-reviews'>from here on Kaggle</a>

        and extract it to the current directory.

          """))

    raise(Exception("missing dataset"))
con=sqlite3.connect('../input/database.sqlite')

pd.read_sql_query('select * from reviews limit 5', con)
all_messages=pd.read_sql_query("""

SELECT 

  *

FROM Reviews 

WHERE Score""", con)



    

messages = pd.read_sql_query("""

SELECT 

  Score, 

  Summary, 

  HelpfulnessNumerator as VotesHelpful, 

  HelpfulnessDenominator as VotesTotal

FROM Reviews 

WHERE Score != 3""", con)
messages.shape

messages.dtypes



messages.head(5)
all_messages.shape
messages.head(4)



messages.dtypes



messages.head(5)
messages['usefulness']=(messages['VotesHelpful']/messages['VotesTotal']).apply(lambda x: 'useful' if x>0.8 else 'not useful')





messages['Sentiment']=messages['Score'].apply(lambda x:'Positive' if x>3 else 'Negative')



messages[messages['Score']==5].head(10)
messages[messages['Score']==5].head(10)


from sklearn.cross_validation import train_test_split



X_train,X_test=train_test_split(messages,test_size=0.2)
print('%d in training set,%d in test set'%(len(X_train),len(X_test)))



print('%d in training set,%d in test set'%(len(X_train),len(X_test)))
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

import re

import nltk





def cleanup(sentences):

    alphabet_only=re.sub(r'^[A-Za-z]+'," ",sentences)

    word_lower=alphabet_only.lower().split()

    from nltk.corpus import stopwords

    useful_word=[w for w in word_lower if  not w in set(stopwords.words("english"))]

    from nltk.stem import WordNetLemmatizer  

    lemma=WordNetLemmatizer()

    finalize_word=[lemma.lemmatize(x) for x in useful_word]

    return(str(" ".join(finalize_word)))

    

messages['Summary_clean']=messages['Summary'].apply(cleanup)    



messages.head()   

#messages['Summary_Clean']=messages['Summary'].apply(lambda x: cleanup(x))

#train['Comments1'] = train.apply(lambda row: textpreprocessing1(row['Comments'],"l"),axis=1)



#X_train,X_test=train_test_split(messages,test_size=0.2)





X_train,X_test=train_test_split(messages,test_size=0.2)



from sklearn.feature_extraction.text  import CountVectorizer

count_vector=CountVectorizer()

x_train_count=count_vector.fit_transform(X_train['Summary_clean']).toarray()



x_train_count.get_feature_names()
from wordcloud import WordCloud

from wordcloud import STOPWORDS

count_vect = CountVectorizer(min_df = 1,ngram_range=(1,4))

x_train_count=count_vect.fit_transform(X_train['Summary_Clean'])

tfidf_transformer=TfidfTransformer()

X_train_tfidf=tfidf_transformer.fit_transform(x_train_count)

X_new_counts=count_vect.fit_transform(X_test['Summary_Clean'])

X_test_tfidf=tfidf_transformer.fit_transform(X_new_counts)
y_train=X_train['Sentiment']

y_test=X_test['Sentiment']



prediction=dict()
print(X_train.shape)

print(X_test.shape)
print((x_train_tfidf.toarrray()).shape)
prediction
from wordcloud import WordCloud

from  wordcloud import STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data,title=None):

    wordcloud=WordCloud(

               background_color='white',

               stopwords=stopwords,

               max_words=300,

               max_font_size=40

    ).generate(str(data))  

    fig = plt.figure(1, figsize=(8, 8))

    plt.axis('off')

    plt.imshow(wordcloud)

    plt.show()







show_wordcloud(messages['Summary_Clean'])
show_wordcloud(messages[messages.Score==1]['Summary_Clean'])
show_wordcloud(messages[messages.Score==5]['Summary_Clean'])
messages['Summary_Clean'].head(3)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB().fit(X_train_tfidf, y_train)

prediction['Bernoulli'] = model.predict(X_test_tfidf)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train_tfidf, y_train)

prediction['Multinomial'] = model.predict(X_test_tfidf)