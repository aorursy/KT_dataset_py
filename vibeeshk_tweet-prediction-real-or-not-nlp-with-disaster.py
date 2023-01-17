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
filepath= '/kaggle/input/nlp-getting-started/train.csv'

traindata= pd.read_csv(filepath)

filepaths= '/kaggle/input/nlp-getting-started/test.csv'

testdata= pd.read_csv(filepaths)

traindata.head()
import pandas as pd

import nltk

import numpy as np

import seaborn as sns

import re

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

import spacy

from nltk.stem import PorterStemmer 

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from nltk.corpus import words

word_list = words.words()









traindata2=traindata

testdata2=testdata









added_df = pd.concat([traindata,testdata])

len(added_df['keyword'])





textlist=added_df['text'].tolist()

keywordlist=added_df['keyword'].tolist()

targetlist=added_df['target'].tolist()



len(textlist)







stop_words = set(stopwords.words('english')) 

sp = spacy.load('en_core_web_sm')

all_stopwords = sp.Defaults.stop_words

textlist2=[]

for i1 in textlist:

    i1 = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', i1, flags=re.MULTILINE)

    i1 = re.sub('[0-9]', '', i1)

    i1 = re.sub(r'^https?:\/\/.*[\r\n]*', '', i1, flags=re.MULTILINE)

    i1=re.sub(r'[^\w\s]','',i1)

    i1=re.sub(r'\W', ' ', i1) 

    i1=i1.lower()

    i2 = word_tokenize(i1)

    i2= [word for word in i2 if not word in all_stopwords]

  #  i2= [word for word in i2 if not word in word_list]

   # word_list

    i2 = [w for w in i2 if not w in stop_words]

    i3=' '.join(i2)

    textlist2.append(i3)







#Now we Lemmetize the text

lemmatizer = WordNetLemmatizer() 

textlist3=[]

for i4 in textlist2:

    i4=lemmatizer.lemmatize(i4)

    textlist3.append(i4)

len(textlist3)





added_df=added_df.drop('text',axis=1)

added_df['text']=textlist3

cv=CountVectorizer()

added_df=cv.fit_transform(added_df['text'])

added_df=pd.DataFrame(added_df.todense())



#addedcv=addedcv.drop('keyword',axis=1)

added_df['keywords']=keywordlist

added_df['target']=targetlist



added_df['keywords'] = added_df['keywords'].replace(np.nan, 'keywords', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

added_df['keywords'] = 'keywords-' + added_df['keywords'].astype(str)

added_df.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(added_df['keywords'])

# Drop column Product as it is now encoded

added_df = added_df.drop('keywords',axis = 1)

# Join the encoded testdata

added_df = added_df.join(one_hot)

added_df.head()









added_df = added_df.replace(np.nan, 0, regex=True)

added_df = added_df.reset_index()

training=added_df.head(7613)

testing=added_df.iloc[7613:]



y=training['target']

x=training.drop('target',axis=1)

testing=testing.drop('target',axis=1)





###We use only the keyword and text column because only they are mostly used to determine the



#Logistic Regression

#df = df.reset_index()

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x, y)

Prediction = LogisticRegressor.predict(testing)









Prediction
predictionlist=Prediction.tolist()

Passengerid=testdata2['id'].tolist() 

output=pd.DataFrame(list(zip(Passengerid, predictionlist)),

              columns=['id','target'])

output.head()

                    

output.to_csv('my_submission(twitterdisaster)1.csv', index=False)