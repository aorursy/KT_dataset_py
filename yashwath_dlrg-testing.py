!pip install scikit-learn==0.23
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score



import warnings

warnings.filterwarnings('ignore')
!pip install multi-imbalance
df= pd.read_excel('../input/german/Germen.xlsx')

df.head(3)
test = pd.read_excel("../input/translated/Germen_test.xlsx")
test.head()
print(df.shape)

print(test.shape)

1135+1260
import nltk

import re

nltk.download('wordnet')

from nltk.tokenize import WordPunctTokenizer 

from nltk.stem import WordNetLemmatizer 

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

tok = WordPunctTokenizer()



lemmatizer = WordNetLemmatizer() 



nltk.download('stopwords')

from nltk.corpus import stopwords

stopword = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer





def tweet_cleaner(text):

   

    wc=[]

    newString = str(text)                #encode to ascii

    newString=re.sub(r'@[A-Za-z0-9]+','',newString)                                #removing user mentions

    letters_only = re.sub("[^a-zA-Z]", " ", newString)                             #Fetching out only ascii characters

    letters_onl = re.sub('(www|http)\S+', '', letters_only)                        #removing links

    lower_case = letters_onl.lower()                                               #converting everything to lowercase

    words = tok.tokenize(lower_case)                                               #tokenize and join together to remove white spaces between words

    rs = [word for word in words if word not in stopword]                           #remove stopwords

    long_words=[]

    for i in rs:

      if len(i)>3:                                                 #removing short words

        long_words.append(lemmatizer.lemmatize(i))                 #converting words to lemma

    return (" ".join(long_words)).strip()      
cleaned_tweets = []

for t in df.textblob_translation:

  cleaned_tweets.append(tweet_cleaner(t))

df['cleaned_tweets']=cleaned_tweets #creating new dataframe

cleaned_tweets[:5]
cleaned_tweets1 = []

for t in test.Germen_test:

  cleaned_tweets1.append(tweet_cleaner(t))

test['cleaned_tweets']=cleaned_tweets1 #creating new dataframe

cleaned_tweets1[:5]
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



tvec = TfidfVectorizer(stop_words='english') #max_features=Max no. of words to be considered, min-df=Min frequency of word 

tfidf = tvec.fit_transform(df['cleaned_tweets'])

tfidf.shape
##testing set



vecc2=TfidfVectorizer(vocabulary=tvec.get_feature_names())

tfidf2 = vecc2.fit_transform(test['cleaned_tweets'])

print(tfidf2.shape)  
from multi_imbalance.utils.plot import plot_cardinality_and_2d_data

plot_cardinality_and_2d_data(tfidf.toarray(), np.array(df.task1), 'Plotting')
from multi_imbalance.resampling.soup import SOUP
mdo = SOUP(maj_int_min={

    'maj': ['NOT'],

    'min': ['HOF']

    })

X_train_res, y_train_res = mdo.fit_resample(tfidf.toarray(), np.array(df.task1))
print(len(X_train_res))

print(len(tfidf.toarray()))
from multi_imbalance.utils.plot import plot_visual_comparision_datasets

plot_visual_comparision_datasets(tfidf.toarray(), df.task1, X_train_res, y_train_res, 'Code-data', 'Resampled data')


from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score





lr = LogisticRegression(solver='lbfgs')

lr_fit =lr.fit(X_train_res, y_train_res)

prediction = lr_fit.predict(tfidf2)

from sklearn.metrics import classification_report

print(classification_report(prediction,test.task1))
print(prediction[:10])
t.label
from sklearn.model_selection import GridSearchCV

from sklearn import svm
parameters = {'kernel':('linear', 'rbf')}

svr = svm.SVC(verbose=2)

SVM = GridSearchCV(svr, parameters, verbose=2)



SVM.fit(X_train_res, y_train_res)



# predict the labels on validation dataset

predictions_SVM = SVM.predict(tfidf2.toarray())

print(classification_report(predictions_SVM,test.task1))