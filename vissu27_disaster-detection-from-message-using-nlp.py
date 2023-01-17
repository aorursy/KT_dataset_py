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
import matplotlib.pyplot as plt

%matplotlib inline



disater_data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

disater_data_test =  pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
disater_data_train.columns
disater_data_target = disater_data_train[['target']]
#Total Count of Each Category

pd.set_option('display.width', 4000)

pd.set_option('display.max_rows', 1000)

distOfDetails = disater_data_train.groupby(by='target', as_index=False).agg({'id': pd.Series.nunique}).sort_values(by='id', ascending=False)

distOfDetails.columns =['Class', 'COUNT']

print(distOfDetails)



#Distribution of All Categories

plt.pie(distOfDetails['COUNT'],autopct='%1.0f%%',shadow=True, startangle=360)

plt.show()
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

def textprocessing(data):

    #Text Preprocessing

    columns = ['id','target', 'text']

    df_ = pd.DataFrame(columns=columns)

    #lower string

    data['text'] = data['text'].str.lower()

    #remove email adress

    data['text'] = data['text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)

    #remove IP address

    data['text'] = data['text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)

    #remove punctaitions and special chracters

    data['text'] = data['text'].str.replace('[^\w\s]','')

    #remove numbers

    data['text'] = data['text'].replace('\d', '', regex=True)

    #remove stop words

    for index, row in data.iterrows():

        word_tokens = word_tokenize(row['text'])

        filtered_sentence = [w for w in word_tokens if not w in stopwords.words('english')]

        df_ = df_.append({"id": row['id'], "target":  row['target'],"text": " ".join(filtered_sentence[0:])}, ignore_index=True)

    data = df_

    return data
disater_data_train = textprocessing(disater_data_train)

disater_data_train.head()
def textprocessing_test(data):

    #Text Preprocessing

    columns = ['id', 'text']

    df_ = pd.DataFrame(columns=columns)

    #lower string

    data['text'] = data['text'].str.lower()

    #remove email adress

    data['text'] = data['text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)

    #remove IP address

    data['text'] = data['text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)

    #remove punctaitions and special chracters

    data['text'] = data['text'].str.replace('[^\w\s]','')

    #remove numbers

    data['text'] = data['text'].replace('\d', '', regex=True)

    #remove stop words

    for index, row in data.iterrows():

        word_tokens = word_tokenize(row['text'])

        filtered_sentence = [w for w in word_tokens if not w in stopwords.words('english')]

        df_ = df_.append({"id": row['id'], "text": " ".join(filtered_sentence[0:])}, ignore_index=True)

    data = df_

    return data
disater_data_test = textprocessing_test(disater_data_test)

disater_data_test.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report



def classification_model(X_train, X_test, y_train):

    #grid search result

    print("X_train:{}".format(len(X_train)))

    vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2') 

    counts = vectorizer.fit_transform(X_train)

    vocab = vectorizer.vocabulary_

    classifier = SGDClassifier(alpha=1e-05,max_iter=100,penalty='elasticnet')

    targets = y_train

    print("X_train:{}".format(len(X_train)))

    print("y_train:{}".format(len(y_train)))

    print("X_test:{}".format(len(X_test)))

    #print(len(y_test))

    classifier = classifier.fit(counts, targets)

    example_counts = vectorizer.transform(X_test)

    #print(example_counts)

    predictions = classifier.predict(example_counts)

    print(predictions)

    len_array = len(X_test)

    return predictions
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(disater_data_train['text'].values.astype('U'),disater_data_train['target'].values.astype('int32'), test_size=0.10, random_state=0)

#classes  = disater_data_train['target'].unique()

#print(X_test)

#classifier,predictions,y_test = classification_model(X_train, X_test, y_train, y_test)

print(disater_data_train['text'].count())

print(disater_data_train['target'].count())

predictions = classification_model(disater_data_train['text'].values.astype('U'), disater_data_test['text'].values.astype('U'), disater_data_train['target'].values.astype('int32'))
y_pred = pd.DataFrame(predictions,columns=['target'])

predictions = pd.concat([disater_data_test['id'],y_pred], axis=1)

predictions.head()