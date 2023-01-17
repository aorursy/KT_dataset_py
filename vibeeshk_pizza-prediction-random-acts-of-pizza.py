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
filepath= '/kaggle/input/random-acts-of-pizza/train.json'

filepath1= '/kaggle/input/random-acts-of-pizza/test.json'



#!/usr/bin/env python

# coding: utf-8





import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import re

import spacy

from nltk.corpus import stopwords 





from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression











#with open(r'C:\Users\Vibeesh\Desktop\kaggle data\Pizza\random-acts-of-pizza\train.json', encoding='utf-8-sig') as f_input:

traindata = pd.read_json(filepath)



#traindata.to_csv('train.csv', encoding='utf-8', index=False)



#with open(r'C:\Users\Vibeesh\Desktop\kaggle data\Pizza\random-acts-of-pizza\test.json', encoding='utf-8-sig') as f_input:

    

testdata = pd.read_json(filepath1)

    

#testdata.to_csv('testdata.csv', encoding='utf-8', index=False)

testdata2=testdata









#We remove some columns in the training dataset so that it contains all the columns of the given test dataset

traindata=traindata[['giver_username_if_known', 'request_id', 'request_text_edit_aware',       'request_title', 'requester_account_age_in_days_at_request',       'requester_days_since_first_post_on_raop_at_request',       'requester_number_of_comments_at_request',       'requester_number_of_comments_in_raop_at_request',       'requester_number_of_posts_at_request',       'requester_number_of_posts_on_raop_at_request',       'requester_number_of_subreddits_at_request',       'requester_subreddits_at_request',       'requester_upvotes_minus_downvotes_at_request',       'requester_upvotes_plus_downvotes_at_request', 'requester_username', 'unix_timestamp_of_request', 'unix_timestamp_of_request_utc','requester_received_pizza']]

#required_columns=testdata.columns.to_list() + ['requester_received_pizza']

testdata.head()





# In[27]:





#Convert the categorical variable requester_received_pizza to numerical 1s and 0s

traindata["requester_received_pizza"] = traindata["requester_received_pizza"].astype(int)

traindata['requester_received_pizza'].head()





# In[28]:





#Remove the unique categrorical variables that arent involved in determining requester_received_pizza



traindata=traindata.drop('giver_username_if_known',axis=1)

traindata=traindata.drop('request_id',axis=1)

traindata=traindata.drop('requester_username',axis=1)





testdata=testdata.drop('giver_username_if_known',axis=1)

testdata=testdata.drop('request_id',axis=1)

testdata=testdata.drop('requester_username',axis=1)









added_df = pd.concat([traindata,testdata])









titlelist=added_df['request_title'].to_list()

lemmatizer = WordNetLemmatizer() 



stop_words = set(stopwords.words('english')) 

sp = spacy.load('en_core_web_sm')

all_stopwords = sp.Defaults.stop_words



titlist=[]

for i1 in titlelist:

    i1 = word_tokenize(i1)

    

    i1= [word for word in i1 if not word in all_stopwords]

    i1 = [w for w in i1 if not w in stop_words]

    i1=' '.join(i1)

    i1=re.sub(r'[^\w\s]','',i1)

    i1=re.sub(r'\W', ' ', i1) 

    i1 = re.sub('[0-9]', '', i1)

    i1=i1.lower()

    i1=lemmatizer.lemmatize(i1)

    i1 = word_tokenize(i1)

  #  i1=i1.pop(0)

    titlist.append(i1)



    

titlist1=[]

for j in titlist:

    j=' '.join(j)

  #  print(j)

    titlist1.append(j)





add1=titlist1

added_df['titles']=add1





sublist=added_df['requester_subreddits_at_request'].to_list()

sublist3=[]

for i in sublist:

    if len(i) is 0:

        i='nonetype'

    elif 'Random_Acts_Of_Pizza' in i:

        i='Racop'

    else:

        i='Nom'

    sublist3.append(i)

    

add3=sublist3

added_df['subs']=add3











textlist=added_df['request_text_edit_aware'].to_list()







titlist=[]

for i1 in textlist:

    i1 = word_tokenize(i1)

    

    i1= [word for word in i1 if not word in all_stopwords]

    i1 = [w for w in i1 if not w in stop_words]

    i1=' '.join(i1)

    i1=re.sub(r'[^\w\s]','',i1)

    i1=re.sub(r'\W', ' ', i1) 

    i1 = re.sub('[0-9]', '', i1)

    i1=i1.lower()

    i1=lemmatizer.lemmatize(i1)

    i1 = word_tokenize(i1)

  #  i1=i1.pop(0)

    titlist.append(i1)



    

titlist1=[]

for j in titlist:

    j=' '.join(j)

 #   print(j)

    titlist1.append(j)





add4=titlist1

added_df['newtexts']=add4











added_df=added_df.drop('request_text_edit_aware',axis=1)

added_df=added_df.drop('request_title',axis=1)

added_df=added_df.drop('requester_subreddits_at_request',axis=1)





#Perform one hot encoding on the Pclass column

one_hot = pd.get_dummies(added_df['subs'])

# Drop column Product as it is now encoded

added_df = added_df.drop('subs',axis = 1)

# Join the encoded df

added_df = added_df.join(one_hot)

added_df.head()









cv=CountVectorizer()

traindata22=cv.fit_transform(added_df['titles'])

traindata22=pd.DataFrame(traindata22.todense())

traindata22.reset_index(drop=True, inplace=True)

added_df.reset_index(drop=True, inplace=True)

added_df1 = pd.concat([added_df,traindata22], axis=1, sort=False)

added_df = added_df1.drop('titles',axis = 1)











cv=CountVectorizer()

traindata22=cv.fit_transform(added_df['newtexts'])

traindata22=pd.DataFrame(traindata22.todense())

traindata22.reset_index(drop=True, inplace=True)

added_df.reset_index(drop=True, inplace=True)

added_df = pd.concat([added_df,traindata22], axis=1, sort=False)

added_df = added_df.drop('newtexts',axis = 1)







added_df = added_df.replace(np.nan, 0, regex=True)

#added_df

traindata=added_df.head(4040)

testdata=added_df.tail(1630)

y=traindata['requester_received_pizza']

x=traindata.drop('requester_received_pizza',axis=1)

testdata=testdata.drop('requester_received_pizza',axis=1)

#Splitting training and testing data

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)









#Logistic Regression

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x, y)

y_predicted = LogisticRegressor.predict(testdata)







y_predicted
predictionlist=y_predicted.tolist()

texts=testdata2['request_id'].tolist() 

output=pd.DataFrame(list(zip(texts, predictionlist)),

              columns=['request_id','requester_received_pizza'])

output.head()

output.to_csv('my_submissionPizza.csv', index=False)
##Now we test the accuracy using the training data:



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)









#Logistic Regression

LogisticRegressor = LogisticRegression(max_iter=10000)

LogisticRegressor.fit(x_train, y_train)

y_predicted = LogisticRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("Mean Absolute Error:",mae)

print('accuracy score:')

print(accuracy_score(y_test,y_predicted))


