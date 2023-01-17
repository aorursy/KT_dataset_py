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
# !wget https://github.com/jpovmarques/twitter-gender-predictor/commits/master/dataset/data.json

# !wget https://github.com/jpovmarques/twitter-gender-predictor/raw/master/dataset/target.json
import json

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB





count_vect = CountVectorizer()

tfidf_transformer = TfidfTransformer()



def load_data():

  with open('/kaggle/working/data.json') as data_file:

    data = json.load(data_file)



  with open('/kaggle/working/target.json') as target_file:

    target = json.load(target_file)



  return data, target



def get_classifier(data, target):

  X_train_counts = count_vect.fit_transform(data)

  X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

  

  classifier = MultinomialNB().fit(X_train_tfidf, target)



  return classifier



def get_gender(twitter_message, classifier):

  X_new_counts = count_vect.transform(twitter_message)

  X_new_tfidf = tfidf_transformer.transform(X_new_counts)



  return classifier.predict(X_new_tfidf)[0]
data = pd.read_csv('../input/gc4classes/gender-classifier-data.csv')

data.dropna(inplace=True,axis=0)



data.gender = [0 if each == "female" else 1 for each in data.gender] 



inputs_d = data.description

inputs_n = data.name

inputs_t = data.text



target = data.gender
inn = data.description
data2 = pd.read_csv('../input/gc4classes/gender-classifier-test.csv')



in_d = data2.description

in_n = data2.name

in_t = data2.text
def predict(inputs,target,inp):

    ans = []

    classifier = get_classifier(inputs, target)

    for msg in range(len(inp)):

        gender = get_gender([format(inp[msg])], classifier)

        ans.append([gender])

        

    return ans

    
ans1 = predict(inputs_d,target,in_d)

ans2 = predict(inputs_n,target,in_n)

ans3 = predict(inputs_t,target,in_t)
Ans = []

for i in range(len(ans1)):

    if (ans1[i][0]+ans2[i][0]+ans3[i][0])/3.0 >= 0.5:

        Ans.append([i+1,1])

    else:

        Ans.append([i+1,0])
sub =  pd.DataFrame(Ans)

sub = sub.rename(index=str, columns={0: "no.", 1: "gender"})

words = set((' '.join(data.description)).split())

print(len(words))
sub
sub.to_csv('submission.csv', index=False)