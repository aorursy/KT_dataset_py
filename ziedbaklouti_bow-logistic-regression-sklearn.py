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
test_set=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

test_set = test_set.fillna('-')

train_set=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

train_set = train_set.fillna('-')
test_set
train_set
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS,CountVectorizer

from sklearn.metrics import f1_score

vect1 = CountVectorizer(token_pattern=r'\w{3,10}',ngram_range=(1, 1),max_features=23000, stop_words=ENGLISH_STOP_WORDS).fit(train_set.text)

#vect2 = TfidfVectorizer(ngram_range=(1, 1),max_features=1000, stop_words=ENGLISH_STOP_WORDS).fit(train_set.text)

X1_txt = vect1.transform(test_set.text)

#X2_txt = vect2.transform(train_set.text)

X1=pd.DataFrame(X1_txt.toarray(), columns=vect1.get_feature_names())

#X2=pd.DataFrame(X2_txt.toarray(), columns=vect2.get_feature_names())

y=test_set.sentiment

#X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0, random_state=42)

#X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0, random_state=42)

log_reg1 = LogisticRegression(C=1500).fit(X1, y)

#log_reg2 = LogisticRegression().fit(X2, y)

#print('Accuracy on train set with BOW: ', log_reg1.score(X1_train, y1_train))

#print('Accuracy on test set with BOW: ', log_reg1.score(X1_test, y1_test))

#print('Accuracy on train set with TFIDF: ', log_reg2.score(X2_train, y2_train))

#print('Accuracy on test set with TFIDF: ', log_reg2.score(X2_test, y2_test))

y1_predicted = log_reg1.predict(X1)

#y2_predicted = log_reg2.predict(X2)
from sklearn.metrics import accuracy_score

print('Accuracy score test set BOW: ', accuracy_score(test_set.sentiment, y1_predicted))
#from sklearn.metrics import accuracy_score

#print('Accuracy score test set BOW: ', accuracy_score(test_set.sentiment, y1_predicted))
prob = log_reg1.predict_proba(X1)
ii=[i for i in np.where(X1_txt.toarray()[1,:]>=1)[0]]
indexes=[np.where(X1_txt.toarray()[index,:]>=1)[0] for index in range(len(X1_txt.toarray()))]
features=[]

for j in range(len(indexes)):

    tokens=[vect1.get_feature_names()[i] for i in indexes[j]]

    features.append(tokens)
#features
coeff1=[]

coeff2=[]

coeff3=[]

for j in range(len(indexes)):

    tokens1=[log_reg1.coef_[0][i] for i in indexes[j]]

    coeff1.append(tokens1)

    tokens2=[log_reg1.coef_[1][i] for i in indexes[j]]

    coeff2.append(tokens2)

    tokens3=[log_reg1.coef_[2][i] for i in indexes[j]]

    coeff3.append(tokens3)
#coeff1
list1=[]

list2=[]

list3=[]

for j in range(len(coeff1)):

    index1=[i for i in range(len(coeff1[j])) if coeff1[j][i]>0]

    list1.append(index1)

    index2=[i for i in range(len(coeff2[j])) if coeff2[j][i]>0]

    list2.append(index2)

    index3=[i for i in range(len(coeff3[j])) if coeff3[j][i]>0]

    list3.append(index3)
#list1
combine=zip(list1,list2,list3)
combined_list=[*combine]
index=[np.where(max(prob[i])==prob[i])[0][0] for i in range(len(prob))]
total_selected_tokens=[]

for j in range(len(features)):

    if index[j]!=1:

        selected_tokens=[features[j][i] for i in combined_list[j][index[j]]]

    else:

        selected_tokens=[test_set.text[j]]

    total_selected_tokens.append(selected_tokens)
selected_text_predicted=[' '.join(str) for str in total_selected_tokens ]
#selected_text_predicted
test_set
submission=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
submission["selected_text"]=selected_text_predicted
submission.to_csv('submission.csv', index=False)