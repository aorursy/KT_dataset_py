import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

from sklearn.model_selection import train_test_split



#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submit = pd.read_csv('../input/sample_submission.csv')
train.head()
X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train['identity_hate'], 

                                                    random_state=0)
x = len(train['comment_text'])

print(x)

lenth = train.comment_text.str.len()

#print(lenth)

print(lenth.mean())

print(lenth.max())

lenth.hist();
X_train.head()
#label_cols = ['id','toxic','severe_toxic','obscene','threat','insult','identity_hate']

train.describe()
len(train),len(test)
COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score



texts = train['comment_text'].T.tolist()

cv = CountVectorizer()

cv_fit = cv.fit_transform(texts)

# convert the dataframe to list for the following processing 

texts = X_train.T.tolist()



# cv train using X_train 

vect = CountVectorizer().fit(X_train)

    

# transform the texts in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)



# Train the model

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)



# Predict the transformed test documents

predictions = model.predict(vect.transform(X_test))

roc_AUS_score = roc_auc_score(y_test, predictions)

    

print(roc_AUS_score)

from sklearn.feature_extraction.text import TfidfVectorizer



# Fit the TfidfVectorizer to the training data

vect = TfidfVectorizer().fit(X_train)

    

# transform the texts in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)



# find the tfidf value and order the tf_idf_index by importance  

values =  X_train_vectorized.max(0).toarray()[0]

index = vect.get_feature_names()

    

# convert the list to the Series required

features_series = pd.Series(values,index = index)



print(features_series.nsmallest(20),features_series.nlargest(20))
# Fit the TfidfVectorizer to the training data

vect = TfidfVectorizer(min_df=3).fit(X_train)

    

# transform the texts in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)



# Train the model

model = MultinomialNB(alpha=0.1)

model.fit(X_train_vectorized, y_train)



# Predict the transformed test documents

predictions = model.predict(vect.transform(X_test))



print(roc_auc_score(y_test, predictions))
submid = pd.DataFrame({'id': submit["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)