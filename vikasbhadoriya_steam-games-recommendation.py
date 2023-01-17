import numpy as np

import pandas as pd
df_train = pd.read_csv("../input/train.csv")

df_overview = pd.read_csv("../input/game_overview.csv")

df_test = pd.read_csv("../input/test.csv")
df_merged_train = pd.merge(df_train,df_overview,on="title",how="left")

df_merged_train["Origin"] = "Train"

df_merged_test = pd.merge(df_test,df_overview,on = "title",how = "left")

df_merged_test["Origin"] = "Test"
df_merged_train.developer.nunique()

df_overview["overview"][63]
df_train_test = pd.concat([df_merged_train,df_merged_test],axis=0,sort = False)
df_train_test.head()
import re 
df_train_test.columns
df_train_test['year'] = df_train_test['year'].fillna(df_train_test.groupby(['title'])['year'].transform('median'))





import nltk 

nltk.download('wordnet')
import string

import nltk

nltk.download('stopwords')



from nltk.corpus import stopwords

stopword = stopwords.words("english")

from nltk.stem import WordNetLemmatizer

lemmateizer = WordNetLemmatizer()
df_train_test.reset_index(inplace=True)
df_train_test.columns
df_train_test.drop("index",inplace=True,axis = 1)
corpus = []

for x in range(0,df_train_test['overview'].count()):

    remv_num = re.sub('[^A-Za-z]',' ',df_train_test['overview'][x])

    remv_punct = "".join([word.lower() for word in remv_num if word not in string.punctuation])

    tokens = re.split("\W+",remv_punct)

    

    noStop = [lemmateizer.lemmatize(word) for word in tokens if word not in stopword]

    

    noStop = " ".join(noStop)

    

    corpus.append(noStop)
len(corpus)
corpus1 = []

for x in range(0,df_train_test['user_review'].count()):

    remv_num = re.sub('[^A-Za-z]',' ',df_train_test['user_review'][x])

    remv_punct = "".join([word.lower() for word in remv_num if word not in string.punctuation])

    tokens = re.split("\W+",remv_punct)

    

    noStop = [lemmateizer.lemmatize(word) for word in tokens if word not in stopword]

    

    noStop = " ".join(noStop)

    

    corpus1.append(noStop)
len(corpus1)
corpus2 = []

for i in range(df_train_test.tags.count()):

    rem_punct = [word for word in df_train_test.tags[i] if word not in string.punctuation]

    cor = "".join([word.lower() for word in rem_punct])

    corpus2.append(cor)
len(corpus2)
corpus_dict = dict({"overview":corpus,"user_review":corpus1,"tags":corpus2})
df_corpus = pd.DataFrame(data=corpus_dict)
df_corpus.shape
df_corpus['mix_corpus'] = df_corpus.tags

for i in range(0,df_corpus.overview.count()):

    df_corpus['mix_corpus'][i] = ''.join(df_corpus['tags'][i]) +' '+ ''.join(df_corpus['user_review'][i])+' '+''.join(df_corpus['overview'][i])

df_corpus.tags[0]
df_corpus.overview[0]
df_corpus.user_review[0]
df_corpus.mix_corpus[1000]
df_corpus["overview_word_count"] = df_corpus['overview'].apply(lambda x : len(x.split(" ")))

df_corpus["tags_word_count"] = df_corpus['tags'].apply(lambda x : len(x.split(" ")))

df_corpus["user_review_word_count"] = df_corpus['user_review'].apply(lambda x : len(x.split(" ")))
df_corpus['year'] = df_train_test['year'].copy()

df_corpus['user_suggestion'] = df_train_test['user_suggestion'].copy()

df_corpus['publisher'] = df_train_test['publisher'].copy()

final_corpus = df_corpus.mix_corpus
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

cv=CountVectorizer(max_features=7000)

X=cv.fit_transform(final_corpus).toarray()
df_corpus["Origin"]= df_train_test.Origin
X
Xfeatures = pd.concat([df_corpus['year'],df_corpus['user_suggestion'],df_corpus["publisher"],df_corpus['overview_word_count'],df_corpus['Origin'],df_corpus['tags_word_count'],df_corpus['user_review_word_count'],pd.DataFrame(X)],axis =1)

dummies = pd.get_dummies(df_corpus["publisher"])
Xfeatures = pd.concat([Xfeatures,dummies],axis = 1)
Xfeatures.shape
Xfeatures.drop("publisher",axis = 1,inplace = True)
Xfeatures.shape
Xfeatures_train = Xfeatures[Xfeatures.Origin == "Train"]

Xfeatures_test = Xfeatures[Xfeatures.Origin == "Test"]
Xfeatures_train.shape,Xfeatures_test.shape
Xfeatures_train.drop("Origin",axis = 1,inplace = True)
Xfeatures_test.drop("Origin",axis = 1,inplace = True)
Xfeatures_test.shape
Xfeatures_test.drop("user_suggestion",axis = 1,inplace=True)
X = Xfeatures_train.drop("user_suggestion",axis = 1)

y = Xfeatures_train.user_suggestion
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200,criterion='entropy',random_state=3)

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score
print("Train accuracy score:",accuracy_score(y_train,rfc.predict(X_train)))

print("Train F1 score:",f1_score(y_train,rfc.predict(X_train)))
print("Test accuracy score:",accuracy_score(y_test,y_pred))

print("Test F1 score:",f1_score(y_test,y_pred))
test_pre = rfc.predict(Xfeatures_test)
test_pre
review_id = df_test.review_id
review_id.shape
d1 = dict({"review_id":review_id,"user_suggestion":test_pre})
test_pred_df = pd.DataFrame(data=d1)
test_pred_df.user_suggestion = test_pred_df.user_suggestion.astype('int')
test_pred_df.head()
test_pred_df.to_csv("submission6.csv",index = False)