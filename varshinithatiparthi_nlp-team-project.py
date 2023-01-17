
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS

#Data Visulaization
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_train.head()
#unique occurance in keyword column
df_train['keyword'].value_counts()
#unique occurance location column
df_train['location'].value_counts()
#checking for null values
df_train.isnull().sum()
#keyword column
sns.barplot(y=df_train['keyword'].value_counts()[:20].index,x=df_train['keyword'].value_counts()[:20],
            orient='k')
df_train['location'].replace({'United States':'USA',
                           'New York':'USA',
                            "London":'UK',
                            "Los Angeles, CA":'USA',
                            "Washington, D.C.":'USA',
                            "California":'USA',
                             "Chicago, IL":'USA',
                             "Chicago":'USA',
                            "New York, NY":'USA',
                            "California, USA":'USA',
                            "FLorida":'USA',
                            "Nigeria":'Africa',
                            "Kenya":'Africa',
                            "Everywhere":'Worldwide',
                            "San Francisco":'USA',
                            "Florida":'USA',
                            "United Kingdom":'UK',
                            "Los Angeles":'USA',
                            "Toronto":'Canada',
                            "San Francisco, CA":'USA',
                            "NYC":'USA',
                            "Seattle":'USA',
                            "Earth":'Worldwide',
                            "Ireland":'UK',
                            "London, England":'UK',
                            "New York City":'USA',
                            "Texas":'USA',
                            "London, UK":'UK',
                            "Atlanta, GA":'USA',
                            "Mumbai":"India"},inplace=True)
#location 

sns.barplot(y=df_train['location'].value_counts()[:5].index,x=df_train['location'].value_counts()[:5],
            orient='h')
df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train.info()
df=df_train.drop(['location','keyword','id'],axis=1)
df.head()
graph=pd.value_counts(df['target']).plot.bar()
graph
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
dis_words=' '.join(list(df[df['target']==1]['text']))
dis= WordCloud(width=512,height=512).generate(dis_words)
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(dis)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
import nltk
from textblob import TextBlob
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import string
import re
def preprocessing(txt):
    
    txt= re.sub(r"http\S+", "", txt)  #http tags
    txt= re.sub('@[^\s]+','',txt)   #special characters
    # stemming
    stemmer = nltk.PorterStemmer()
    txt=[stemmer.stem(word) for word in txt.split()]
    txt=" ".join(str(v) for v in txt)   
    p = string.digits + string.punctuation
    table = str.maketrans(p, len(p)*" ")
    txt = txt.translate(table)
    txt = txt.lower()
    
    return txt

for index, row in df.iterrows():
    df.at[index,'text'] = remove_emoji(row['text'])
    df.at[index,'text']=preprocessing(row['text'])
dis_words=' '.join(list(df[df['target']==1]['text']))
dis= WordCloud(width=512,height=512).generate(dis_words)
plt.figure(figsize=(10,8),facecolor='k')
plt.imshow(dis)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
from nltk import tokenize  #tokenizing
from nltk.corpus import stopwords   #stopwords

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#tokenization and stopwords

def preprocess_tweet(text):
    tokenized_text = tokenizer.tokenize(text)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

df['text'] = df['text'].apply(lambda x: preprocess_tweet(x))
print('After removing stopwords: ', df['text'][6])
#count vectorization
countvectorizer = CountVectorizer(stop_words = 'english', min_df = 2)
countvectorizer.fit(df['text'])
countvetor = countvectorizer.transform(df['text'])


#Hash vectorization
import sklearn.feature_extraction.text as txt

Hashvectorizer = txt.HashingVectorizer(stop_words='english', binary=False, norm=None,alternate_sign=False)
Hashvectorizer.fit(df_train['text'])
Hashvetor = Hashvectorizer.transform(df_train['text'])
#Tfidf vectorization
Tfidfvectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
Tfidfvectorizer.fit(df['text'])
Tfidfvetor = Tfidfvectorizer.transform(df['text'])

test_tfidf = Tfidfvectorizer.transform(df_test['text'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Tfidfvetor,df['target'], random_state=999)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
a_space=[0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8,0.9,1,10,25,50,56,70,100]
param_grid = {'alpha': a_space}
MNBclf = MultinomialNB()
MNBccc= GridSearchCV(MNBclf, param_grid, cv=5)
MNBccc.fit(Hashvetor,df['target'])
print("Tuned alpha Parameters: {}".format(MNBccc.best_params_)) 
print("Best score is {}".format(MNBccc.best_score_))
from sklearn.naive_bayes import MultinomialNB
MNBclf = MultinomialNB()
MNBclf.fit(X_train,y_train)
#train score
MNBclf.score(X_train,y_train)
#test score
MNBclf.score(X_test,y_test)
from sklearn.naive_bayes import BernoulliNB
NBclf = BernoulliNB()
NBclf.fit(X_train,y_train)
#train score
NBclf.score(X_train,y_train)
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(max_iter=1000)
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
from sklearn.svm import SVC
SVClassifier = SVC()
SVClassifier.fit(X_train,y_train)
SVClassifier.score(X_train,y_train)
SVClassifier.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 1000,random_state = 55)
clf_rf.fit(X_train,y_train)
ran_pred = clf_rf.predict(X_test)
accuracy_score(y_test,ran_pred)
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators = 2000,random_state = 42,max_depth = 10)
clf.fit(X_train,y_train)

clf.score(X_train,y_train)

clf.score(X_test,y_test)
from xgboost import XGBClassifier
alg = XGBClassifier(learning_rate = 0.1, n_estimators = 2400, max_depth = 10,
                        min_child_weight = 3, gamma = 0.2, subsample = 0.6, colsample_bytree = 1.0,
                        objective ='binary:logistic', nthread = 8, scale_pos_weight = 1, seed = 42)
alg.fit(X_train,y_train)
alg.score(X_train,y_train)
alg.score(X_test,y_test)

def submission(submission_file_path,model,test_vectors):
    sample_submission = pd.read_csv(submission_file_path)
    sample_submission["target"] = model.predict(test_vectors)
    sample_submission.to_csv("submission.csv", index=False)
submission_file_path = "../input/nlp-getting-started/sample_submission.csv"
test_vectors=test_tfidf
submission(submission_file_path,SVClassifier,test_vectors)
