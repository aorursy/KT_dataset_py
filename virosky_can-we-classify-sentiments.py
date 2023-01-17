# Data Manipulation & Visualization

import os

import pandas as pd

import numpy as np

import seaborn as sns # used for plot interactive graph. 

sns.set_style('darkgrid')

import matplotlib.pyplot as plt



# Text Manipulation

import re

from wordcloud import STOPWORDS

from nltk import FreqDist, word_tokenize

from nltk import bigrams, trigrams

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

stopwords = set(STOPWORDS)



# Machine Learning

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,log_loss,accuracy_score,classification_report

from sklearn.metrics import precision_score

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train = pd.read_csv("../input/product-sentiment-classification/Participants_Data/Train.csv")

test = pd.read_csv("../input/product-sentiment-classification/Participants_Data/Test.csv")
# we take only what we need

train = train[['Product_Description','Product_Type','Sentiment']]
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(17,5))

sns.countplot(data=train,x="Product_Type",hue="Sentiment",edgecolor="black",ax=ax[0],linewidth=2)

ax[0].legend(loc="upper left",labels=["Cannot say","Negative","Positive","No sentiment"])

ax[0].set_title('Sentiment by product in train set',size=17)

ax[0].set_xlabel("Product type")



sns.countplot(data=train,x="Sentiment",edgecolor="black",ax=ax[1],linewidth=2)

ax[1].legend(loc="upper left")

ax[1].set_title('Sentiment distribution in train set',size=17)

ax[1].set_xticklabels(["Cannot say","Negative","Positive","No sentiment"])

ax[1].set_xlabel("")



plt.show() 
test_str = train.loc[0, 'Product_Description']



def clean_text(text):

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text=  re.sub('@mention',' ', text )

    text=  re.sub('{link}',' ', text )

    text=  re.sub('Ûª',' ', text )

    text=  re.sub('  ',' ', text )

    text=  re.sub('RT',' ', text )

    text=  re.sub('//',' ', text )

    text=  re.sub('&quot',' ', text )

    text=  re.sub('&amp',' ', text )

    text=  re.sub(r'[^\w\s]',' ', text )

    text=  re.sub(' +',' ', text )

    return text



def process_text(df):

    df['description'] = df['Product_Description'].apply(lambda x: clean_text(x))

    return df



print("Original text: " + test_str)

print("Cleaned text: " + clean_text(test_str))

train = process_text(train)

test = process_text(test)

train.drop('Product_Description',1,inplace=True)

test.drop('Product_Description',1,inplace=True)
plt.figure(figsize=(10,5))

word_freq = FreqDist(w for w in word_tokenize(' '.join(train['description']).lower()) if 

                     (w not in stopwords) & (w.isalpha()))

df_word_freq = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])

top20w = df_word_freq.sort_values('count',ascending=False).head(5)

last20w=df_word_freq.sort_values('count',ascending=False).tail(5)



sns.barplot(top20w['count'],top20w.index,color='purple',edgecolor="black",linewidth=2)

plt.title("Top 5 words in train",size=17)

plt.show()
fig,axes=plt.subplots(ncols=2,figsize=(17,5),dpi=100)

###bigrams

bigram = list(bigrams([w for w in word_tokenize(' '.join(train['description']).lower()) if 

              (w not in stopwords) & (w.isalpha())]))

fq = FreqDist(bg for bg in bigram)

bgdf = pd.DataFrame.from_dict(fq, orient='index', columns=['count'])

bgdf.index = bgdf.index.map(lambda x: ' '.join(x))

bgdf = bgdf.sort_values('count',ascending=False)



#trigrams

trigram = list(trigrams([w for w in word_tokenize(' '.join(train['description']).lower()) if 

              (w not in stopwords) & (w.isalpha())]))

tr_fq = FreqDist(bg for bg in trigram)

trdf = pd.DataFrame.from_dict(tr_fq, orient='index', columns=['count'])

trdf.index = trdf.index.map(lambda x: ' '.join(x))

trdf = trdf.sort_values('count',ascending=False)



sns.barplot(bgdf.head(10)['count'], bgdf.index[:10], ax=axes[1],color='green',edgecolor='black',linewidth=2)

sns.barplot(trdf.head(10)['count'], trdf.index[:10],ax=axes[0], color='red',edgecolor='black',linewidth=2)



axes[0].set_title('Top 10 Trigrams',size=18)

axes[1].set_title('Top 10 Bigrams',size=18)



plt.show()
d = '../input/masks-for-wordcloud/'

comments_mask = np.array(Image.open(d + 'oval.jpg'))



long_string = ','.join(list(train['description'].values))

wordcloud = WordCloud(background_color='black',max_words=500, contour_width=5, contour_color='black',

                      width=1000,height=200,stopwords=stopwords,mask=comments_mask)

wordcloud.generate(str(long_string))

wordcloud.to_image()
from sklearn.model_selection import train_test_split



X = train['description']

y = train['Sentiment']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
nb = Pipeline([('vect', TfidfVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', MultinomialNB())

              ])



nb.fit(X_train, y_train)



y_pred = nb.predict(X_test)



print("_"*25+"Classification Report"+"_"*25)

print(classification_report(y_pred,y_test,target_names=['Cannot say','Negative','Positive','No sentiment'],zero_division=0))

print("_"*25+"Evaluation Metrics"+"_"*25)

print("\n")

print("Accuracy: %f" % accuracy_score(y_pred,y_test))

print("Weighted Precision :%f" % precision_score(y_pred,y_test,average="weighted"))



cm=confusion_matrix(y_pred,y_test)

g=sns.heatmap(cm,annot=True,fmt='d',linewidths=1,linecolor='black',

                  annot_kws={"size":14},cmap='viridis',cbar=False)

g.set_xticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 15,rotation=45)

g.set_yticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 8,rotation=45)



plt.xlabel('Actual',size=16)

plt.ylabel('Predicted',size=16)

plt.title('Confusion Matrix \n Naive Bayes',size=16)

plt.show()
lr = Pipeline([('vect', TfidfVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', LogisticRegression())

              ])



lr.fit(X_train, y_train)



y_pred = lr.predict(X_test)



print("_"*25+"Classification Report"+"_"*25)

print(classification_report(y_pred,y_test,target_names=['Cannot say','Negative','Positive','No sentiment'],zero_division=0))

print("_"*25+"Evaluation Metrics"+"_"*25)

print("\n")

print("Accuracy: %f" % accuracy_score(y_pred,y_test))

print("Weighted Precision :%f" % precision_score(y_pred,y_test,average="weighted"))



cm=confusion_matrix(y_pred,y_test)

g=sns.heatmap(cm,annot=True,fmt='d',linewidths=1,linecolor='black',

                  annot_kws={"size":14},cmap='Spectral',cbar=False)

g.set_xticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 15,rotation=45)

g.set_yticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 8,rotation=45)



plt.xlabel('Actual',size=16)

plt.ylabel('Predicted',size=16)

plt.title('Confusion Matrix \n Logistic Regression',size=16)

plt.show()
lsvc = Pipeline([('vect', TfidfVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', LinearSVC())

              ])



lsvc.fit(X_train, y_train)



y_pred = lsvc.predict(X_test)



print("_"*25+"Classification Report"+"_"*25)

print(classification_report(y_pred,y_test,target_names=['Cannot say','Negative','Positive','No sentiment'],zero_division=0))

print("_"*25+"Evaluation Metrics"+"_"*25)

print("\n")

print("Accuracy: %f" % accuracy_score(y_pred,y_test))

print("Weighted Precision :%f" % precision_score(y_pred,y_test,average="weighted"))





cm=confusion_matrix(y_pred,y_test)

g=sns.heatmap(cm,annot=True,fmt='d',linewidths=1,linecolor='black',

                  annot_kws={"size":14},cmap='Blues',cbar=False)

g.set_xticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 15,rotation=45)

g.set_yticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 8,rotation=45)



plt.xlabel('Actual',size=16)

plt.ylabel('Predicted',size=16)

plt.title('Confusion Matrix \n Linear SVC',size=16)

plt.show()
rf = Pipeline([('vect', TfidfVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', RandomForestClassifier(n_estimators=300))

              ])



rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)



print("_"*25+"Classification Report"+"_"*25)

print(classification_report(y_pred,y_test,target_names=['Cannot say','Negative','Positive','No sentiment'],zero_division=0))

print("_"*25+"Evaluation Metrics"+"_"*25)

print("\n")

print("Accuracy: %f" % accuracy_score(y_pred,y_test))

print("Weighted Precision :%f" % precision_score(y_pred,y_test,average="weighted"))





cm=confusion_matrix(y_pred,y_test)

g=sns.heatmap(cm,annot=True,fmt='d',linewidths=1,linecolor='black',

                  annot_kws={"size":14},cmap='BuGn',cbar=False)

g.set_xticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 15,rotation=45)

g.set_yticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 8,rotation=45)



plt.xlabel('Actual',size=16)

plt.ylabel('Predicted',size=16)

plt.title('Confusion Matrix \n Random Forest',size=16)

plt.show()
xgb = Pipeline([('vect', TfidfVectorizer()),

               ('tfidf', TfidfTransformer()),

               ('clf', XGBClassifier(objective="multi:softmax",n_estimators=200,learning_rate=0.01))

              ])



xgb.fit(X_train, y_train)



y_pred = xgb.predict(X_test)



print("_"*25+"Classification Report"+"_"*25)

print(classification_report(y_pred,y_test,target_names=['Cannot say','Negative','Positive','No sentiment'],zero_division=0))

print("_"*25+"Evaluation Metrics"+"_"*25)

print("\n")

print("Accuracy: %f" % accuracy_score(y_pred,y_test))

print("Weighted Precision :%f" % precision_score(y_pred,y_test,average="weighted"))





cm=confusion_matrix(y_pred,y_test)

g=sns.heatmap(cm,annot=True,fmt='d',linewidths=1,linecolor='black',

                  annot_kws={"size":14},cmap='Set1',cbar=False)

g.set_xticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 15,rotation=45)

g.set_yticklabels(['Cannot say','Negative','Positive','No sentiment'],fontsize = 8,rotation=45)



plt.xlabel('Actual',size=16)

plt.ylabel('Predicted',size=16)

plt.title('Confusion Matrix \n XGB Classifier',size=16)

plt.show()
test['Sentiment']= nb.predict(test.description)
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(17,5))

sns.countplot(data=test,x="Product_Type",hue="Sentiment",edgecolor="black",ax=ax[0],linewidth=2)

ax[0].legend(loc="upper left",labels=["Negative","Positive","No sentiment"])

ax[0].set_title('Sentiment by Product in test set',size=17)

ax[0].set_xlabel("Product type")



sns.countplot(data=test,x="Sentiment",edgecolor="black",ax=ax[1],linewidth=2)

ax[1].legend(loc="upper left")

ax[1].set_title('Sentiment test set',size=17)

ax[1].set_xticklabels(["Negative","Positive","No sentiment"])

ax[1].set_xlabel("")



plt.show()
submission = pd.DataFrame(nb.predict_proba(test.description))

submission.head()
submission.to_csv('sample_submission.csv',index=False)