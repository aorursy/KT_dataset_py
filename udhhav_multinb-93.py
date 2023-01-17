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
import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

from sklearn.linear_model import PassiveAggressiveClassifier



sns.set(style="darkgrid")

df_true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')

df_false = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
df_true.head()
df_false.head()
df_true['label'] = 1

df_false['label'] =0

df = pd.concat([df_true,df_false])
df.head()
df.shape
df.iloc[5]
df.columns
df.isnull().sum()
df.info()
df['label'].value_counts()
plt.figure(figsize =(15,10))

sns.countplot(df['subject'])
wordcloud1 = WordCloud().generate(' '.join(df['text']))
text=list(df['text'].dropna().unique())

fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])

wordcloud2 = WordCloud().generate(" ".join(text))

ax2.imshow(wordcloud2,interpolation='bilinear')

ax2.axis('off')
texts = df.copy()

texts.head(2)
texts.iloc[5]
texts['title'].iloc[5]
ps=PorterStemmer()

corpus=[]

for i in range(len(texts)):

    review=re.sub('[^a-zA-Z]',' ',texts['title'].iloc[i])

    review=review.lower()

    review=review.split()

    

    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]

    review=' '.join(review)

    corpus.append(review)
corpus[:5]
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))

X = cv.fit_transform(corpus).toarray()


y=texts['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
cv.get_feature_names()[:10]
cv.get_params()
countdf=pd.DataFrame(X_train,columns=cv.get_feature_names())

countdf.head()


def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score=metrics.accuracy_score(y_test,pred)

print("Accuracy: %0.3f"%score)
import itertools

cm=metrics.confusion_matrix(y_test,pred)

plot_confusion_matrix(cm,classes=['FAKE','REAL'])
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(X_train,y_train)

pred=linear_clf.predict(X_test)

score= metrics.accuracy_score(y_test,pred)

print('accuracy: %0.3f'%score)

cm=metrics.confusion_matrix(y_test,pred)

plot_confusion_matrix(cm,classes=['Fake Data','Real Data'])
classifier = MultinomialNB(alpha=0.1)
previous_score=0

for alpha in np.arange(0,1,0.1):

    sub_classifier=MultinomialNB(alpha=alpha)

    sub_classifier.fit(X_train,y_train)

    y_pred=sub_classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    if score>previous_score:

        classifier=sub_classifier

    print("Alpha: {}, Score : {}".format(alpha,score))
feature_names = cv.get_feature_names()
classifier.coef_[0]