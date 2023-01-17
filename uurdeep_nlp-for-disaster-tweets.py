# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv",encoding="latin1")

data=pd.concat([data.text,data.target],axis=1)

data.dropna(axis=0,inplace=True)
CleanData=[]

def prepare(data,out):

    for descr in data.text:

        descr=re.sub("[^a-zA-Z]"," ",descr) #harf olMAyanları bul

        descr=descr.lower()

        descr = nltk.word_tokenize(descr)

        descr = [word for word in descr if not word in set(stopwords.words("english"))] 

        lemma = nltk.WordNetLemmatizer()

        descr = [lemma.lemmatize(word) for word in descr]

        descr = " ".join(descr)

        out.append(descr)

    

prepare(data,CleanData)
max_features= 300



count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix=count_vectorizer.fit_transform(CleanData).toarray()

print("Most used {} word: {}".format(max_features,count_vectorizer.get_feature_names()))

#%%

y= data.iloc[:,1].values # Disaster or not

x=sparce_matrix #Texts for training

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.01,random_state=42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state=42,max_iter=100, C=10)

logreg.fit(x_train,y_train.T)
#Used callback function to save best weights



from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.layers.core import Dropout

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacklists=[checkpoint]

model = Sequential()

model.add(Dense(input_dim=x_train.shape[1],

                output_dim = 1,

                init =   'uniform',

                activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(256, kernel_initializer='uniform'))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='uniform'))

model.add(Activation('relu'))

model.add(Dense(1, kernel_initializer='uniform'))

model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["acc"])

model.fit(x_train,

          y_train,

          epochs = 35,

          batch_size = 50,

          validation_data = (x_test,y_test),

          callbacks=callbacklists,

          verbose=1)

model.load_weights("best.hdf5")
print(model.summary())
print("Test accuracy of naive bayes: ",nb.score(x_test,y_test))

print("Test accuracy of Logistic Regression:  ",logreg.score(x_test,y_test.T))

print("Test loss and accuracy of Neural Network: ",model.evaluate(x_test,y_test))
#%%Small Visualization

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import seaborn as sns



y_axis=[nb.score(x_test,y_test),logreg.score(x_test,y_test.T),model.evaluate(x_test,y_test)[1]]

x_axis=["Naive Bayes","Logistic Regression","Neural Network"]



fig,ax=plt.subplots(figsize=(10,6))

ax.bar(x_axis,y_axis)

plt.show()

def print_confusion_matrix(confusion_matrix, class_names, figsize = (5,5), fontsize=15):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d",cmap='Blues')

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



result_Float=model.predict(x_test)

result_bin=[]

for num in result_Float:

    result_bin.append(int(num>=0.5))



cf_matrix=confusion_matrix(y_test,result_bin)

print_confusion_matrix(cf_matrix,['1','2'])
sub=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

x_lastTest=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

CleanTestData=[]

prepare(x_lastTest,CleanTestData)

x_lastTest=count_vectorizer.transform(CleanTestData).toarray()
sub["target"]=nb.predict(x_lastTest)

sub.set_index(["id"],inplace=True)

sub.to_csv("cevap.csv")