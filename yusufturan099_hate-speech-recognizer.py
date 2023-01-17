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
# %% [code]
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:13:34 2018

@author: kcy
"""
import pandas as pd
import numpy as np

#import twitter data
data = pd.read_csv(r"/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv",encoding = "latin1")

data_drop = data[data.label==0].index                                    # dropping 0 labelled tweets to balance our dataset
data = data.drop(data_drop[0:22000]).reset_index().drop("index",axis=1)


import re                                                                # Regular Expression Library
import nltk as nlp  
nlp.download("stopwords")                                                # Download stopwords (Irrelevant Words) to folder named "Corpus"
from nltk.corpus import stopwords                                        # Import Stopwords we downloaded

nlp.download('punkt')

description_list = []                                                    # We will put all words' last version into this list.
for description in data.tweet:
    description = re.sub("[^a-zA-Z]"," ",description)                    # Drop all characters excluded letters(a-z) and replace them whith " " (space)
    description = description.lower()                                    # Turn all letters to lowercase
    description = nlp.word_tokenize(description)                         # Advance version of Split Funtion
    lemma = nlp.WordNetLemmatizer()                                      # Sort words by grouping inflected or variant forms of the same word.
    description = [ lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

#bag of words

from sklearn.feature_extraction.text import CountVectorizer             
max_features = 3000                                                     # We will choose most frequent 3000 words

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() # Sparce Matrix

print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))

# %%
y = data.label                                                          # Hate or normal tweet classes
x = pd.DataFrame(sparce_matrix)
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

x_train = pd.DataFrame(x_train).reset_index().drop("index",axis=1)
x_test = pd.DataFrame(x_test).reset_index().drop("index",axis=1)
y__train = pd.DataFrame(y_train).reset_index().drop("index",axis=1)
y_test = pd.DataFrame(y_test).reset_index().drop("index",axis=1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV ,train_test_split
RFC = RandomForestClassifier()
parameters = { "max_depth" : [100,1000,2000], "min_samples_split" : [20,50,100]}
#creating our grid to find best parameters
tree_grid_search = GridSearchCV(RFC,param_grid=parameters,scoring="accuracy",cv = 3)  
tree_grid_search.fit(x_train,y_train) # adding data to grid search
print(" best parameters :", tree_grid_search.best_params_,"\n best score : " ,tree_grid_search.best_score_)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
RFC = RandomForestClassifier(max_depth=1000,min_samples_split=100)         # Use Best Parameters to train model
RFC.fit(x_train,y_train)
RFC_pred = RFC.predict(x_train)                                            # Predictions of train set
RFC_pred_test = RFC.predict(x_test)                                        # Predictions of test set
RFC_test_report = classification_report(y_test, RFC_pred_test)             # Test Report
RFC_test_confusion = confusion_matrix(y_test, RFC_pred_test)               # COnfusion matrix of predictions of test set

RFC_train_report = classification_report(y_train,RFC_pred)                 # Train Report
RFC_train_confusion = confusion_matrix(y_train,RFC_pred)                   # COnfusion matrix of predictions of train set
print("RFC Confusion MAtrix :\n  ",RFC_test_confusion)
print("Test Reports For RFC: \n",RFC_test_report)
from sklearn.metrics import plot_precision_recall_curve,plot_roc_curve
plot_precision_recall_curve(RFC,x_test,y_test)
import pickle
# save the model to disk
RVC_filename = 'finalized_RFC_model.sav'
pickle.dump(RFC, open(RVC_filename, 'wb'))
# load the model from disk
import pickle

RVC_filename = 'finalized_RFC_model.sav'
from sklearn.metrics import classification_report, confusion_matrix
loaded_RFC_model = pickle.load(open(RVC_filename, 'rb'))
RVC = loaded_RFC_model
train_svm_x, test_svm_x, train_svm_y, test_svm_y = train_test_split(x,y, test_size = 0.92, random_state = 42) # USE SMALL DATA FOR SVC!
from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.svm import SVC
import numpy as np
svc = SVC()
#creating paramater dictionary
parameters = {"kernel" : ["linear", "rbf", "poly"] , "C" : [0.1, 0.5, 1, 5], "tol" : [0.001, 0.1 ]}

grid_search = GridSearchCV(svc,param_grid=parameters,scoring="recall",cv = 5) # creating our grid to find best parameters
grid_search.fit(train_svm_x,train_svm_y) # adding data to grid search
print(" best parameters :", grid_search.best_params_,"\n best score : " ,grid_search.best_score_)
from sklearn.svm import SVC
svc = SVC(C=1, kernel="linear",tol=0.1)                                  # Use Best Parameters to train model
svc.fit(x_train,y_train)

svc_y_pred = svc.predict(x_train)                                        # Predictions of train set
svc_y_pred_test = svc.predict(x_test)                                    # Predictions of test set
svc_test_report = classification_report(y_test, svc_y_pred_test)         # Test Report
svc_test_confusion = confusion_matrix(y_test, svc_y_pred_test)           # COnfusion matrix of predictions of test set

svc_train_report = classification_report(y_train,svc_y_pred)             # Train Report
svc_train_confusion = confusion_matrix(y_train,svc_y_pred)               # COnfusion matrix of predictions of train set
print("SVC Confusion MAtrix \n: ",svc_test_confusion)
print("Test Reports For SVC: \n",svc_test_report)
plot_precision_recall_curve(svc,x_test,y_test)
import pickle
# save the model to disk
SVC_filename = 'finalized_SVC_model.sav'
pickle.dump(svc, open(SVC_filename, 'wb'))
# load the model from disk
SVC_filename = 'finalized_SVC_model.sav'
loaded_SVC_model = pickle.load(open(SVC_filename, 'rb'))
svc = loaded_SVC_model

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 1500, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 750, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 15)
classifier.fit(x_train,y_train)

ANN_pred = classifier.predict(x_train)                                  # Predictions of train set
ANN_pred_test = classifier.predict(x_test)                              # Predictions of test set
classifier.score(x_test,y_test)

from sklearn.metrics import classification_report, confusion_matrix
ANN_test_report = classification_report(y_test, ANN_pred_test)          # Test Report
ANN_test_confusion = confusion_matrix(y_test, ANN_pred_test)            # COnfusion matrix of predictions of test set

ANN_train_report = classification_report(y_train,ANN_pred)              # Train Report
ANN_train_confusion = confusion_matrix(y_train,ANN_pred)                # COnfusion matrix of predictions of train set

print("ANN Confusion MAtrix: \n ",ANN_test_confusion)
print("Test Report For ANN \n" ,ANN_test_report)
ensemble_test_predictions = ((pd.DataFrame(RFC_pred_test)+pd.DataFrame(svc_y_pred_test)+pd.DataFrame(ANN_pred_test))/3).round(0)  # taking average of all models' answers
ensemble_test_predictions.head()
ensemble_confsuion= confusion_matrix(y_test, ensemble_test_predictions)

print("Ensemble Model Confusion MAtrix: \n ",ensemble_confsuion)
ensemble_test_report = classification_report(y_test, ensemble_test_predictions)
print("Test Report For Ensemble Learning \n" ,ensemble_test_report)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

RandomFC = accuracy_score(y_test,RFC_pred_test)*100
SVCC = accuracy_score(y_test,svc_y_pred_test)*100
ANN = accuracy_score(y_test,ANN_pred_test)*100
Ensemble = accuracy_score(y_test,ensemble_test_predictions)*100
import matplotlib.pyplot as plt 
import seaborn as sns
fig = sns.barplot(["RFC","SVC","ANN","ENSEMBLE"],[RandomFC,SVCC,ANN,Ensemble])
fig.set_title('Accurcy Scores of Models')
fig.set(xlabel="Models",ylabel="Acuuracy (%)")
fig.set(ylim=(80,100))
RandomFC = precision_score(y_test,RFC_pred_test)*100
SVCC = precision_score(y_test,svc_y_pred_test)*100
ANN = precision_score(y_test,ANN_pred_test)*100
Ensemble = precision_score(y_test,ensemble_test_predictions)*100
import matplotlib.pyplot as plt 
import seaborn as sns
fig = sns.barplot(["RFC","SVC","ANN","ENSEMBLE"],[RandomFC,SVCC,ANN,Ensemble])
fig.set_title('Precision Scores of Models')
fig.set(xlabel="Models",ylabel="Precision (%)")
fig.set(ylim=(80,100))
RandomFC = recall_score(y_test,RFC_pred_test)*100
SVCC = recall_score(y_test,svc_y_pred_test)*100
ANN = recall_score(y_test,ANN_pred_test)*100
Ensemble = recall_score(y_test,ensemble_test_predictions)*100
import matplotlib.pyplot as plt 
import seaborn as sns
fig = sns.barplot(["RFC","SVC","ANN","ENSEMBLE"],[RandomFC,SVCC,ANN,Ensemble])
fig.set_title('Recall Scores of Models')
fig.set(xlabel="Models",ylabel="Recall (%)")
fig.set(ylim=(60,100))
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
RandomFC = f1_score(y_test,RFC_pred_test)*100
SVCC = f1_score(y_test,svc_y_pred_test)*100
ANN = f1_score(y_test,ANN_pred_test)*100
Ensemble = f1_score(y_test,ensemble_test_predictions)*100
import matplotlib.pyplot as plt 
import seaborn as sns
fig = sns.barplot(["RFC","SVC","ANN","ENSEMBLE"],[RandomFC,SVCC,ANN,Ensemble])
fig.set_title('F1 Scores of Models')
fig.set(xlabel="Models",ylabel="F1 (%)")
fig.set(ylim=(70,90))
