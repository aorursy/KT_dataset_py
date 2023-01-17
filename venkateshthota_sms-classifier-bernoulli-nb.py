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

docs = pd.read_table('../input/sms-data-labelled-spam-and-non-spam/SMSSpamCollection', header=None, names=['Class', 'sms'])

docs.head()

docs.Class.value_counts()
ham_spam=docs.Class.value_counts()

ham_spam
print("Spam % is ",(ham_spam[1]/float(ham_spam[0]+ham_spam[1]))*100)
docs['label'] = docs.Class.map({'ham':0, 'spam':1})
docs.head()
X=docs.sms

y=docs.label
X = docs.sms

y = docs.label

print(X.shape)

print(y.shape)
from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.head()
from sklearn.feature_extraction.text import CountVectorizer



vect = CountVectorizer(stop_words='english')
vect.fit(X_train)
vect.vocabulary_
vect.get_feature_names()
X_train_transformed = vect.transform(X_train)

X_test_tranformed =vect.transform(X_test)
from sklearn.naive_bayes import BernoulliNB



bnb = BernoulliNB()

bnb.fit(X_train_transformed,y_train)

y_pred_class = bnb.predict(X_test_tranformed)

y_pred_proba =bnb.predict_proba(X_test_tranformed)

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)

bnb
metrics.confusion_matrix(y_test, y_pred_class)
confusion = metrics.confusion_matrix(y_test, y_pred_class)

print(confusion)

#[row, column]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]

TP = confusion[1, 1]
sensitivity = TP / float(FN + TP)

print("sensitivity",sensitivity)
specificity = TN / float(TN + FP)



print("specificity",specificity)
precision = TP / float(TP + FP)



print("precision",precision)

print(metrics.precision_score(y_test, y_pred_class))
print("precision",precision)

print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))

print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))

print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))
y_pred_proba
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])

roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
print(true_positive_rate)
print(false_positive_rate)
print(thresholds)
%matplotlib inline  

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC')

plt.plot(false_positive_rate, true_positive_rate)