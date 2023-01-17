# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import nltk
from nltk.stem.wordnet import WordNetLemmatizer 
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the data
data = pd.read_csv('../input/train.csv')
data.head()
print("There is {} messages.".format(len(data)))
classes = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
occurence = []
print("\n{:^15} | {:^15} | {:^5}".format("Class", "Occurrence", "%"))
print("*"*42)
for clas in classes:
    print("{:15} | {:>15} | {:^5.2f}".format(clas, 
                                             data[clas].value_counts()[1], 
                                             data[clas].value_counts()[1]*100/len(data)
                                            )
         )
    occurence.append(data[clas].value_counts()[1])
plt.figure(figsize=(10, 6))
plt.bar(classes, occurence)
plt.title("Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)
data['all'] = data[classes].sum(axis=1)
data['any'] = data['all'].apply(lambda x:1 if x>0 else 0)
data.head()
in_classes = data['all'].value_counts()
print("\n{:^10} | {:^10} | {:^6}".format("# Classes", "# Comment", "%"))
print("*"*33)
for idx in range(7):
    print("{:10} | {:>10} | {:>6.2f}".format(idx, 
                                             in_classes[idx], 
                                             in_classes[idx]*100/len(data)
                                            )
         )
print("*"*33)
print("{:^10} | {:>10} | {:>6}".format("", len(data), "100.00"))
df = pd.DataFrame(in_classes.values)
ax = df.plot.bar(stacked=True, figsize=(10, 6), legend=False)
ax.set_ylabel('# of Occurrences', fontsize=12)
ax.set_xlabel('# of classes', fontsize=12)
ax.set_title("# of messages per # of classes associated")
# toxic
data[data['toxic']==1].iloc[1,1]
# severe_toxic
data[data['severe_toxic']==1].iloc[2,1]
# obscene
data[data['obscene']==1].iloc[3,1]
# threat
data[data['threat']==1].iloc[4,1]
# insult
data[data['insult']==1].iloc[5,1]
# identity_hate
data[data['identity_hate']==1].iloc[6,1]
lens = data['comment_text'].str.len()
lens.head()
# Statistics:
print('Minimum : ', lens.min())
print('Maximum : ', lens.max())
print('Median : ', lens.median())
# horizontal boxplot
plt.figure(figsize=(15,4))
plt.boxplot(lens, 0, 'gD', 0, showmeans=True)
# The length of comment text is varying a lot. There is a lot of outlier.
# Split data using stratifying variable "all" to take into account the imbalanced data throw calsses
datatrain, datatest = train_test_split(data, test_size=0.2, stratify=data["all"], random_state=42)
# Here we create a list of noisy entities
useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation) + ["\'m"] + ["\'s"] + ["\'\'"] + ["``"] + ["n\'t"] + ["ca"]
lem = WordNetLemmatizer()
def clean_data(txt):
    txt = nltk.word_tokenize(txt.lower())
    txt = [word for word in txt if not word in useless_words]
    txt = [lem.lemmatize(w, "v") for w in txt]
    return ' '.join(word for word in txt)
# datatest['comment_text'] = datatest['comment_text'].apply(lambda x:clean_data(x))
# datatrain['comment_text'] = datatrain['comment_text'].apply(lambda x:clean_data(x))

datatest['comment_text'] = datatest['comment_text'].apply(lambda x:clean_data(x))
datatrain['comment_text'] = datatrain['comment_text'].apply(lambda x:clean_data(x))
def ROC_curve_plot(datatest, prediction, classes, figure_title):
    # Compute ROC curve and ROC area for each class
    nbr_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y = np.zeros(nbr_classes*len(datatest))
    y_hat = np.zeros(nbr_classes*len(datatest))

    for idx,clas in enumerate(classes):
        print('... Processing {}'.format(clas))
        print('Cofusion Matrix:\n', confusion_matrix(datatest[clas], prediction[:,idx]))
        fpr[clas], tpr[clas], _ = roc_curve(datatest[clas], prediction[:,idx])
        roc_auc[clas] = auc(fpr[clas], tpr[clas])

        y[idx*len(datatest):(idx+1)*len(datatest)] = datatest[clas].values
        y_hat[idx*len(datatest):(idx+1)*len(datatest)] = prediction[:,idx]
        
    # Compute average ROC curve and ROC area
    fpr["all"], tpr["all"], _ = roc_curve(y, y_hat)
    roc_auc["all"] = auc(fpr["all"], tpr["all"])
    
    plt.figure(figsize=(10,10))
    for i in ["all"] + classes:
        plt.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate' , fontsize=12)
    plt.title(figure_title,           fontsize=12)
    plt.legend(loc="lower right",     fontsize=12)
    plt.show()
NB_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer()),
                        ('clf', OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))),
                       ])

NB_pipeline.fit(datatrain['comment_text'], datatrain[classes])
prediction = NB_pipeline.predict(datatest['comment_text'])

ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Naive Bayes Classifier')
SVC_pipeline = Pipeline([
                         ('tfidf', TfidfVectorizer()),
                         ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
                        ])

SVC_pipeline.fit(datatrain['comment_text'], datatrain[classes])
prediction = SVC_pipeline.predict(datatest['comment_text'])

ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Linear SVC Classifier')
LogReg_pipeline = Pipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
                           ])

LogReg_pipeline.fit(datatrain['comment_text'], datatrain[classes])
prediction = LogReg_pipeline.predict(datatest['comment_text'])

ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Logistic Regression Classifier')
# from sklearn.ensemble import RandomForestClassifier
# RandomForest_pipeline = Pipeline([
#                             ('tfidf', TfidfVectorizer()),
#                             ('clf', OneVsRestClassifier(RandomForestClassifier(), n_jobs=1)),
#                            ])

# RandomForest_pipeline.fit(datatrain['comment_text'], datatrain[classes])
# prediction = RandomForest_pipeline.predict(datatest['comment_text'])

# ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Random Forest Classifier')
from xgboost import XGBClassifier
XGBoost_pipeline = Pipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('clf', OneVsRestClassifier(XGBClassifier(), n_jobs=1)),
                           ])

XGBoost_pipeline.fit(datatrain['comment_text'], datatrain[classes])
prediction = XGBoost_pipeline.predict(datatest['comment_text'])

ROC_curve_plot(datatest, prediction, classes, 'ROC curve : XGBoost Classifier')
### Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
DecisionTree_pipeline = Pipeline([
                            ('tfidf', TfidfVectorizer()),
                            ('clf', OneVsRestClassifier(DecisionTreeClassifier())),
                           ])

DecisionTree_pipeline.fit(datatrain['comment_text'], datatrain[classes])
prediction = DecisionTree_pipeline.predict(datatest['comment_text'])

ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Decision Tree Classifier')
### Multi-layer Perceptron
# from sklearn.neural_network import MLPClassifier
# MLPClassifier_pipeline = Pipeline([
#                             ('tfidf', TfidfVectorizer()),
#                             ('clf', OneVsRestClassifier(MLPClassifier())),
#                            ])

# MLPClassifier_pipeline.fit(datatrain['comment_text'], datatrain[classes])
# prediction = MLPClassifier_pipeline.predict(datatest['comment_text'])

# ROC_curve_plot(datatest, prediction, classes, 'ROC curve : Multi-layer Perceptron Classifier')
