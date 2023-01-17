# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('../input/spam.csv', encoding = "ISO-8859-1")

df.head()
df.describe()
df.isnull().sum()
# Majority of the values in Unnamed: 2, Unnamed: 3 & Unnamed: 4 are null values
# Dropping the three columns and renaming the columns v1 & v2

df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
df.rename(columns={"v1":"label", "v2":"sms"}, inplace=True)

df.head()
df.label.value_counts()
# convert label to a numerical variable
df.label = pd.Categorical(df.label).codes

df.head()
# Train the classifier if it is spam or ham based on the text
# TFIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
vectorizer.fit(df)
y = df.label

X = vectorizer.fit_transform(df.sms)
X
## Spliting the SMS to separate the text into individual words
splt_txt1=df.sms[0].split()
print(splt_txt1)
## Finding the most frequent word appearing in the SMS
max(splt_txt1)
## Count the number of words in the first SMS
len(splt_txt1)
X[0]
print(X)
## Spliting the SMS to separate the text into individual words
splt_txt2 = df.sms[1].split()
print(splt_txt2)
print(max(splt_txt2))
## The most freaquent word across all the SMSes
max(vectorizer.get_feature_names())
print (y.shape)
print (X.shape)
##Split the test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
## Let us try different models, and check how thye accuracy is for each of the models

clf = naive_bayes.MultinomialNB()
model = clf.fit(X_train, y_train)
prediction = dict()
prediction['Naive_Bayes'] = model.predict(X_test)
accuracy_score(y_test, prediction["Naive_Bayes"])
models = dict()
models['Naive_Bayes'] = naive_bayes.MultinomialNB()
models['SVC'] = SVC()
models['KNC'] = KNeighborsClassifier()
models['RFC'] = RandomForestClassifier()
models['Adaboost'] = AdaBoostClassifier()
models['Bagging'] = BaggingClassifier()
models['ETC'] = ExtraTreesClassifier()
models['GBC'] = GradientBoostingClassifier()
results = dict()
accuracies = dict()

for key, value in models.items():
    value.fit(X_train, y_train)
    output = value.predict(X_test)
    accuracies[key] = accuracy_score(y_test, output)
accuracies
# With the default values, Gradient Boost sems to be performing the best
# Let's fine tune and make predictions

paramGrid = dict(n_estimators=np.array([50, 100, 200,400,600,800,900]))

model = GradientBoostingClassifier(random_state=10)

grid = GridSearchCV(estimator=model, param_grid=paramGrid)

grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))