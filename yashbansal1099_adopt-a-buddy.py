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

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder
def generate_model_report(y_actual, y_predicted):

    print("Accuracy = " , (accuracy_score(y_actual, y_predicted))*100)

    print("Precision = " ,precision_score(y_actual, y_predicted, average = 'micro'))

    print("Recall = " ,recall_score(y_actual, y_predicted, average = 'micro'))

    print("F1 Score = " ,f1_score(y_actual, y_predicted, average = 'micro'))

    print(confusion_matrix(y_actual,y_predicted))

    pass
df = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/train.csv")

df = df.fillna(df.mean())

x = df[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]

y1 = df['breed_category']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

x['color_type'] = le.fit_transform(x['color_type'])

x.head()

plt.xlabel('breed category')

plt.ylabel('pet category')

plt.scatter(df['breed_category'], df['pet_category'])
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(x[['color_type']]).toarray())

x = x.join(enc_df)

x = x.drop(columns = ['color_type'], axis = 1)

x.head()
# dft = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/test.csv")

# dft = dft.fillna(dft.mean())

# xt = dft[['condition', 'color_type', 'length(m)', 'height(cm)','X1', 'X2']]

# xt['color_type'] = le.transform(xt['color_type'])

# yt1 = []

# yt2 = []

# xt.head()

X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.25)
classifier = RandomForestClassifier(n_estimators = 10, 

                                    criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

generate_model_report(y_test,y_predict)
model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

generate_model_report(y_test,y_predict)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

generate_model_report(y_test,y_predict)
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

generate_model_report(y_test,y_predict)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_predict = gnb.predict(X_test)

generate_model_report(y_test,y_predict)
clf=BernoulliNB()

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

generate_model_report(y_test,y_predict)
# max accuracy is achieved at using Decision Tree Classifier
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

generate_model_report(y_test,y_predict)
X_train['breed_category'] = y_train

X_test['breed_category'] = y_predict



classifier = RandomForestClassifier(n_estimators = 10, 

                                    criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

generate_model_report(y_test,y_predict)
model = KNeighborsClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

generate_model_report(y_test,y_predict)
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

generate_model_report(y_test,y_predict)
classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

generate_model_report(y_test,y_predict)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_predict = gnb.predict(X_test)

generate_model_report(y_test,y_predict)
clf=BernoulliNB()

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)

generate_model_report(y_test,y_predict)