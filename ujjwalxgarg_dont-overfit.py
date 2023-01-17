# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

sns.set(color_codes=True)



import matplotlib.pyplot as plt



%matplotlib inline

%precision 2
data_df = pd.read_csv('../input/dont-overfit-ii/train.csv')



data_df.sample(3)
data_df['target'].value_counts().plot(kind='bar')

plt.xlabel('label')

plt.title('Distribution of classes among the training data')
plt.figure(figsize=(26, 24))

for i, col in enumerate(data_df.columns[2:26]):

    plt.subplot(6, 4, i + 1)

    sns.distplot(data_df[col])
data_df[data_df.columns[2:]].mean().plot('hist')

plt.title('Distribution of Mean of each column');
plt.figure(figsize=(8, 5))

for i, col in enumerate(data_df.columns[2:6]):

    plt.subplot(2, 2, i + 1)

    sns.violinplot(data=data_df, y=col, x='target')
X = data_df.drop(columns=["target", "id"])

y = data_df["target"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(max_depth=2, bootstrap=True)

rfc.fit(X_train, y_train)

# make predictions using the trained model

y_pred = rfc.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_pred, y_test))



from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred, y_test))



from sklearn.metrics import accuracy_score

print('accuracy is ',accuracy_score(y_pred, y_test))
from sklearn.ensemble import BaggingClassifier



bag_model = BaggingClassifier()

bag_model.fit(X_train, y_train)

y_pred=bag_model.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('accuracy is ',accuracy_score(y_pred, y_test))
from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier()

abc.fit(X_train, y_train)

y_pred = abc.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('accuracy is ',accuracy_score(y_pred, y_test))
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('accuracy is ',accuracy_score(y_pred, y_test))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis()

lda.fit(X_train, y_train)

y_pred = lda.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('accuracy is ',accuracy_score(y_pred, y_test))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



qda = QuadraticDiscriminantAnalysis()

qda.fit(X_train, y_train)

y_pred = qda.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('accuracy is ',accuracy_score(y_pred, y_test))
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)



rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfc_model, random_state=1).fit(test_X, test_y)

eli5.show_weights(perm, feature_names=test_X.columns.tolist(), top=300)