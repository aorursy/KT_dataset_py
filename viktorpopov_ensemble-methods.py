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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn')

%matplotlib inline

# plt.('figure', figsize=(10, 7))

plt.rcParams["figure.figsize"] = (10,7)





from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, f1_score

from scipy.stats import shapiro, ttest_ind, mannwhitneyu

from scipy.stats import mode

from sklearn.model_selection import cross_val_score, LeaveOneOut

import random



from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier, AdaBoostClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





import warnings

from pprint import pprint

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/heart-disease-data/Heart_Disease_Data.csv', sep=',')

df.pred_attribute = df.pred_attribute.replace([1, 2, 3, 4], 1)

df = df.replace('?', 0, method='ffill') #replaces empty values with zero.

df.head()
#divide into training and test

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1],test_size = 0.5, random_state=1121, shuffle=True)
log = LogisticRegression()

log.fit(x_train, y_train)



acc_log_train = accuracy_score(y_train, log.predict(x_train))

acc_log_test = accuracy_score(y_test, log.predict(x_test))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)



acc_knn_train = accuracy_score(y_train, knn.predict(x_train))

acc_knn_test = accuracy_score(y_test, knn.predict(x_test))
rfc = RandomForestClassifier(max_depth=2)

rfc.fit(x_train, y_train)



acc_rfc_train = accuracy_score(y_train, rfc.predict(x_train))

acc_rfc_test = accuracy_score(y_test, rfc.predict(x_test))
gnb = GaussianNB()

gnb.fit(x_train, y_train)



acc_gnb_train = accuracy_score(y_train, gnb.predict(x_train))

acc_gnb_test = accuracy_score(y_test, gnb.predict(x_test))
ens = VotingClassifier([('LogR', LogisticRegression()), 

                        ('NaiveBayes',GaussianNB()), 

                        ('kNN', KNeighborsClassifier(n_neighbors=11)),

                        ('RandomForest', RandomForestClassifier(max_depth=2))], 

                        voting='soft', weights=[1, 1, 1, 1])

ens.fit(x_train, y_train)



acc_ens_train = accuracy_score(y_train, ens.predict(x_train))

acc_ens_test = accuracy_score(y_test, ens.predict(x_test))
print('Accuracy in the training set:')

print('log train: %.3f'%acc_log_train)

print('knn train: %.3f'%acc_knn_train)

print('rfc train: %.3f'%acc_rfc_train)

print('gnb train: %.3f'%acc_gnb_train)

print('ens train: %.3f'%acc_ens_train)

print('--------')

print('Accuracy in the test set:')

print('log test: %.3f'%acc_log_test)

print('knn test: %.3f'%acc_knn_test)

print('rfc test: %.3f'%acc_rfc_test)

print('gnb test: %.3f'%acc_gnb_test)

print('ens test: %.3f'%acc_ens_test)

print('--------')
log_report = classification_report(y_test, log.predict(x_test))

knn_report = classification_report(y_test, knn.predict(x_test))

rfc_report = classification_report(y_test, rfc.predict(x_test))

gnb_report = classification_report(y_test, gnb.predict(x_test))

ens_report = classification_report(y_test, ens.predict(x_test))

print('log', log_report)

print('--------')

print('kne', knn_report)

print('--------')

print('rfc', rfc_report)

print('--------')

print('gnb', gnb_report)

print('--------')

print('ens', ens_report)
bar_width = 0.3 # Chart column width

position = np.arange(5) # Specify the number of positions on the chart: 4 classifiers – 4 positions



# Write to the lists of accuracy estimates that were obtained above

total_accuracy_train = [acc_log_train, acc_knn_train, 

                        acc_rfc_train, acc_gnb_train,

                        acc_ens_train] # Accuracy on training

total_accuracy_test = [acc_log_test, acc_knn_test, 

                       acc_rfc_test, acc_gnb_test,

                       acc_ens_test] # The accuracy on the test



# Building graphics



plt.bar(position, total_accuracy_train, width = bar_width, 

        label='Training sample', align='center')

plt.bar(position+bar_width, total_accuracy_test, width=bar_width, 

        label='Test sample')

# Подписываем

plt.xticks(position+bar_width/2, ['LogR', 'KNN', 'RandomForest', 'GaussianNB', 'Ensemble'])

plt.ylabel('Accuracy')





plt.ylim((0, 1.1)) # Edit the vertical axis

plt.legend(loc=1) # Specify the position of the legend

plt.title('Classification accuracy on training and test set'); # Print the title