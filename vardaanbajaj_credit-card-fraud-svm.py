# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn import preprocessing, svm
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
#Checking for equal sampling
print(df['Class'].value_counts())
#There are 492 fraudulent transactions as compared to 284315 non-fraudulent ones. Clearly we need to resample our data else by simply predicting every transaction as 1 (non-fraudulent), the following happens
print("Accuracy by predicting all transactions as non-fraudulent: " + str((284315 / (284315 + 492)) * 100) + "%")
#Analysing data for fraudulent transactions
df_fraud = df[df['Class'] == 1]
plt.figure(figsize=(15, 12))
plt.scatter(df_fraud['Time'], df_fraud['Amount'])
plt.title("Fraudulent transactions")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.show()
#At all times we have some number of fraudulent transactions making time an irrelevant factor while predicting them

#Additionally, most of the fraudulent transactions are very small amount values as seen from the graph
df_huge_fraud_amounts = df_fraud[df_fraud['Amount'] > 1000]
print("Number of fraudulent transactions over the amount of 1000 are: " + str((df_huge_fraud_amounts.shape[0])))
df_huge_fraud_amounts
# The given dataset says that the features have been generated using PCA, hence they should be independent of each other. Let us check if this is the case by calculating the correlation between the features
import seaborn as sns
plt.figure(figsize = (15, 12))
df_correlation = df.corr()
sns.heatmap(df_correlation)

plt.title("Heatmap representing correlations")
plt.show()

# The diagonal across the heatmap represents the highest correlation (close to 1.0) and the correlation between other pairs of features has values between -0.2 to 0.2, which corresponds to very less correlation. This represents the features mentioned are indeed independent of each other and hence no feature can be eliminated based on their dependency on each other
#Since the number of total fraudulent transactions is too small as compared to non-fraudulent transactions, we need to resample the dataset.
# By applying oversampling, we repeat the fraudulent transactions until they are close in number to non-fraudulent transactions
# By applying undersampling, we eliminate a number of non-fraudulent transactions so that the final number of non-fraudulent transactions is roughly the same as fraudulent transactions in the dataset
# By applying oversampling, the training dataset will become huge, so we use undersampling
df_train = df[:200000]
df_train_fraud = df_train[df_train['Class'] == 1]
df_train_not_fraud = df_train[df_train['Class'] == 0]

print(df_train_fraud.shape[0])

# Since there are 385 fraud transactions, we'll bring down non-fraudulent transactions to around this number to have equal number of training examples of each kind

df_sample = df_train_not_fraud.sample(400)
df_train_final = df_train_fraud.append(df_sample)
df_train_final = df_train_final.sample(frac = 1).reset_index(drop = True)
df_train_final.head()
X_train = df_train_final.drop(['Time', 'Class'],axis=1)
y_train = df_train_final['Class']

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
df_test = df[200000:]

X_test = df_test.drop(['Time', 'Class'],axis=1)
y_test = df_test['Class']
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
print(df_test['Class'].value_counts())
# Applying SVM
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
#Plotting confusion matrix
import itertools
classes = np.array(['0','1'])

def plot_confusion_matrix(cm, classes,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cm,classes)
#Accuracy alone is not a good metric to determine the final result since the number of non-fraudulent transactions heavily outnumber the fraudulent transactions
#For a banking system, it's fine if some non-fraudulent transactions are detected as fraudulent, they'll look into it, but if fraudulent transactions are labelled as non-fraudulent, then that can cause huge losses. Hence, our objective is to reduce the total number of false negatives as much as possible and on the same hand, try that the number of false positives are on the lower side as well.

print('Total fraudulent transactions detected: ' + str(cm[1][1]) + ' / ' + str(cm[1][1]+cm[1][0]))
print('Total non-fraudulent transactions detected: ' + str(cm[0][0]) + ' / ' + str(cm[0][1]+cm[0][0]))

print('Probability to detect a fraudulent transaction: ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print('Probability to detect a non-fraudulent transaction: ' + str(cm[0][0]/(cm[0][1]+cm[0][0])))

print("Accuracy of the SVM model : "+str(100*(cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))) + "%")
