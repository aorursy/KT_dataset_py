# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from scipy.spatial.distance import mahalanobis

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
df.describe()
print(df.shape)
data= df.sample(frac = 0.2,random_state=1)
print(data.shape)
df.isnull().values.any()
num_classes = pd.value_counts(df['Class'], sort = True)
num_classes.plot(kind = 'bar')
plt.title("Transaction Class Distribution")
plt.xticks(range(2), ["Normal", "Fraud"])
plt.xlabel("Class")
plt.ylabel("Frequency");
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print(fraud.shape, normal.shape)
fraud.describe()
normal.describe()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction v/s Amount by Class type')
bins = 10
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in secs)')
plt.ylabel('Amount')
plt.xlim((0, 20000))

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 10
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')

plt.show()
fraud = data[data['Class']==1]
normal = data[data['Class']==0]
outlier_fraction = len(fraud)/float(len(normal))

anomaly_fraction = len(fraud)/float(len(normal))
print(anomaly_fraction)

print("Fraud Cases: " + str(len(fraud)))
print("Normal Cases: " + str(len(normal)))
data.hist(figsize=(15,15), bins = 64)
plt.show()
data.drop(['Time', 'V1', 'V24'], axis=1, inplace=True)
columns = data.columns.tolist()

target=columns[-1]
columns = columns[:-1]

X_train = data.iloc[:45000, :-1]
y_train = data.iloc[:45000, -1]

X_test = data.iloc[45000:, :-1]
y_test = data.iloc[45000:, -1]

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"

# state = np.random.RandomState(0)
X = data[columns]
Y = data[target]
# X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

print(X.shape)
print(Y.shape)
# model = LocalOutlierFactor(algorithm='auto', metric='mahalanobis', metric_params={'V': np.cov(X)})
model = LocalOutlierFactor(contamination=anomaly_fraction)
y_train_pred = model.fit_predict(X_train)
# print(y_train[:5], y_train_pred[:5])
y_train_pred[y_train_pred == 1] = 0
y_train_pred[y_train_pred == -1] = 1

y_test_pred = model.fit_predict(X_test)
# print(y_train[:5], y_train_pred[:5])
y_test_pred[y_test_pred == 1] = 0
y_test_pred[y_test_pred == -1] = 1
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
cm_train = confusion_matrix(y_train, y_train_pred)
plot_confusion_matrix(cm_train,["Normal", "Fraud"])
cm_test = confusion_matrix(y_test_pred, y_test)
plot_confusion_matrix(cm_test,["Normal", "Fraud"])
print('Total fraudulent transactions detected in training set: ' + str(cm_train[1][1]) + ' / ' + str(cm_train[1][1]+cm_train[1][0]))
print('Total non-fraudulent transactions detected in training set: ' + str(cm_train[0][0]) + ' / ' + str(cm_train[0][1]+cm_train[0][0]))

print('Probability to detect a fraudulent transaction in the training set: ' + str(cm_train[1][1]/(cm_train[1][1]+cm_train[1][0])))
print('Probability to detect a non-fraudulent transaction in the training set: ' + str(cm_train[0][0]/(cm_train[0][1]+cm_train[0][0])))

print("Accuracy of unsupervised anomaly detection model on the training set: "+str(100*(cm_train[0][0]+cm_train[1][1]) / (sum(cm_train[0]) + sum(cm_train[1]))) + "%")
print('Total fraudulent transactions detected in test set: ' + str(cm_test[1][1]) + ' / ' + str(cm_test[1][1]+cm_test[1][0]))
print('Total non-fraudulent transactions detected in test set: ' + str(cm_test[0][0]) + ' / ' + str(cm_test[0][1]+cm_test[0][0]))

print('Probability to detect a fraudulent transaction in the test set: ' + str(cm_test[1][1]/(cm_test[1][1]+cm_test[1][0])))
print('Probability to detect a non-fraudulent transaction in the test set: ' + str(cm_test[0][0]/(cm_test[0][1]+cm_test[0][0])))

print("Accuracy of unsupervised anomaly detection model on the test set: "+str(100*(cm_test[0][0]+cm_test[1][1]) / (sum(cm_test[0]) + sum(cm_test[1]))) + "%")
