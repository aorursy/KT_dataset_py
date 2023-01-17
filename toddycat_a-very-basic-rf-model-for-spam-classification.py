import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn.metrics import f1_score
df = pd.read_csv("../input/spam.csv",encoding='latin-1')
df.head()
del(df["Unnamed: 2"])
del(df["Unnamed: 3"])
del(df["Unnamed: 4"])
df = df.rename(columns = {"v1" : "Label", "v2" : "Message" })
df['Flag'] = df.Label.map({'ham':0, 'spam':1})
df = df.drop(['Label'],axis =1)
df.head()
Count = pd.value_counts(df['Flag'], sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Spam - Ham histogram")
plt.xlabel("Flag")
plt.ylabel("Frequency")

data_X = df.loc[:, df.columns != 'Flag']
data_Y = df.loc[:,df.columns == 'Flag']

numberofrecords_spam = len(df[df.Flag == 1])
spam_indices = np.array(df[df.Flag == 1].index)

notspam_indices = df[df.Flag == 0].index

random_notspam_indices = np.random.choice(notspam_indices,numberofrecords_spam,replace = False)
random_notspam_indices = np.array(random_notspam_indices)

under_sample_indices = np.concatenate([spam_indices,random_notspam_indices])

under_sample_data = df.iloc[under_sample_indices,:]

X_undersample = under_sample_data.loc[:,df.columns != 'Flag']
Y_undersample = under_sample_data.loc[:,df.columns == 'Flag']
print("Number of spam messages: " , len(under_sample_data[under_sample_data.Flag == 1]))
print("Number of ham messages : " , len(under_sample_data[under_sample_data.Flag == 0]))
print("Total messages : ", len(under_sample_data))
X_train,X_test,Y_train,Y_test = train_test_split(data_X['Message'],data_Y,test_size = 0.3, random_state = 0)

print("Number of datapoints in training : " , len(X_train))
print("Number of datapoints in testing : ", len(X_test))

X_train_undersample,X_test_undersample,Y_train_undersample,Y_test_undersample = train_test_split(X_undersample['Message'],Y_undersample,test_size = 0.3, random_state = 0)

print("Number of datapoints in Undersampled training data : " , len(X_train_undersample))
print("Number of datapoints in Undersampled testing data : ", len(X_test_undersample))
vect = CountVectorizer()
vect.fit(X_train)

#Print first five features
print(vect.get_feature_names()[0:5])

X_train_csr = vect.transform(X_train)
X_test_csr = vect.transform(X_test)
X_train_undersample_csr = vect.transform(X_train_undersample)
X_test_undersample_csr = vect.transform(X_test_undersample)
RANDOM_STATE = 123


ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

#Map a classifier name to a list of pairs
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

#Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 175

import warnings
warnings.filterwarnings("ignore")
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_train_undersample_csr, Y_train_undersample.values.ravel())

        #Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

#Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()
clf = RandomForestClassifier(n_estimators= 155,max_features= 'sqrt',oob_score=True)
clf.fit(X_test_undersample_csr,Y_test_undersample)
print("OOB Score for undersampled test data = ",clf.oob_score_)

#Lets check the F1 score for the model created using undersampled training data on undersampled test data
rf = RandomForestClassifier(n_estimators= 155, max_features= 'sqrt',oob_score= True)
rf.fit(X_train_undersample_csr,Y_train_undersample)
Y_pred = rf.predict(X_test_undersample_csr)
print('F1 score for the undersampled test data = ', f1_score(Y_test_undersample,Y_pred))

clf = RandomForestClassifier(n_estimators= 155,max_features= 'sqrt',oob_score=True)
clf.fit(X_test_csr,Y_test)
print("OOB Score for test data = ",clf.oob_score_)

#Lets check the F1 score for the model created using undersampled training data on undersampled test data
rf = RandomForestClassifier(n_estimators= 155, max_features= 'sqrt',oob_score= True)
rf.fit(X_train_undersample_csr,Y_train_undersample)
Y_pred = rf.predict(X_test_csr)
print('F1 score for the undersampled test data = ', f1_score(Y_test,Y_pred))