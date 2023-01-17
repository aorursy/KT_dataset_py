import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn import svm
import random

random_seed = 72
random.seed(random_seed)
np.random.seed(random_seed)

df = pd.read_csv("../input/aps_failure_training_set.csv")
df_test = pd.read_csv("../input/aps_failure_test_set.csv")
df.head()
df = df.rename(columns = {'class' : 'Flag'})
df['Flag'] = df.Flag.map({'neg':0, 'pos':1})
df = df.replace(['na'],[np.NaN])
df.isnull().any()
Count = pd.value_counts(df['Flag'], sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Class count")
plt.xlabel("Flag")
plt.ylabel("Frequency")
len(df.columns)
df_X = df.loc[:,df.columns != 'Flag']
df_Y = df.loc[:,df.columns == 'Flag']

df_X = df_X.apply(pd.to_numeric)

df_X= df_X.fillna(df_X.mean()).dropna(axis =1 , how ='all')

scaler = StandardScaler()

scaler.fit(df_X)

df_X = scaler.transform(df_X)

pca = PCA(0.95)

pca.fit(df_X)

pca.n_components_
df_X = pca.transform(df_X)

df_X= pd.DataFrame(df_X)
df_test = df_test.rename(columns = {'class' : 'Flag'})
df_test = df_test.replace(['na'],[np.NaN])

Count = pd.value_counts(df_test['Flag'], sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Class count")
plt.xlabel("Flag")
plt.ylabel("Frequency")

df_test['Flag'] = df_test.Flag.map({'neg':0, 'pos':1})

df_test_X = df_test.loc[:,df_test.columns != 'Flag']
df_test_Y = df_test.loc[:,df_test.columns == 'Flag']

df_test_X = df_test_X.apply(pd.to_numeric)

df_test_X= df_test_X.fillna(df_test_X.mean()).dropna(axis =1 , how ='all')

scaler = StandardScaler()

scaler.fit(df_test_X)

df_test_X = scaler.transform(df_test_X)

pca = PCA(82)

pca.fit(df_test_X)

pca.n_components_

df_test_X = pca.transform(df_test_X)

df_test_X= pd.DataFrame(df_test_X)
X_train,X_validation,Y_train,Y_validation = train_test_split(df_X,df_Y,test_size = 0.2,random_state = 0)
DF = pd.concat([X_train,Y_train],axis = 1)

print("Percentage Neg in training: " , len(Y_train[Y_train.Flag == 0])/len(Y_train))
print("Percentage Pos in training: ", len(Y_train[Y_train.Flag == 1])/len(Y_train))
print("Total number of datapoints in training: ", len(Y_train))


print("Percentage Neg in Validation: " , len(Y_validation[Y_validation.Flag == 0])/len(Y_validation))
print("Percentage Pos in Validation: ", len(Y_validation[Y_validation.Flag == 1])/len(Y_validation))
print("Total number of datapoints in Validation: ", len(Y_validation))

numberofrecords_pos = len(DF[DF.Flag == 1])
pos_indices = np.array(DF[DF.Flag == 1].index)

#Picking the indices of the normal class
neg_indices = DF[DF.Flag == 0].index

#out of indices selected, randomly select "x" number of records
random_neg_indices = np.random.choice(neg_indices, numberofrecords_pos, replace = False)
random_neg_indices =np.array(random_neg_indices)

#Appending the two indices
under_sample_indices = np.concatenate([pos_indices,random_neg_indices])

#Undersample dataset
under_sample_data = DF.loc[under_sample_indices,:]

X_undersample = under_sample_data.loc[:,under_sample_data.columns != 'Flag']
Y_undersample = under_sample_data.loc[:,under_sample_data.columns == 'Flag']

print("Percentage Neg: " , len(under_sample_data[under_sample_data.Flag == 0])/len(under_sample_data))
print("Percentage Pos : ", len(under_sample_data[under_sample_data.Flag == 1])/len(under_sample_data))
print("Total number of datapoints : ", len(under_sample_data))
c_parameter_range = [0.0001,0.001,0.01,0.1,1,10,100]
penalty = ['l1','l2']
for penal in penalty:
    for c_param in c_parameter_range:
        
        print('------------------------')
        print("C Parameter :", c_param)
        print("Penalty: ", penal)
        print('------------------------')
        print('')
        lr = LogisticRegression(C = c_param, penalty = penal)
        lr.fit(X_undersample,Y_undersample.values.ravel())
        y_pred = lr.predict(X_validation)
        Recall = recall_score(Y_validation,y_pred)
        print ('Recall score for c param', c_param,'and penalty',penal,'=',Recall)
        print('-------------------------')
        print('')
lr = LogisticRegression(C =0.001,penalty = 'l2')

lr.fit(X_undersample,Y_undersample.values.ravel())

y_pred = lr.predict(df_test_X)

recall_score(df_test_Y,y_pred)
confusion_matrix(df_test_Y,y_pred)
RANDOM_STATE = 123

import warnings
warnings.filterwarnings("ignore")
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

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 150

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_undersample, Y_undersample)

        # Record the OOB error for each `n_estimators=i` setting.
        y_pred = clf.predict(X_validation)
        recall = recall_score(Y_validation,y_pred)
        error = 1 - recall
        error_rate[label].append((i, error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("Recall error rate")
plt.legend(loc="upper right")
plt.show()


clf = RandomForestClassifier(n_estimators=25,max_features= 'log2',oob_score =True)

clf.fit(X_undersample,Y_undersample.values.ravel())

clf.oob_score_

y_pred = clf.predict(df_test_X)
recall_score(df_test_Y,y_pred)
confusion_matrix(df_test_Y,y_pred)
c_parameter_range = [0.001,0.01,0.1,10,100]
kernel = ['linear','poly','rbf','sigmoid']
for kern in kernel:
    for c_param in c_parameter_range:
        print('------------------------')
        print("C Parameter :", c_param)
        print("Kernel: ", kern)
        print('------------------------')
        print('')
        clf = svm.SVC(C = c_param,kernel = kern,gamma = 0.01)
        clf.fit(X_undersample,Y_undersample)
        y_pred = clf.predict(X_validation)
        Recall = recall_score(Y_validation,y_pred)
        print ('Recall Score for c parameter', c_param, 'and kernel',kern,'=',Recall)
        print('-------------------------')
        print('')
clf = svm.SVC(C =0.01,gamma = 0.01, kernel = 'sigmoid')

clf.fit(X_undersample,Y_undersample)
y_pred = clf.predict(df_test_X)
recall_score(df_test_Y,y_pred)
confusion_matrix(df_test_Y,y_pred)
