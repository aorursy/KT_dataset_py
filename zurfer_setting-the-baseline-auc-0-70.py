import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline
data = pd.read_csv("../input/biddings.csv")

print(data.shape)

data.head()
count_classes = pd.value_counts(data['convert'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("bidding conversion histogram")

plt.xlabel("Conversion")

plt.ylabel("Count")
#advantage of being the creator of the dataset I already shuffled the

train = data[:800000]

test = data[800000:]
def undersample(data, ratio=1):

    conv = data[data.convert == 1]

    oth = data[data.convert == 0].sample(n=ratio*len(conv))

    return pd.concat([conv, oth]).sample(frac=1) #shuffle data



ustrain = undersample(train)



y = ustrain.convert

X = ustrain.drop('convert', axis=1)



print("Remaining rows", len(ustrain))
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import auc,roc_curve
from sklearn.linear_model import LogisticRegression

C_s = np.logspace(-10, 1, 11)

scores = list()

scores_std = list()

lr = LogisticRegression(penalty = 'l1')



for C in C_s:

    lr.C = C

    this_scores = cross_val_score(lr, X, y, cv=4,scoring='roc_auc')

    scores.append(np.mean(this_scores))

    scores_std.append(np.std(this_scores))

    

lr_results = pd.DataFrame({'score':scores, 'C':C_s}) 

lr_results
from sklearn.ensemble import RandomForestClassifier

msl_s = [1,2,4,8,16,32,64,128,256]

scores = list()

scores_std = list()

rf = RandomForestClassifier(n_estimators = 15)



for msl in msl_s:

    rf.min_samples_leaf = msl

    this_scores = cross_val_score(rf, X, y, cv=4,scoring='roc_auc')

    scores.append(np.mean(this_scores))

    scores_std.append(np.std(this_scores))

    

rf_results = pd.DataFrame({'score':scores, 'Minimum samples leaf': msl_s}) 

rf_results
from sklearn import svm

C_s = np.logspace(-10, 1, 11)

scores = list()

scores_std = list()

svc = svm.SVC(kernel='linear', probability=True)



for C in C_s:

    svc.C = C

    this_scores = cross_val_score(svc, X, y, cv=4,scoring='roc_auc', n_jobs=-1)

    scores.append(np.mean(this_scores))

    scores_std.append(np.std(this_scores))

    

svm_results = pd.DataFrame({'score':scores, 'C':C_s})    

svm_results
#not really elegant, but fits the pupose

y_preds = []



lr.C = lr_results.loc[lr_results['score'].idxmax()]['C']

y_preds.append(lr.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])



rf.min_samples_leaf = int(rf_results.loc[rf_results['score'].idxmax()]['Minimum samples leaf'])

y_preds.append(rf.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])



svc.C = svm_results.loc[svm_results['score'].idxmax()]['C']

y_preds.append(svc.fit(X,y).predict_proba(test.drop('convert', axis=1))[:,1])
model = ['LogR','RanF','SVM']

colors = ['b','r','g']



for i in range(0,3):

    fpr, tpr, thresholds = roc_curve(test.convert,y_preds[i])

    roc_auc = auc(fpr,tpr)

    plt.plot(fpr, tpr, 'b',label='%s AUC = %0.2f'% (model[i] ,roc_auc),  color=colors[i], linestyle='--')

    plt.legend(loc='lower right')

    

plt.title('Receiver Operating Characteristic')

plt.plot([-0.1,1.1],[-0.1,1.1],color='gray', linestyle=':')

plt.xlim([-0.1,1.1])

plt.ylim([-0.1,1.1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()