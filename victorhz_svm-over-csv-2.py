import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import f1_score
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output as clear
import sklearn
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
train_df_sorted = train_df.sort_values('Target')
plt.plot(train_df_sorted.Target.tolist())
plt.show()
train_label = np.array(train_df_sorted.Target.tolist())
train_data = []
for i in range(len(train_df_sorted)):
    li = train_df_sorted.iloc[i]
    li = li.drop('Target')
    li = li.drop('Id').tolist()
    for _ in range(len(li)):
        try:
            float(li[_])
        except ValueError:
            if li[_] == 'yes':
                li[_] = np.exp(1)
            else:
                li[_] = 0
    li = np.array(li, dtype='float64')
    li = np.log1p(np.abs(li))
    train_data.append(li)
train_data = np.array(train_data)
print('done')
test_Id = test_df['Id'].tolist()
test_data = []
for i in range(len(test_df)):
    li = test_df.iloc[i]
    li = li.drop('Id').tolist()
    for _ in range(len(li)):
        try:
            float(li[_])
        except ValueError:
            if li[_] == 'yes':
                li[_] = np.exp(1)
            else:
                li[_] = 0
    li = np.array(li, dtype='float64')
    li = np.log1p(np.abs(li))
    test_data.append(li)
test_data = np.array(test_data)
print('done')
clf_svc = svm.SVC(verbose=True)
clf_svr = svm.SVR(verbose=True)
clf_svc.fit(train_data.clip(0,100000000),train_label)
clf_svr.fit(train_data.clip(0,100000000),train_label)
plt.plot(clf_svc.predict(train_data.clip(0,100000000)))
#clf_svc.predict(train_data.clip(0,100000000))
plt.plot(np.round(clf_svr.predict(train_data.clip(0,100000000))).clip(1,4))
f1_score(clf_svc.predict(train_data.clip(0,100000000)), train_label,average='macro')
f1_score(np.round(clf_svr.predict(train_data.clip(0,100000000))), train_label,average='macro')
pre_svc = clf_svc.predict(test_data.clip(0,100000000))
pre_svr = np.array(np.round(clf_svr.predict(test_data.clip(0,100000000))).clip(1,4),dtype='int64')
pre_svc_df = pd.DataFrame({'Id':test_Id, 'Target':pre_svc})
pre_svr_df = pd.DataFrame({'Id':test_Id, 'Target':pre_svr})
pre_svc_df.to_csv('submission_svc.csv',index=False)
pre_svr_df.to_csv('submission_svr.csv',index=False)