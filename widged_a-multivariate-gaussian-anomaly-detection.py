import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score, confusion_matrix,classification_report
import math
df = pd.read_csv('../input/creditcard.csv')
df.shape
col_to_drop = ['V8','V13','V15','V20','V21','V22','V23','V24','V26','V27','V28','Time']
a = df.drop(col_to_drop, axis=1)


a['Amount']=a['Amount'].apply(lambda x: math.log(x**.5+1)) # Getting to a normal curve is done based on visualization and iteration to get the shape
a['V1']=a['V1'].apply(lambda x: x**.2) # Getting to a normal curve is done based on visualization and iteration to get the shape

mu = a['Amount'].mean()
sigma = a['Amount'].std()
a['Amount'] = (a['Amount']-mu)/sigma
a_good = a[a['Class']==0]
a_fraud = a[a['Class']==1]
gr = len(a_good)
fr = len(a_fraud)
g_tr = a_good[:gr*60//100]
g_cv = a_good[(gr*60//100)+1:(gr*80//100)]
g_t = a_good[(gr*80//100)+1:]
fr_cv = a_fraud[:fr*50//100]
fr_t = a_fraud[(fr*50//100)+1:]
a_cv = pd.concat([g_cv,fr_cv])
a_t = pd.concat([g_t,fr_t])
g_tr.drop('Class',inplace=True,axis=1)
p = multivariate_normal(mean=np.mean(g_tr,axis=0), cov=np.cov(g_tr.T))
a_cv_X = a_cv.drop('Class',axis=1)
x = p.pdf(a_cv_X)
epsilons = [1e-60,1e-65,1e-70,1e-75,1e-80,1e-85,1e-90,1e-95,1e-100,1e-105]
pred = (x<epsilons[2])
f = f1_score(a_cv['Class'],pred,average='binary')
print(f)
f_max = 0
e_final=0

for e in epsilons:
    pred = (x<e)
    f = f1_score(a_cv['Class'],pred,average='binary')
    print(f,e)
    if f>f_max:
        f_max=f
        e_final=e

print (e_final,f_max)
a_t_X = a_t.drop('Class',axis=1)
x = p.pdf(a_t_X)
from sklearn.metrics import confusion_matrix
pred = (x<e_final)
confusion_matrix(a_t['Class'],pred)
from sklearn.metrics import classification_report
print(classification_report(a_t['Class'],pred))
