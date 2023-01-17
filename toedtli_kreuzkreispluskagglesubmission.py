from sklearn.model_selection import cross_val_score
import numpy as np
import os
!ls ../input/csvversion-der-kreuz-kreis-und-plusdaten
outputpath = r"../input/csvversion-der-kreuz-kreis-und-plusdaten"
os.listdir(outputpath)
import pandas as pd
dftrain = pd.read_csv(os.path.join(outputpath,'KreuzKreisPlus_train.csv'),index_col='basename')
dftest = pd.read_csv(os.path.join(outputpath,'KreuzKreisPlus_test.csv'),index_col='basename')
dftrain.head()
import re
test_indices = dftest.index.map(lambda s:int(re.sub('-u.*$','',s)))
from sklearn.tree import DecisionTreeClassifier
ytrain = dftrain.target
del dftrain['target']
Xtrain = dftrain
clf=DecisionTreeClassifier()
clf.fit(Xtrain,ytrain)
scores = cross_val_score(clf,Xtrain,ytrain,cv=10)
f'{np.mean(scores):1.2f}+/-{np.std(scores,ddof=1):1.2f}'
yhat = clf.predict(dftest.values)
pred_test = pd.Series(yhat,name='target')
pred_test.index = test_indices
pred_test.index.name='id'
fn = r'Submission.csv'
pred_test.to_csv(fn,header=True)
from IPython.display import FileLink
FileLink(fn)
#Linux (?):
#!head -n r'C:\Users\TOB\Data_Science_Scratch\MaLe-Entwicklung\FFHS\MaLe\Daten\sampleprediction.csv'
#Windows:
#%alias head powershell -command "& {Get-Content %s -Head 5}"
#%head 'sampleprediction.csv'
!head -n 5 Submission.csv