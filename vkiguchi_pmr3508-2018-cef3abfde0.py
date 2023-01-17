import pandas as pd
import numpy as np
import sklearn
import os
from matplotlib import pyplot as plt
os.listdir("../input/")
filepath = "../input/train.csv"
crica = pd.read_csv(filepath,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
crica
listNull = [x for x in [(row,crica[row].isnull().sum()) for row in crica] if x[1]!=0]
print(listNull)
for row in crica.drop("Id",axis='columns'):
    print(row, crica[row].value_counts(), sep='\n')
crica.drop(["v2a1", "v18q1", "rez_esc", "SQBmeaned", "meaneduc"], axis='columns', inplace=True) #NaN
crica["idhogar"] = crica["idhogar"].transform(lambda x: int(x,16))
print(crica["idhogar"].value_counts())
crica["dependency"] = crica["dependency"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
print(crica["dependency"].value_counts())
crica["edjefe"] = crica["edjefe"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
print(crica["edjefe"].value_counts())
crica["edjefa"] = crica["edjefa"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
print(crica["edjefa"].value_counts())
#crica.drop(["dependency", "edjefe", "edjefa", "idhogar", "Id"], axis='columns', inplace=True) #strings
#crica.drop(["dependency", "edjefe", "edjefa", "Id"], axis='columns', inplace=True) #strings
crica.drop(["idhogar", "Id"], axis='columns', inplace=True) #strings
nCrica = crica.dropna()
nCrica
testfilepath = "../input/test.csv"
testCrica = pd.read_csv(testfilepath,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testCrica
listNull = [x for x in [(row,testCrica[row].isnull().sum()) for row in testCrica] if x[1]!=0]
print(listNull)
testCrica.drop(["v2a1","v18q1", "rez_esc", "SQBmeaned", "meaneduc"], axis='columns', inplace=True) #Nan
testCrica["idhogar"] = testCrica["idhogar"].transform(lambda x: int(x,16))
testCrica["dependency"] = testCrica["dependency"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
testCrica["edjefe"] = testCrica["edjefe"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
testCrica["edjefa"] = testCrica["edjefa"].transform(lambda x: 1 if x=="yes" else 0 if x=="no" else x)
#testCrica.drop(["dependency", "edjefe", "edjefa", "idhogar"], axis='columns', inplace=True) #strings
#testCrica.drop(["dependency", "edjefe", "edjefa"], axis='columns', inplace=True) #strings
testCrica.drop("idhogar", axis='columns', inplace=True) #strings
nTestCrica = testCrica
nTestCrica
Ycrica = nCrica.Target
Xcrica = nCrica.drop("Target", axis="columns")
Xcrica
XtestCrica = nTestCrica.drop("Id", axis='columns')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, Xcrica, Ycrica, cv=10)
scores
np.mean(scores)
knn.fit(Xcrica,Ycrica)
YtestPred = knn.predict(XtestCrica)
YtestPred
savepath = "results.csv"
#YtestPred.to_csv(savepath)
#import numpy as np
#np.savetxt("foo.csv", YtestPred, delimiter=",")
ypanda = pd.DataFrame(YtestPred, columns = ["Target"])
totalypanda = nTestCrica.combine_first(ypanda)
totalypanda["Target"] = totalypanda["Target"].transform(lambda x: int(x))
saveYpanda = totalypanda[["Id", "Target"]]
saveYpanda.to_csv(savepath, index=False)
saveYpanda
