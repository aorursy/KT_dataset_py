#PMR 3508 
#wesley silva soares 

#classificador de deteccao de spam
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/spambase"))

# Any results you write to the current directory are saved as output.
trainbase = pd.read_csv("../input/spambase/train_data.csv")
testbase = pd.read_csv("../input/spambase/test_features.csv")
trainbase.shape
trainbase.head()
trainbase["ham"].value_counts().plot(kind="bar")
trainbase.corr()
plt.matshow(trainbase.corr())
plt.colorbar()
correlacoes = trainbase.corr()
correlacao_ham = correlacoes["ham"].abs()
correlacao_ham.sort_values()
Xtrainspam = trainbase[["word_freq_your","word_freq_000","char_freq_$","word_freq_remove",
                      "word_freq_you","word_freq_free","word_freq_business","capital_run_length_total",
                      "word_freq_hp","capital_run_length_total"]]
Xtestspam = testbase[["word_freq_your","word_freq_000","char_freq_$","word_freq_remove",
                      "word_freq_you","word_freq_free","word_freq_business","capital_run_length_total",
                      "word_freq_hp","capital_run_length_total"]]
Ytrainspam = trainbase.ham

n_neighborsMaiorScore =0
melhorneighbors = 0
for n_neighborsTemp in range(3,20,1):
    knn = KNeighborsClassifier(n_neighbors=n_neighborsTemp)
    scores = cross_val_score(knn, Xtrainspam, Ytrainspam, cv=10)
    scores
    if scores.mean() > n_neighborsMaiorScore:
        n_neighborsMaiorScore = scores.mean()
        melhorneighbor = n_neighborsTemp



#mostra a media dos scores de treino
n_neighborsMaiorScore
#melhor valor de k 
melhorneighbor
#fez o predict usando no metodo knn 
knn.fit(Xtrainspam,Ytrainspam)
YtestPred = knn.predict(Xtestspam)
YtestPred

# Fazendo o classificador usando o metodo Naive Bayes, obtivemos um resultado melhor que o knn
clf = BernoulliNB()
scores = cross_val_score(clf, Xtrainspam, Ytrainspam, cv=10)

scores.mean()
clf.fit(Xtrainspam, Ytrainspam)
YtestPred = clf.predict(Xtestspam)
sub = pd.DataFrame({"id":testbase.Id, "ham":YtestPred})
sub.to_csv("submission.csv", index=False)
sub