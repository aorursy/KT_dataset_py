# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
cancerDataSet=pd.read_csv('../input/wisconsin_breast_cancer.csv')

cancerDataSet.fillna(0,inplace=True) #removes non numbers

print(cancerDataSet['nuclei'])



cancerDataSet.head(10)
from sklearn.model_selection import train_test_split

x=cancerDataSet.iloc[:, 1:10]

y=cancerDataSet.iloc[:,10:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,random_state=1266415)

print(x_train.shape,y_train.shape)
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import precision_score,accuracy_score

my_classifier=SGDClassifier(random_state=1266415)



from sklearn.model_selection import cross_val_score

def findcolumns(alt, x_train):

    cols=list()

    

    if(alt&1==1):

      cols.append(x_train.columns[0])

    if(alt&2==2):

      cols.append(x_train.columns[1])

    if(alt&4==4):

      cols.append(x_train.columns[2])

    if(alt&8==8):

      cols.append(x_train.columns[3])

    if(alt&16==16):

      cols.append(x_train.columns[4])

    if(alt&32==32):

      cols.append(x_train.columns[5])

    if(alt&64==64):

      cols.append(x_train.columns[6])

    if(alt&128==128):

      cols.append(x_train.columns[7])

    if(alt&256==256):

      cols.append(x_train.columns[8])

    return cols

   

subsets=list()

for alt in range(1,512):

    subset = findcolumns(alt, x_train)

    print(alt, subset)

    subsets.append(subset)

  

print(len(subsets))
scores=list() 

sgdscores=list()

max_score=0

best_subset=0

sgdmodel=SGDClassifier(random_state=1266415)

for subset in subsets:

    cvscore=cross_val_score(sgdmodel,x_train[subset],y_train,cv=10,scoring="accuracy")

    scores.append(np.mean(cvscore))

    if scores[-1]>max_score:

        max_score=scores[-1]

        best_subset=subset

    #print(subset, scores[-1])

    sgdmodel.fit(x_train[subset],y_train)

    prediction=sgdmodel.predict(x_test[subset])

    sgdscores.append(accuracy_score(y_test,prediction))

    print(sgdscores)

    

    
print(scores)

print(max(scores))

print(max_score)

print(best_subset)

# the "thickness,shape,adhesion,nuclei,nucleoli,mitosis" subset looks the best given the cv scores.

# the accuracy of this subset is 0.9768762816131238
import matplotlib.pyplot as plt

colors=(0,0,0)



plt.scatter(scores, sgdscores, c=colors, alpha=0.2)

plt.title('scatter-plot')

plt.xlabel('cross validation scores')

plt.ylabel('test accuracy scores')

plt.show()
# Based on the plot, the the best cross validation score does not correspond to the best test accuracy



    





from sklearn.ensemble import RandomForestClassifier

rf_scores=list() 

rf_testscores=list()

rfmax_score=0

rfbest_subset=0

rfc=RandomForestClassifier(n_estimators=30,random_state=1266415)

for subset in subsets:

    rfscore=cross_val_score(rfc,x_train[subset],y_train,cv=10,scoring="accuracy")

    rf_scores.append(np.mean(rfscore))

    if rf_scores[-1]>rfmax_score:

        rfmax_score=rf_scores[-1]

        rfbest_subset=subset

    #print(subset, scores[-1])

    rfc.fit(x_train[subset],y_train)

    prediction=rfc.predict(x_test[subset])

    rf_testscores.append(accuracy_score(y_test,prediction))

    print(rf_testscores)
print(rfmax_score)

print(rfbest_subset)

# questions one and two are answered below
plt.scatter(rf_scores, rf_testscores, c=colors, alpha=0.2)

plt.title('scatter-plot')

plt.xlabel('cross validation scores')

plt.ylabel('test accuracy scores')

plt.show()
# again the best cvscore does not match the best test score

from sklearn.naive_bayes import GaussianNB

gnb_scores=list() 

gnb_testscores=list()

gnbmax_score=0

gnbbest_subset=0

gnb=GaussianNB()

for subset in subsets:

    gnbscore=cross_val_score(gnb,x_train[subset],y_train,cv=10,scoring="accuracy")

    gnb_scores.append(np.mean(gnbscore))

    if gnb_scores[-1]>gnbmax_score:

        gnbmax_score=gnb_scores[-1]

        gnbbest_subset=subset

    #print(subset, scores[-1])

    gnb.fit(x_train[subset],y_train)

    prediction=gnb.predict(x_test[subset])

    gnb_testscores.append(accuracy_score(y_test,prediction))

    print(gnb_testscores)

print(gnbmax_score)

print(gnbbest_subset)

#Question one and question two are answered below
plt.scatter(gnb_scores, gnb_testscores, c=colors, alpha=0.2)

plt.title('scatter-plot')

plt.xlabel('cross validation scores')

plt.ylabel('test accuracy scores')

plt.show()

#again the best cv result does not seem to match the best test-result
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



def plot_roc_curve(fpr,tpr,label=None):

    plt.plot(fpr,tpr,linewidth=2,label=label)

    plt.plot([0,1],[0,1],'k--')

sgd_clfRoc=SGDClassifier(random_state=1266415)

y_probas_sgd=cross_val_predict(sgd_clfRoc,x,y,cv=10,method="decision_function")

sgd_fp,sgd_tp,sgd_thr=roc_curve(y,y_probas_sgd)



rf_clfRoc=RandomForestClassifier(random_state=1266415)

y_probas_forest=cross_val_predict(rf_clfRoc,x,y,cv=10,method="predict_proba")

y_scores_forest=y_probas_forest[:,1]

rf_fp,rf_tp,rf_thr=roc_curve(y,y_scores_forest)



gnb_clfRoc=GaussianNB()

y_probas_gnb=cross_val_predict(gnb_clfRoc,x,y,cv=10,method="predict_proba")

y_scores_gnb=y_probas_gnb[:,1]

gnb_fp,gnb_tp,gnb_thr=roc_curve(y,y_scores_gnb)



plt.plot(sgd_fp,sgd_tp,"b:",label="SGD")

plt.plot(rf_fp,rf_tp,"r:",label="Random Forest")

plot_roc_curve(gnb_fp,gnb_tp,"GaussianNB")

plt.legend(loc="lower right")