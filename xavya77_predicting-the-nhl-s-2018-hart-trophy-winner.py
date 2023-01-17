import pandas as pd

import numpy as np

import scipy as sp

import statsmodels.api as sm

import statsmodels.formula.api as smf

import sklearn.linear_model as sklm

from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score, roc_curve, roc_auc_score

from sklearn import preprocessing

from sklearn import neighbors

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})

import matplotlib.pylab as plt

%matplotlib inline 

plt.style.use('seaborn-whitegrid')

plt.rc('text', usetex = False)

plt.rc('font', family = 'serif')

plt.rc('xtick', labelsize = 10) 

plt.rc('ytick', labelsize = 10) 

plt.rc('font', size = 12) 

plt.rc('figure', figsize = (12, 5))
#reading file

nhl = pd.read_csv('../input/NHL.csv', sep=',', encoding='latin-1')

#was receiving an error due to weird characters in the file so I had to specify the encoding to read
nhl.info()
nhl.head(11)
#nhl.Player = pd.factorize(nhl.Player)[0]

nhl.Pos = pd.factorize(nhl.Pos)[0]

nhl.Tm = pd.factorize(nhl.Tm)[0]

#nhl.ATOI = pd.factorize(nhl.ATOI)[0]

#nhl.Season = pd.factorize(nhl.Season)[0]
nhl.shape
nhl = nhl.dropna( how='any')

nhl.shape
#first, let us split the dataset to train and test portions. winners = nhl[nhl['HART'] == 1]

ctest =  nhl[nhl['Season'] > 2013]

ctrain = nhl[nhl['Season'] <= 2013]
ctrain.shape
ctest.shape
ctest.Season
ctrain.Season
# the response variable to use for classification: Hart trophy winner 1 or 0? Designating 1 as the response variable

ytest = ctest.HART 

ytrain = ctrain.HART
Xtrain=preprocessing.scale(ctrain.drop(['HART','Votes','Player'], axis=1).astype('float64'))

Xtest=preprocessing.scale(ctest.drop(['HART','Votes','Player'], axis=1).astype('float64'))
lreg=sklm.LogisticRegression()

lreg.fit(Xtrain, ytrain)
yhattest1 = lreg.predict(Xtest)

cm = confusion_matrix(ytest, yhattest1)

print("Test Classification accuracy:", lreg.score(Xtest,ytest))

print("\n")

print("Confusion Matrix:\n", cm)

print("\n")

print("Classification Report:\n",classification_report(ytest, yhattest1))
lreg.predict(Xtest)[0:10]
lreg.predict_proba(Xtest)[0:10]
ypredprob = lreg.predict_proba(Xtest)[:, 1]
ypredprob
plt.hist(ypredprob, bins=6)

plt.xlim(0,1)

plt.title('Histogram of predicted probabilities')

plt.xlabel('Predicted probability of insurance purchase')

plt.ylabel('Frequency')
ctest['probscore']=ypredprob
ctest.head(6)
#first, let us split the dataset to train and test portions. winners = nhl[nhl['HART'] == 1]

mvps17 =  ctest[(ctest['Season'] == 2017) & (ctest['probscore'] > 0)]

mvps18 = ctest[(ctest['Season'] == 2018) & (ctest['probscore'] > 0)]
print ('The average probability score of 2017 players is: ', mvps17['probscore'].mean(), '.') 

print ('The average probability score of 2018 players is: ', mvps18['probscore'].mean(), '.') 
print(mvps17.shape)

print(mvps18.shape)
#reshaping to only players above the mean

mvps17 =  ctest[(ctest['Season'] == 2017) & (ctest['probscore'] > 0.002)]

mvps18 = ctest[(ctest['Season'] == 2018) & (ctest['probscore'] > 0.002)]
print(mvps17.shape)

print(mvps18.shape)
mvps17.probscore.describe()
mvps18.probscore.describe()
mvps17.sort_values(by='probscore', ascending=False, inplace=True)

mvps17.head(20)
mvps18.sort_values(by='probscore', ascending=False, inplace=True)

mvps18.head(15)
mvps17hist=mvps17['probscore']

mvps17hist.hist(normed=0, histtype='stepfilled', bins=6)



plt.xlabel('probscore',fontsize=15)

plt.ylabel('# of players',fontsize=15)

plt.show()
mvps18hist=mvps18['probscore']

mvps18hist.hist(normed=0, histtype='stepfilled', bins=6)



plt.xlabel('probscore',fontsize=15)

plt.ylabel('# of players',fontsize=15)

plt.show()
mvps18hist2 = mvps18[['Player','probscore']]

mvps17hist2 = mvps17[['Player','probscore']]
mvps18hist2.set_index('Player', inplace=True)

mvps17hist2.set_index('Player', inplace=True)
mvps18hist2 = mvps18hist2[0:10]
mvps18hist2
mvps18hist2.plot(kind='barh', color='Red', alpha=0.4, 

              title='Probabilty Score for Winning the Hart Trophy 2018')

plt.savefig('probscore2018.png', dpi=300, bbox_inches='tight')
mvps17hist2 = mvps17hist2[0:10]
mvps17hist2
mvps17hist2.plot(kind='barh', color='Red', alpha=0.4,

              title='Probabilty Score for Winning the Hart Trophy 2017')

plt.savefig('probscore2017.png', dpi=300, bbox_inches='tight')
fpr, tpr, thresholds = roc_curve(ctest.HART, ypredprob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1], 'k--', color='r')

plt.title('ROC curve for Hart Trophy classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.savefig('roc.png')

print("Area under the curve (AUC):\n", roc_auc_score(ctest.HART, ypredprob))

#higher AUC indicates better classifier