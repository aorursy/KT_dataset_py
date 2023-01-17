import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Hier sind die Daten der assoziierten Kaggle-Competition:

!ls ../input
#Pandas ist praktisch, um .csv-Dateien zu laden...

dftrain = pd.read_csv('../input/train.csv',index_col='Id')

xtest = pd.read_csv('../input/Xtest.csv',index_col='Id')

dftrain.head()
#...aber längst nicht nur deswegen

xtrain = dftrain.x.values.reshape(-1,1)

ytrain = dftrain.y.values.reshape(-1,1)
#Wir instantiieren einen Regressor:

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
#Wir trainieren den Regressor auf den Trainingsdaten

reg.fit(xtrain,ytrain)
#Wir generieren Vorhersagen: Sowohl auf dem Trainingsdatensatz zur Plausibilitätsüberprufung,... 

yhat_train = reg.predict(xtrain).ravel()

#...als auch auf dem Testdatensatz zur Vorhersage für die Competition:

yhat_test = reg.predict(xtest).ravel()
dftrain.plot(x='x',y='y');

plt.plot(xtrain,yhat_train,label='prediction on training set');

plt.legend();
plt.plot(xtrain,ytrain,'.',ms=0.8,label='training data');

plt.plot(xtrain,yhat_train,label='prediction on training set');

plt.plot(xtest,yhat_test,label='prediction on test set');

plt.legend();
submission = pd.Series(yhat_test,name='y')

submission.index.name='Id'
#Diese Submission-Datei kann später direkt zur Bewertung an die Kaggle-Competition weitergeleitet werden:

submission.to_csv('testsubmission.csv',header=True)