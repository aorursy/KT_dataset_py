import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/cardiovasc-preprocessed/cardiovasc.csv',index_col='Unnamed: 0')

#Aufspalten in Features und Zielvariable

Xfeaturenames = set(df.columns) - set(['y'])

X = df[Xfeaturenames]

y = df['y'] #kopiere y

X.shape,y.shape #Anzahl Zeilen müssen übereinstimmen (je 68737)

X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)

X_train,X_valid, y_train, y_valid = train_test_split(X_train,y_train,train_size=5000,random_state=42)

X_train.shape,X_valid.shape
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

svc = SVC(kernel='linear',C=1.0)

svc.fit(X_train,y_train);

y_hat_valid = svc.predict(X_valid)

accuracy_score(y_hat_valid,y_valid)
def plot_parametercurve(train_scores,test_scores,param_range,xscale='log',xlabel='',title='Validation Curve'):

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel("Score")

    plt.ylim(0.0, 1.1)

    lw = 2

    if xscale=='log':

        pl=plt.semilogx

    else:

        pl=plt.plot

    pl(param_range, train_scores_mean, label="Training score",

                 color="darkorange", lw=lw,marker='x')

    



    plt.fill_between(param_range, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.2,

                     color="darkorange", lw=lw)

    pl(param_range, test_scores_mean, label="Cross-validation score",

                 color="navy", lw=lw,marker='x')

    plt.fill_between(param_range, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.2,

                     color="navy", lw=lw)

    plt.legend(loc="best")

    plt.ylim(0.5,1)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import validation_curve

svc_linear = SVC(kernel='linear')

C_param_range = [0.01,0.1,1,10,50,100]

#accuracies_svc = cross_val_score(estimator=svc, X=x_train_part, y=y_train_part, cv=10)

train_scores ,valid_scores = validation_curve(svc_linear,X_train,y_train,param_name='C',param_range=C_param_range,verbose=2,cv=5,n_jobs=5)
plot_parametercurve(train_scores,valid_scores,C_param_range,xscale='log',xlabel='C',title='Validation Curve for linear SVC')
optimaler_C_Wert=1.0 #<--- Finden Sie einen geeigneten Wert
accuracies_svc = cross_val_score(estimator=SVC(C=optimaler_C_Wert,#<--- Hier Ihre Wahl des Hyperparameters C eingeben

                                               kernel='linear' # <-- sollte gleich sein wie in svc in der vorletzeten Zelle

                                              ), 

                                 X=X_train, y=y_train, 

                                 cv=10 # die Anzahl Cross-Validation Folds. Sie sollten verstehen, was das bedeutet

                                )

print(f'Genauigkeit lineare SVC: {np.mean(accuracies_svc):1.3f}+/-{np.std(accuracies_svc):1.3f}')
gamma=1.0  #ob dieser Wert gut ist...? 

svc_rbf = SVC(kernel='rbf',gamma=gamma)

C_param_range = [0.01,0.1,1,10,50,100]



train_scores ,valid_scores = validation_curve(svc_rbf,X_train,y_train,param_name='C',param_range=C_param_range,verbose=2,cv=5,n_jobs=5)

plot_parametercurve(train_scores,valid_scores,C_param_range,xscale='log',xlabel='C',title=fr'Validation Curve for rbf SVM with $\gamma={gamma}$')
gamma_param_range = [0.001,0.01,0.1,0.5,1,10]

svc_rbf = SVC(kernel='rbf',C=1.0)



train_scores ,valid_scores = validation_curve(svc_rbf,X_train,y_train,param_name='gamma',param_range=gamma_param_range,verbose=2,cv=5,n_jobs=5)

plot_parametercurve(train_scores,valid_scores,C_param_range,xscale='log',xlabel='C',title=fr'Validation Curve for rbf SVM with $\gamma={gamma}$')
accuracies_svc = cross_val_score(estimator=SVC(C=1,kernel='rbf',gamma=1.0), X=X_train, y=y_train,n_jobs=5, cv=10,verbose=1)

print(f'Genauigkeit rbf-SVM: {np.mean(accuracies_svc):1.3f}+/-{np.std(accuracies_svc):1.3f}')
svc_rbf = SVC(kernel='rbf',C=2, gamma=0.1)
C=1.0 # <- Spielen Sie mit dem Wert des Parameters!

svc_rbf = SVC(kernel='rbf',C=C, gamma=0.1)





train_scores ,valid_scores = validation_curve(svc_rbf,X_train,y_train,param_name='gamma',param_range=gamma_param_range,verbose=2,cv=5,n_jobs=5)

plot_parametercurve(train_scores,valid_scores,C_param_range,xscale='log',xlabel=r'$\gamma$',title=f'Validation Curve for rbf SVM for C={C}')
from sklearn.model_selection import GridSearchCV

C_param_range=  [0.001,0.01,0.1,1,10,100,500,1000]

gamma_param_range

svc = SVC(kernel='rbf')

#GridSearchCV?

gs = GridSearchCV(svc,{'C':C_param_range,'gamma':gamma_param_range},cv=5,n_jobs=5,verbose=1,return_train_score=True)

gs.fit(X_train,y_train)
gs.cv_results_


accuracies_svc = cross_val_score(estimator=gs.best_estimator_, X=X_valid.iloc[:10000], y=y_valid.iloc[:10000],n_jobs=10, cv=10,verbose=2)

print(f'Genauigkeit rbf-SVM: {np.mean(accuracies_svc):1.3f}+/-{np.std(accuracies_svc):1.3f}')