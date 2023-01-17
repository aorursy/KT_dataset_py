# import packages

import pandas as pd

import numpy as np

from itertools import cycle

from scipy import interp

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

import seaborn as sns

%matplotlib inline

# import classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import ShuffleSplit,GridSearchCV

from sklearn.svm import SVC

from sklearn.metrics import fbeta_score,make_scorer,roc_curve,auc,confusion_matrix

from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.ensemble.partial_dependence import partial_dependence

from imblearn.over_sampling import SMOTE

import xgboost as xgb
traindata=pd.read_csv('../input/train.csv')
Xtrain=traindata.iloc[:,:16]

ytrain=traindata.iloc[:,-1]

X_train,X_test,y_train, y_test=train_test_split(Xtrain, ytrain, test_size=0.2, random_state=50)
def FitModelGridSearch(X, y,model,params,classification=1,beta=1,accuracy=0,average='macro'):

    # Create cross-validation sets from the training data

    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 50)

    # build scorer

    if classification:

        if not accuracy:

            scoring_fnc = make_scorer(fbeta_score,beta=beta,average=average)

        else:

            scoring_fnc = make_scorer(accuracy_score)

    else:

        scoring_fnc = make_scorer(r2_score)

    grid = GridSearchCV(model,param_grid=params,scoring=scoring_fnc,cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data

    return grid



def plot_cv_results(results,params):

    time=results['mean_fit_time'];

    score_time=results['mean_score_time'];

    test_score=results['mean_test_score']

    print(list(params.values()))

    train_score=results['mean_train_score']

    ax=plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)

    plt.plot(list(params.values())[0],test_score,'r.-')

    plt.plot(list(params.values())[0],train_score,'k.-')

    plt.xlabel(list(params.keys())[0])

    plt.ylabel('score')

    plt.subplot(1,2,2)

    plt.plot(list(params.values())[0],time,'k.-')

    plt.plot(list(params.values())[0],score_time,'r.-')

    plt.xlabel(list(params.keys())[0])

    plt.ylabel('time (s)')

    

def PlotMultiClassROC(clf,X_test,y_test,lw,n_classes,title,save=1):

    y_score = clf.predict_proba(X_test)

    y_test_binary=np.zeros([y_test.shape[0],n_classes])

    for i in range(n_classes):

        y_test_binary[y_test==i+1,i]=1

    fpr = dict()

    tpr = dict()

    threshold=dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], threshold[i] = roc_curve(y_test_binary[:,i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])



    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])



    # Finally average it and compute AUC

    mean_tpr /= n_classes



    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



    # Plot all ROC curves

    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],

             label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

             color='deeppink', linestyle=':', linewidth=4)



    plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    for i, color in zip(range(n_classes), colors):

        plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i+1, roc_auc[i]))



    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title(title)

    plt.legend(loc="lower right")

    if save==1:

        plt.savefig('clf_xg_roc_unweight.eps',dpi=150)

    return tpr,fpr,threshold
#params={'learning_rate':[0.6,0.8,1.0,1.2,1.4]}

#clf_xg= FitModelGridSearch(X_train,y_train,

#                           xgb.XGBClassifier(random_state=50),

#                           params,beta=1)
#plot_cv_results(clf_xg.cv_results_,params=params)

#plt.savefig('clf_xg_unweight.eps',dpi=150)
# confusion_matrix

#X_test=np.array(X_test)

#cm_rf=confusion_matrix(y_test,clf_xg.predict(X_test))

#print(cm_rf)

#cm_rf = cm_rf.astype('float')/cm_rf.sum(axis=1)[:, np.newaxis]

#fig,ax=plt.subplots()

#sns.heatmap(cm_rf,cmap='jet')

#ax.set_xlabel('Predictive label')

#ax.set_ylabel('Observed label')

#ax.set_xticklabels([1,2,3])

#ax.set_yticklabels([3,2,1])

#plt.savefig('clf_xg_cm_unweight.eps',dpi=150)
#print('micro mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='micro'))

#print('mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='macro'))

#print('f1-score for class 1')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[1],average='macro'))

#print('f1-score for class 2')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[2],average='macro'))

#print('f1-score for class 3')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[3],average='macro'))
#lw=2

#n_classes=3

#xgb_tpr,xgb_fpr,xgb_threshold=PlotMultiClassROC(clf_xg,X_test,y_test,lw,n_classes,'Xgboost',save=1)
#fig,ax=plt.subplots()

#ax.plot(xgb_threshold[0],xgb_tpr[0],'--')

#ax.plot(xgb_threshold[0],xgb_fpr[0],'.-')

#ax.plot(xgb_threshold[1],xgb_tpr[1],'--')

#ax.plot(xgb_threshold[1],xgb_fpr[1],'.-')

#ax.plot(xgb_threshold[2],xgb_tpr[2],'--')

#ax.plot(xgb_threshold[2],xgb_fpr[2],'.-')

#ax.set_xlabel('Threshold')

#ax.set_ylabel('TPR or FPR')

#ax.set_title('Xgboost')

#ax.legend(['TPR1','FPR1','TPR2','FPR2','TPR3','FPR3'])

#ax.set_xlim([0,1])

#plt.savefig('clf_xg_tpr_fpr_unweight.eps',dpi=150)
#Xtest=pd.read_csv('../input/test.csv')

# learning rate is best as 1

# retrain the model with all training data set

#clf_xg=xgb.XGBClassifier(random_state=50,learning_rate=1.0)

#clf_xg.fit(Xtrain,ytrain)

#ypred=clf_xg.predict(Xtest)
#ypred=pd.Series(ypred)
#ypred.to_csv('xgboost_ypred_unweight.csv')
from sklearn import base
# three classes threshold model

# threshold1 is the 

class ThresholdClassifier(base.BaseEstimator, base.ClassifierMixin):

    def __init__(self,clf,threshold1,threshold2):

        self.clf = clf

        self.threshold1=threshold1

        self.threshold2=threshold2

    def fit(self, X, y):

        self.clf.fit(X,y)

        return self

    def set_threshold(self,threshold1,threshold2):

        self.threshold1=threshold1

        self.threshold2=threshold2

    def predict(self, X):

        prob=self.clf.predict_proba(X)

        y=self.clf.predict(X)

        check=np.zeros(len(X))

        for i in range(len(X)):

            if self.threshold1 is not None:

                if prob[i][2]>self.threshold1:

                    if check[i]==0:

                        y[i]=3

                        check[i]=1

            if self.threshold2 is not None:

                if prob[i][1]>self.threshold2:

                    if check[i]==0:

                        y[i]=2

                        check[i]=1

                #if prob[i][1]>2*self.threshold2:

                #        y[i]=2                

        return y
threshold_clf=ThresholdClassifier(xgb.XGBClassifier(random_state=50,learning_rate=1.0),0.3,None)
#for x in [0.1,0.2,0.3,0.4]:

#    threshold_clf.set_threshold(x,None)

#    # confusion_matrix

#    cm_rf=confusion_matrix(y_test,threshold_clf.predict(X_test))

#    cm_rf = cm_rf.astype('float')/cm_rf.sum(axis=1)[:, np.newaxis]

#    fig,ax=plt.subplots()

#    sns.heatmap(cm_rf,cmap='jet')

#    ax.set_xlabel('Predictive label')

#    ax.set_ylabel('Observed label')

#    ax.set_xticklabels([1,2,3])

#    ax.set_yticklabels([3,2,1])

#    plt.savefig(str(x)+'clf_xg_cm_weight.eps',dpi=150)
#print('micro mean f1-score')

#print(fbeta_score(y_pred=threshold_clf.predict(X_test),y_true=y_test,beta=1,average='micro'))

#print('mean f1-score')

#print(fbeta_score(y_pred=threshold_clf.predict(X_test),y_true=y_test,beta=1,average='macro'))

#print('f1-score for class 1')

#print(fbeta_score(y_pred=threshold_clf.predict(X_test),y_true=y_test,beta=1,labels=[1],average='macro'))

#print('f1-score for class 2')

#print(fbeta_score(y_pred=threshold_clf.predict(X_test),y_true=y_test,beta=1,labels=[2],average='macro'))

#print('f1-score for class 3')

#print(fbeta_score(y_pred=threshold_clf.predict(X_test),y_true=y_test,beta=1,labels=[3],average='macro'))
#from imblearn.over_sampling import SMOTE
#Xtrain=traindata.iloc[:,:16]

#ytrain=traindata.iloc[:,-1]

#X_train,X_test,y_train, y_test=train_test_split(Xtrain, ytrain, test_size=0.2, random_state=50)
# SMOTE method, generate minority class with nearest neighbour

# N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, 

# “SMOTE: synthetic minority over-sampling technique,

# ” Journal of artificial intelligence research, 321-357, 2002.

#sm = SMOTE(random_state=50)

#X_res, y_res = sm.fit_sample(X_train, y_train)

# check oversampling result

#print(pd.Series(y_res).value_counts())
# learning rate is best as 1

# retrain the model with all training data set

#clf_xg=xgb.XGBClassifier(random_state=50,learning_rate=1)

#clf_xg.fit(X_res,y_res)
# confusion_matrix

#X_test=np.array(X_test)

#cm_rf=confusion_matrix(y_test,clf_xg.predict(X_test))

#print(cm_rf)

#cm_rf = cm_rf.astype('float')/cm_rf.sum(axis=1)[:, np.newaxis]

#fig,ax=plt.subplots()

#sns.heatmap(cm_rf,cmap='jet')

#ax.set_xlabel('Predictive label')

#ax.set_ylabel('Observed label')

#ax.set_xticklabels([1,2,3])

#ax.set_yticklabels([3,2,1])

#plt.savefig('clf_xg_cm_weight_oversample.eps',dpi=150)
#print('micro mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='micro'))

#print('mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='macro'))

#print('f1-score for class 1')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[1],average='macro'))

#print('f1-score for class 2')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[2],average='macro'))

#print('f1-score for class 3')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[3],average='macro'))
#traindata=pd.read_csv('../input/train.csv')

#Xtrain=traindata.iloc[:,:16]

#ytrain=traindata.iloc[:,-1]

#X_train,X_test,y_train, y_test=train_test_split(Xtrain, ytrain, test_size=0.2, random_state=50)
#clf_xg=xgb.XGBClassifier(random_state=50,learning_rate=1)

#sample_weight=np.zeros([len(y_train),1])

#sample_weight[y_train==1]=1

#sample_weight[y_train==2]=60

#sample_weight[y_train==3]=1000

#clf_xg.fit(X_train,y_train,sample_weight=sample_weight)
#X_test=np.array(X_test)

#cm_rf=confusion_matrix(y_test,clf_xg.predict(X_test))

#print(cm_rf)

#cm_rf = cm_rf.astype('float')/cm_rf.sum(axis=1)[:, np.newaxis]

#fig,ax=plt.subplots()

#sns.heatmap(cm_rf,cmap='jet')

#ax.set_xlabel('Predictive label')

#ax.set_ylabel('Observed label')

#ax.set_xticklabels([1,2,3])

#ax.set_yticklabels([3,2,1])

#plt.savefig('clf_xg_cm_weight_sample_weight.eps',dpi=150)
#print('micro mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='micro'))

#print('mean f1-score')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,average='macro'))

#print('f1-score for class 1')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[1],average='macro'))

#print('f1-score for class 2')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[2],average='macro'))

#print('f1-score for class 3')

#print(fbeta_score(y_pred=clf_xg.predict(X_test),y_true=y_test,beta=1,labels=[3],average='macro'))
Xtest=pd.read_csv('../input/test.csv')

# learning rate is best as 1

# retrain the model with all training data set

threshold_clf=ThresholdClassifier(xgb.XGBClassifier(random_state=50,learning_rate=1.0),0.3,None)

threshold_clf.fit(Xtrain,ytrain)

ypred=threshold_clf.predict(Xtest)

ypred=pd.Series(ypred)

ypred.to_csv('xgboost_ypred_weight.csv')