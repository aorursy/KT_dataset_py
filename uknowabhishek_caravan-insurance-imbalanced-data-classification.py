import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Restrict minor warnings

import warnings

warnings.filterwarnings('ignore')



# to display all outputs of one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



pd.options.display.max_columns = 100



from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer as CTT

from sklearn.preprocessing import StandardScaler as ss, OneHotEncoder as ohe

from sklearn.ensemble import ExtraTreesClassifier as etc,RandomForestClassifier as rf

from sklearn.ensemble import AdaBoostClassifier as adc,BaggingClassifier as bgc

from imblearn.ensemble import BalancedRandomForestClassifier as brf

from sklearn.ensemble import GradientBoostingClassifier as GBC

from imblearn.ensemble import EasyEnsembleClassifier

from imblearn.ensemble import RUSBoostClassifier



from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import TomekLinks

from imblearn.combine import SMOTETomek



from imblearn.metrics import classification_report_imbalanced
df = pd.read_csv('../input/caravan-insurance-challenge/caravan-insurance-challenge.csv')

base = df[df['ORIGIN']=='train']

vldn = df[df['ORIGIN']=='test']

_=base.pop('ORIGIN')

_=vldn.pop('ORIGIN')

y_base=base.pop('CARAVAN')

y_vldn=vldn.pop('CARAVAN')
base.describe()
base.columns[base.isna().sum()>0]
pd.crosstab(df.ORIGIN,df.CARAVAN)
pd.DataFrame(df.nunique()).T

pd.DataFrame(base.nunique()).T

pd.DataFrame(vldn.nunique()).T

ax= plt.axes()

_=sns.heatmap(df.drop(columns=['ORIGIN']).corr(),cmap='Oranges',cbar=None,ax=ax)

_=ax.set_title('Correlation Heatmap')
f,axs=plt.subplots(1,2,figsize=(12,6))

sns.heatmap(df.iloc[:,1:44].corr(),ax=axs[0],vmin=-2, vmax=2,cbar=None)

sns.heatmap(df.iloc[:,44:-1].corr(),ax=axs[1],vmin=-2, vmax=2,cbar=None)

cat_col = ['MOSTYPE','MOSHOOFD']

num_cols = list(base.columns.values[43:])

ctt = CTT([('ss',ss(),num_cols),

           ('ohe',ohe(),cat_col)],remainder='passthrough')
cat_col = ['MOSTYPE','MOSHOOFD']

num_cols = list(base.columns.values[43:])

ctt = CTT([ ('ss',ss(),num_cols),

           ('ohe',ohe(),cat_col)],remainder='passthrough')

sm_trainX , sm_trainY = SMOTE(random_state=42).fit_resample(base,y_base)

tm_trainX ,tm_trainY = TomekLinks().fit_resample(base,y_base)

cmb_trainX, cmb_trainY = SMOTETomek(random_state=42).fit_resample(base,y_base)
print('Imbalanced Sample')

y_base.value_counts()

print('Over_sampled Sample')

sm_trainY.value_counts()

print('Under_Sampled Sample')

tm_trainY.value_counts()

print('Combine_sampled Sample')

cmb_trainY.value_counts()
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,classification_report

import itertools

def plot_confusion_matrix(y_true, y_pred, classes, ax=None, cmap=plt.cm.Blues):

    

    cm = confusion_matrix(y_true, y_pred)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    print(classification_report_imbalanced(y_true,y_pred))

    

    fig, ax = (plt.gcf(), ax)

    

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_title('Confusion Matrix')



    tick_marks = np.arange(len(classes))

    ax.set_xticks(tick_marks)

    ax.set_xticklabels(classes, rotation=45)

    ax.set_yticks(tick_marks)

    ax.set_yticklabels(classes)



    fmt = '.2f'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        ax.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    fig.tight_layout()

    ax.set_ylabel('True label')

    ax.set_xlabel('Predicted label')

def plot_roc(y_true, y_pred, ax=None):

    """Plot ROC curve""" 

    false_positive_rate, true_positive_rate, threshold = roc_curve(y_true, y_pred)

    roc_score = roc_auc_score(y_true,y_pred)

    

    fig, ax = (plt.gcf(), ax) if ax is not None else plt.subplots(1,1)



    ax.set_title("Receiver Operating Characteristic")

    ax.plot(false_positive_rate, true_positive_rate)

    ax.plot([0, 1], ls="--")

    ax.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

    ax.annotate('ROC: {:.5f}'.format(roc_score), [0.75,0.05])

    ax.set_ylabel("True Positive Rate")

    ax.set_xlabel("False Positive Rate")

    fig.tight_layout()

    return roc_score

def feat_imps(model, X_train, plot=False, n=None):

    """ Dataframe containing each feature with its corresponding importance in the given model"""

    fi_df = pd.DataFrame({'feature':X_train.columns,

                          'importance':model.feature_importances_}

                        ).sort_values(by='importance', ascending=False)

    if plot:

        fi_df[:(n if n is not None else 15)].plot.bar(x='feature',y='importance')

    else:

        return fi_df

def plot_cmroc(y_true, y_pred, classes=[0,1], normalize=True):

    """Convenience function to plot confusion matrix and ROC curve """

    fig,axes = plt.subplots(1,2, figsize=(9,4))

    plot_confusion_matrix(y_true, y_pred, classes=classes, ax=axes[0])

    roc_score = plot_roc(y_true, y_pred, ax=axes[1])

    fig.tight_layout()

    plt.show()

    return roc_score


InteractiveShell.ast_node_interactivity = "last"
from sklearn.metrics import recall_score,make_scorer,fbeta_score

def recall_1(y_true, y_pred):

    tp,fp,fn,tn = confusion_matrix(y_true, y_pred).ravel()

    return tn/(tn+fp)



recall_class_1 = make_scorer(recall_1, greater_is_better=True)



def f2(y_true, y_pred):

    return fbeta_score(y_true, y_pred,beta=2)



f2_score = make_scorer(f2, greater_is_better=True)
from sklearn.metrics import fbeta_score

from sklearn.model_selection import GridSearchCV

estimators = [rf(),brf(),adc()]

n_est = [50,55,60,65]

rep = []

for e in estimators:

    pipe = Pipeline([

                    ('ct',ctt),

                    ('e',e)

                ])

    rcv = GridSearchCV(pipe,{'e__n_estimators':n_est},scoring=recall_class_1,cv=5)

    rcv.fit(base,y_base)

    rep.append(['Biased data',e.__class__.__name__,rcv.best_params_['e__n_estimators'],rcv.best_score_])

    rcv.fit(tm_trainX,tm_trainY)

    rep.append(['Tomek Link',e.__class__.__name__,rcv.best_params_['e__n_estimators'],rcv.best_score_])

    rcv.fit(sm_trainX,sm_trainY)

    rep.append(['SMOTE',e.__class__.__name__,rcv.best_params_['e__n_estimators'],rcv.best_score_])

rep  
pipe = Pipeline([

                    ('ct',ctt),

                    ('e',rf(random_state=0,n_jobs=-1))

                ])



pipe.fit(cmb_trainX,cmb_trainY)

plot_cmroc(y_base,pipe.predict(base))

plot_cmroc(y_vldn,pipe.predict(vldn))
rus = RUSBoostClassifier(n_estimators=10,

                         base_estimator=bgc(n_estimators=10,base_estimator=brf(random_state=0,n_jobs=-1)))

pipe = Pipeline([('ct',ctt),('e',rus)])

pipe.fit(cmb_trainX,cmb_trainY)

plot_cmroc(y_base,pipe.predict(base))

plot_cmroc(y_vldn,pipe.predict(vldn))
eec=EasyEnsembleClassifier(base_estimator=GBC(),n_jobs=-1, random_state=0,sampling_strategy='majority')

pipe_eec = Pipeline([('ct',ctt),('e',eec)])

pipe_eec.fit(base,y_base)

plot_cmroc(y_base,pipe_eec.predict(base))

plot_cmroc(y_vldn,pipe_eec.predict(vldn))
