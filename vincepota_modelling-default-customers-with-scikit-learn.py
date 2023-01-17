%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib.pylab import plt



from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import preprocessing, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

import itertools



df = pd.read_csv('../input/UCI_Credit_Card.csv')

df.columns = df.columns.str.lower()

df.drop('id', axis=1, inplace=True) # we do not need id
df['sex'] = df['sex'].map({2:'female', 1:'male'})

df['marriage'] = df['marriage'].map({1:'married', 2:'single', 3:'other', 0: 'other'}) 

df['education'] = df['education'].map({1:'graduate school', 2:'university', 3:'high school', 4:'others', 5:'unknown', 6:'unknown', 0:'unknown'})

df['pay_0'] = df['pay_0'].astype(str) 

df['pay_2'] = df['pay_2'].astype(str) 

df['pay_3'] = df['pay_3'].astype(str) 

df['pay_4'] = df['pay_4'].astype(str) 



df.head()
X = pd.get_dummies(df[df.columns[:-1]],columns=['sex','marriage','education','pay_0','pay_2','pay_3','pay_4','pay_5','pay_6'])

y = df[df.columns[-1]]

features = X.columns



scaler = preprocessing.StandardScaler()

X = scaler.fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)
clfs = {'GradientBoosting': GradientBoostingClassifier(learning_rate= 0.05, max_depth= 6,

                                                        n_estimators=200, max_features = 0.3,

                                                        min_samples_leaf = 5),

        'LogisticRegression' : LogisticRegression(C = 1.0),

        'GaussianNB': GaussianNB(),

        'RandomForest': RandomForestClassifier(n_estimators=50)

        }
cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score', 'accuracy']

models_report = pd.DataFrame(columns = cols)

feature_importance = pd.DataFrame()



conf_matrix = dict()



for clf, clf_name in zip(clfs.values(), clfs.keys()):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    y_score = clf.predict_proba(X_test)[:,1]



    print('Computing{}'.format(clf_name))

    

    if (clf_name == 'RandomForest') | (clf_name == 'GradientBoosting'):

        tmp_fi = pd.Series(clf.feature_importances_)

        feature_importance[clf_name] = tmp_fi

        



    tmp = pd.Series({ 

                     'model': clf_name,

                     'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),

                     'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),

                     'precision_score': metrics.precision_score(y_test, y_pred),

                     'recall_score': metrics.recall_score(y_test, y_pred),

                     'f1_score': metrics.f1_score(y_test, y_pred),

                     'accuracy': metrics.accuracy_score(y_test, y_pred)},

                   )



    models_report = models_report.append(tmp, ignore_index = True)



    conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)



    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

    fpr, tpr, _ = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)



    plt.figure(1, figsize = (6,5))

    plt.xlabel('fpr')

    plt.ylabel('tpr')

    plt.plot(fpr, tpr, label = clf_name)

    plt.legend(prop={'size':11})

plt.plot([0,1], [0,1], c = 'black')

plt.show()
models_report
def plot_confusion_matrix(cm, ax, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set_title(title)

    #ax.set_colorbar()

    tick_marks = np.arange(len(classes))

    ax.set_yticks(tick_marks)

    ax.set_yticklabels(classes, rotation=35)



    ax.set_xticks(tick_marks)

    ax.set_xticklabels(classes, rotation=35)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        ax.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    #plt.tight_layout()

    ax.set_ylabel('True label')

    ax.set_xlabel('Predicted label')
import matplotlib.gridspec as gridspec



fig = plt.figure(figsize=(8, 8)) 

gs = gridspec.GridSpec(2, 2)



ax1 = plt.subplot(gs[0,0])

ax2 = plt.subplot(gs[0,1])

ax3 = plt.subplot(gs[1,0])

ax4 = plt.subplot(gs[1,1])



for c, ax in zip(conf_matrix.keys(), [ax1,ax2,ax3,ax4]):

    plot_confusion_matrix(conf_matrix[c].values, ax, title = c, classes=['No default','Default'])



plt.tight_layout()

plt.show()
fi = feature_importance



fi.index = features

fi = fi.head(15) # Only take the 15 most important metrics

fi = fi.sort_values('GradientBoosting', ascending=False)

fi = (fi / fi.sum(axis=0)) * 100

fi.plot.barh(title = 'Feature importances for Tree algorithms', figsize = (6,9))