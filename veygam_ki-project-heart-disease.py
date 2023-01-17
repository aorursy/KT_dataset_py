import numpy as np 

import pandas as pd

import os

import pandas_profiling

import matplotlib.pyplot as plt

%matplotlib inline



random_state = 42



print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/heart.csv')

df.head()
df.info()
df.describe()
values = [96, 207]

labels = ['Weiblich', 'Männlich']

plt.pie(values, labels= values,counterclock=False, shadow=True)

plt.title('Geschlechterverteilung der Patienten')

plt.legend(labels,loc=3)

plt.show()
df.groupby('sex').age.plot(kind='kde')

plt.title('Alter der Patienten abhängig von Geschlecht')

labels = ['Weiblich', 'Männlich']

plt.legend(labels,loc=1)
values = [138, 165]

labels = ['Nicht herzkrank', 'Herzkrank']

plt.pie(values, labels= values,counterclock=False, shadow=True)

plt.title('Verteilung der Präsenz von Herzkrankheiten')

plt.legend(labels,loc=3)

plt.show()
ct_tarsex = pd.crosstab(df.target, df.sex)

ct_tarsex.plot.bar(stacked=True)

labels = ['Weiblich', 'Männlich']

plt.legend(labels,loc=3)

plt.title('Präsenz von Herzkrankheiten nach Geschlechtern')
values = [143, 86, 50, 23]

labels = ['Typisches Beklemmungsgefühl / Angina', 'Untpyische Angina', 'Nicht-anginale Schmerzen', 'Ohne erkennbare Symptome']

plt.pie(values, labels= values,counterclock=False, shadow=True)

plt.title('Verteilung der Schmerzkategorien')

plt.legend(labels, loc=3)

plt.show()
ct2_anglabel = pd.crosstab(df.target, df.cp)

ct2_anglabel.plot.bar(stacked=True)



plt.title('Schmerzkategorien gemappt zu vorhandenen Herzkrankheiten')
ct2_calabel = pd.crosstab(df.target, df.ca)

ct2_calabel.plot.bar(stacked=True)



plt.title('Anzahl erkannter Gefäße im Verhältnis zu präsenten Herzkrankheiten')
norm_bot = 120

norm_top = 129



result = plt.hist(df[['trestbps']].values, bins=24, color='dodgerblue', edgecolor='k', alpha=0.65)

plt.axvline(norm_bot, color='r', linewidth=2)

plt.axvline(norm_top, color='r', linewidth=2)



_, max_ = plt.ylim()



plt.text(160 + 160/10, 

         max_ - max_/10, 

         'Optimal      {:}'.format(norm_bot))



plt.text(160 + 160/10, 

         max_ - max_/10 - 5, 

         'Normal <= {:}'.format(norm_top))



plt.title('Werteverteilung des Ruhepuls')
avg = 240



result = plt.hist(df[['chol']].values, bins=24, color='dodgerblue', edgecolor='k', alpha=0.65)

plt.axvline(avg, color='r', linewidth=2)



_, max_ = plt.ylim()



plt.text(400 + 400/10, 

         max_ - max_/10, 

         'Durchschnitt: {:}'.format(avg))



plt.title('Werteverteilung der Cholesterin-Konzentration in mg/dl')
#pandas_profiling.ProfileReport(df)
# Löschen der Duplikate und dauerhafte Speicherung mtihilfe von inplace=True, andernfalls würde eine Kopie erstellt werden

df.drop_duplicates(inplace=True)
df.corr()
plt.matshow(df.corr())
from sklearn.feature_selection import f_classif, SelectKBest
def select_kbest_clf(data_frame, target, k=5):



    feat_selector = SelectKBest(f_classif, k=k)

    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])

    

    feat_scores = pd.DataFrame()

    feat_scores["F Score"] = feat_selector.scores_

    feat_scores["P Value"] = feat_selector.pvalues_

    feat_scores["Support"] = feat_selector.get_support()

    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns

    

    return feat_scores 
select_kbest_clf(df, 'target')
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import f_classif, SelectKBest

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
#Visualizieren der Confusion Matrix

#Source: https://github.com/rohanjoseph93/Python-for-data-science/blob/master/Grid%20Search%20-%20Breast%20Cancer.ipynb

from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=2)



import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)

    print('False Negative Rate (should be close to 0): ', cnf_matrix[0][1] / (cnf_matrix[0][1] + cnf_matrix[1][1]))



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
from sklearn.model_selection import train_test_split



X, y = df.iloc[:,:-1],df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)



print('Training Shapes:', X_train.shape, y_train.shape)

print('Test Shapes:', X_test.shape, y_test.shape)
from sklearn.linear_model import SGDClassifier



sgdcd = SGDClassifier(random_state = random_state)

sgdcd.fit(X_train,y_train)

ac_sgdcd = sgdcd.score(X_test, y_test)



print('Dummy Classifier Accuracy:', ac_sgdcd)
cnf_matrix = confusion_matrix(y_test, sgdcd.predict(X_test))



plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - Dummy Classifier')

a = plt.gcf()

a.set_size_inches(4,3)

plt.show()
from sklearn.linear_model import SGDClassifier



sgdc_pipe  = Pipeline([

    ('kbest', SelectKBest(f_classif)),

    ('scaler', StandardScaler()),

    ('sgdc', SGDClassifier())

])



sgdc_pipe.set_params(

    kbest__k=5,

    sgdc__eta0=0.1, sgdc__random_state=random_state, sgdc__n_jobs=-1

)



sgdc_pipe.fit(X_train,y_train)



ac_sgdc = sgdc_pipe.score(X_test, y_test)

print('SGD Classifier Accuracy:', ac_sgdc)
sgdc_params = {

        'sgdc__penalty': ['l1', 'l2', 'none', 'elasticnet'],

        'sgdc__learning_rate': ['constant', 'optimal', 'invscaling'],

    

        'kbest__k': [3, 5, 7, 9, 11, 13]

        }
grid_search_sgdc = GridSearchCV(sgdc_pipe, param_grid=sgdc_params, scoring='roc_auc', n_jobs=-1)

grid_search_sgdc.fit(X_train, y_train)



grid_search_sgdc.best_params_
grid_search_sgdc.best_score_
ac_sgdc_cv = grid_search_sgdc.best_estimator_.score(X_test,y_test)

print('SGD Classifier Accuracy CV:', ac_sgdc_cv)
cnf_matrix = confusion_matrix(y_test, grid_search_sgdc.best_estimator_.predict(X_test))



plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - SGDClassifier + Grid Search')

a = plt.gcf()

a.set_size_inches(4,3)

plt.show()
from sklearn.tree import DecisionTreeClassifier



dtc_pipe = Pipeline([

    ('kbest', SelectKBest(f_classif)),

    ('scaler', StandardScaler()),

    ('dtc', DecisionTreeClassifier(random_state=random_state))

])



dtc_pipe.set_params(

    kbest__k=5

)



dtc_pipe.fit(X_train, y_train)



ac_dtc = dtc_pipe.score(X_test, y_test)

print('DecisionTree Classifier Accuracy:', ac_dtc)
dtc_params = {

        'dtc__max_depth': [2, 3, 5, 10, None],

        'dtc__max_leaf_nodes': [3, 5, 8, 10, 15, 20, None],

    

        'kbest__k': [3, 5, 7, 9, 11, 13]

        }
grid_search_dtc = GridSearchCV(dtc_pipe, param_grid=dtc_params, scoring='roc_auc', n_jobs=-1)

grid_search_dtc.fit(X_train, y_train)



grid_search_dtc.best_params_
grid_search_dtc.best_score_
ac_dtc_cv = grid_search_dtc.best_estimator_.score(X_test, y_test)

print('DecisionTree Classifier Accuracy CV:', ac_dtc_cv)
cnf_matrix = confusion_matrix(y_test, grid_search_dtc.best_estimator_.predict(X_test))



plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - DecisionTreeClassifier + Grid Search')

a = plt.gcf()

a.set_size_inches(4,3)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn_pipe = Pipeline([

    ('kbest', SelectKBest(f_classif)),

    ('scaler', StandardScaler()),

    ('knn', KNeighborsClassifier())

])



knn_pipe.set_params(

    kbest__k=5,

    knn__n_jobs=-1

)



knn_pipe.fit(X_train,y_train)



ac_knn = knn_pipe.score(X_test, y_test)

print('KNN Classifier Accuracy:', ac_knn)
knn_params = {

        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],

        'knn__algorithm': ['ball_tree', 'kd_tree', 'brute'],

        'knn__leaf_size': [20, 30, 40],

    

        'kbest__k': [3, 5, 7, 9, 11, 13]

        }
grid_search_knn = GridSearchCV(knn_pipe, param_grid=knn_params, scoring='roc_auc', n_jobs=-1)

grid_search_knn.fit(X_train, y_train)



grid_search_knn.best_params_
grid_search_knn.best_score_
ac_knn_cv = grid_search_knn.best_estimator_.score(X_test,y_test)

print('KNN Classifier Accuracy CV:', ac_knn_cv)
cnf_matrix = confusion_matrix(y_test, grid_search_knn.best_estimator_.predict(X_test))



plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - KNN + Grid Search')

a = plt.gcf()

a.set_size_inches(4,3)

plt.show()
from sklearn.ensemble import RandomForestClassifier



rfc_pipe = Pipeline([

    ('kbest', SelectKBest(f_classif)),

    ('scaler', StandardScaler()),

    ('rfc', RandomForestClassifier())

])



rfc_pipe.set_params(

    rfc__random_state=random_state,

    kbest__k=5

)



rfc_pipe.fit(X_train, y_train)



ac_rfc = rfc_pipe.score(X_test, y_test)

print('RandomForest Classifier Accuracy:', ac_rfc)
rfc_params = {

        'rfc__min_samples_split': [2, 5, 10, 15, 20],

        'rfc__max_depth': [5, 10, 15, 20, 25, None],

        'rfc__n_estimators': [100, 250, 500, 750, 1000],

        'rfc__bootstrap': [True, False],

        'rfc__min_samples_leaf': [1, 2, 5, 6, 10],

    

        'kbest__k': [3, 5, 7, 9, 11, 13]

        }



params_comb_all = int(5 * 6 * 5 * 2 * 5  * 6)

param_comb = int(params_comb_all / 100)



print('GridSearchCV parameter combinations: ' + str(params_comb_all) + '\n' + 'RandomizedSearchCV parameter combinations: ' + str(param_comb))
rnd_search_rfc = RandomizedSearchCV(rfc_pipe, param_distributions=rfc_params, n_iter=param_comb, scoring='roc_auc', n_jobs=-1, random_state=random_state )

rnd_search_rfc.fit(X_train, y_train)



rnd_search_rfc.best_params_
rnd_search_rfc.best_score_
ac_rfc_cv = rnd_search_rfc.best_estimator_.score(X_test,y_test)

print('RandomForest Classifier Accuracy CV:', ac_rfc_cv)
cnf_matrix = confusion_matrix(y_test, rnd_search_rfc.best_estimator_.predict(X_test))



plt.figure()

class_names = [0,1]

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix - RandomForestClassifier + Randomized Grid Search')

a = plt.gcf()

a.set_size_inches(4,3)

plt.show()
ac_df = pd.DataFrame()

ac_df['Dummy'] = [ac_sgdcd, '-']

ac_df['Linear'] = [ac_sgdc, ac_sgdc_cv]

ac_df['Decision Tree'] = [ac_dtc, ac_dtc_cv]

ac_df['KNN'] = [ac_knn, ac_knn_cv]

ac_df['Random Forest'] = [ac_rfc, ac_rfc_cv]

ac_df



#0 ist ohne CV, 1 ist mit CV
from xgboost import XGBClassifier



xgc_pipe = Pipeline([

    ('kbest', SelectKBest(f_classif)),

    ('scaler', StandardScaler()),

    ('xgc', XGBClassifier())

])



xgc_pipe.set_params(

    kbest__k=5

)



xgc_pipe.fit(X_train, y_train)



xgc_pipe.score(X_test, y_test)
xgc_params = {

        'xgc__learning_rate': [.0001, .001, .01, .1],

        'xgc__max_depth': [5, 10, 15, 20, 25],

        'xgc__n_estimators': [250, 500, 750, 1000],

    

        'kbest__k': [3, 5, 7, 9, 11, 13]

        }



params_comb_all = int(4 * 5 * 4 * 6)

param_comb = int(params_comb_all / 5)



print('GridSearchCV parameter combinations: ' + str(params_comb_all) + '\n' + 'RandomizedSearchCV parameter combinations: ' + str(param_comb))
rnd_search_xgc = RandomizedSearchCV(xgc_pipe, param_distributions=xgc_params, n_iter=param_comb, scoring='roc_auc', n_jobs=-1, random_state=random_state )

rnd_search_xgc.fit(X_train, y_train)



rnd_search_xgc.best_params_
rnd_search_xgc.best_score_
rnd_search_xgc.best_estimator_.score(X_test,y_test)
y_pred = rnd_search_xgc.best_estimator_.predict(X_test)

confusion_matrix(y_test, y_pred)
grid_search_xgc = GridSearchCV(xgc_pipe, param_grid=xgc_params, scoring='roc_auc', n_jobs=-1)

grid_search_xgc.fit(X_train, y_train)



grid_search_xgc.best_params_
grid_search_xgc.best_score_
grid_search_xgc.best_estimator_.score(X_test,y_test)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
def keras_model():

    model = Sequential()

    model.add(Dense(22, input_dim=5, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
estimators = []

estimators.append(('scaler', StandardScaler()))

estimators.append(('keras_clf', KerasClassifier(build_fn=keras_model, epochs=200, batch_size=5, verbose=0)))



pipeline = Pipeline(estimators)
results = cross_val_score(pipeline, X[['cp', 'thalach', 'exang', 'oldpeak', 'ca']], y, cv=kfold)

print("Accuracy der verschiedenen Folds: ", results)

print("Accuracy: mean %.2f%% (std %.2f%%)" % (results.mean()*100, results.std()*100))
pipeline.fit(X_train[['cp', 'thalach', 'exang', 'oldpeak', 'ca']], y_train)



y_pred = pipeline.predict(X_test[['cp', 'thalach', 'exang', 'oldpeak', 'ca']])
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)