import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
import multiprocessing

n_jobs = multiprocessing.cpu_count()-1
n_jobs
import os

FILEDIR = '../input/santander-customer-transaction-prediction-dataset/'

os.listdir(FILEDIR)
# Load data

df = pd.read_csv(FILEDIR + 'train.csv',

                 header=0)



df.head()
df.shape
correlation = df.iloc[:, 1:].corr()

correlation
f = plt.figure(figsize=(20, 18))

plt.matshow(correlation, fignum=f.number)

# plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)

# plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()

# cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16)

plt.show()
fig = plt.figure(figsize=(15,5))

corr = correlation.iloc[1:,1:].values.reshape(-1,)  # drop target

corr = corr[corr != 1]

sns.distplot(corr)

plt.xlim((-0.02, 0.02))

plt.show()
# distribution of var_0

fig = plt.figure(figsize=(15,5))

var_0_0 = df.loc[df['target']==0,df.columns[2]].values.reshape(-1,)

var_0_1 = df.loc[df['target']==1,df.columns[2]].values.reshape(-1,)

sns.distplot(var_0_0)

sns.distplot(var_0_1)

plt.legend([0, 1])

plt.show()
def getplot_var(df, r=4, c=4):

    fig, axs = plt.subplots(r, c, figsize=(c*4, r*3))

    cnt = 0

    for i in range(r):

        for j in range(c):

            sns.distplot(df.loc[df['target']==0,df.columns[cnt+2]].values.reshape(-1,), ax=axs[i,j], axlabel=str(df.columns[cnt+2]))

            sns.distplot(df.loc[df['target']==1,df.columns[cnt+2]].values.reshape(-1,), ax=axs[i,j], axlabel=str(df.columns[cnt+2]))

            axs[i,j].legend([0, 1])

            cnt += 1

    plt.tight_layout()

    return
getplot_var(df, 4, 4)
def missing_value_checker(df):

    """

    The missing value checker



    Parameters

    ----------

    df : dataframe

    

    Returns

    ----------

    The variables with missing value and their proportion of missing value

    """

    

    variable_proportion = [[variable, df[variable].isna().sum() / df.shape[0]] 

                           for variable in df.columns 

                           if df[variable].isna().sum() > 0]



    print('%-30s' % 'Variable with missing values', 'Proportion of missing values')

    for variable, proportion in sorted(variable_proportion, key=lambda x : x[1]):

        print('%-30s' % variable, proportion)

        

    return variable_proportion
variable_proportion = missing_value_checker(df)
def categorical_feature_checker(df, target, dtype):

    """

    The categorical feature checker



    Parameters

    ----------

    df : dataframe

    target : the target

    dtype : the type of the feature

    

    Returns

    ----------

    The categorical features and their number of unique value

    """

    

    feature_number = [[feature, df[feature].nunique()] 

                      for feature in df.columns 

                      if feature != target and df[feature].dtype.name == dtype]

    

    print('%-30s' % 'Categorical feature', 'Number of unique value')

    for feature, number in sorted(feature_number, key=lambda x : x[1]):

        print('%-30s' % feature, number)

    

    return feature_number
feature_number = categorical_feature_checker(df, 'target', 'object')
X_raw = df.drop(['ID_code', 'target'], axis=1)

y_raw = df['target']
X, y = X_raw.copy(), y_raw.copy()
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', '^']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
from mpl_toolkits.mplot3d import Axes3D



def plot_3d_space(X, y):

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d')



    for m, c, label in [('o', '#1F77B4', 0), ('^', '#FF7F0E', 1)]:

        xs = X[y==label].T[0]

        ys = X[y==label].T[1]

        zs = y[y==label]

        ax.scatter(xs, ys, zs, c=c, marker=m)



    ax.set_xlabel('PC1 Label')

    ax.set_ylabel('PC2 Label')

    ax.set_zlabel('Target')

    # rotate the axes and update

#     for angle in range(0, 360):

#         ax.view_init(30, angle)

#         plt.draw()

#         plt.pause(.001)

    plt.show()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

# X_test = ss.transform(X_test)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca_X = pca.fit_transform(X)
plot_2d_space(pca_X, y)
plot_3d_space(pca_X, y)
# %matplotlib qt  #interactive plot

# plot_3d_space(pca_X, y_test)
from sklearn.model_selection import train_test_split

X, X_test, y, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw)
y_raw.value_counts()
# bar chart

df.target.value_counts().plot(kind='bar', title='target counts')

plt.show()
from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



# RandomOverSampler (with random_state=0)

# ros = RandomOverSampler(random_state=0)

# X, y = ros.fit_sample(X, y)





# adjust the weight of undersampling, 2:1 (for target 0,1) is a little bit better in practice.

num_class1 = len(y[y==1])

rus = RandomUnderSampler(random_state=0, ratio={0: int(1*num_class1), 1: num_class1})



X, y = rus.fit_sample(X, y)

pd.DataFrame(data=y, columns=['target'])['target'].value_counts()
pd.DataFrame(data=y, columns=['target']).target.value_counts().plot(kind='bar', title='target counts with balanced ')

plt.show()
pca_X = pca.fit_transform(X)
plot_2d_space(pca_X, y)
plot_3d_space(pca_X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
clfs = {'rf': RandomForestClassifier(random_state=0),

        'lr': LogisticRegression(random_state=0),

        'mlp': MLPClassifier(random_state=0),

        'dt': DecisionTreeClassifier(random_state=0),

        'rf': RandomForestClassifier(random_state=0),

        'xgb': XGBClassifier(seed=0),

#         'svc': SVC(random_state=0),

#         'knn': KNeighborsClassifier(),

        'gnb': GaussianNB()}
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



pipe_clfs = {}



for name, clf in clfs.items():

    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),

                                ('clf', clf)])
# For GridSearchCV



param_grids = {}

# ------

C_range = [10 ** i for i in range(-5, 1)]

param_grid = [{'clf__multi_class': ['ovr'],

               'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

               'clf__C': C_range},



              {'clf__multi_class': ['multinomial'],

               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],

               'clf__C': C_range}]



param_grids['lr'] = param_grid

# ------

param_grid = [{'clf__hidden_layer_sizes': [(10,), (10, 10, 10), (100,)],

               'clf__activation': ['tanh', 'relu']}]



param_grids['mlp'] = param_grid

# ------

param_grid = [{'clf__min_samples_split': [2, 5, 10],

               'clf__min_samples_leaf': [20, 60, 150]}]



param_grids['dt'] = param_grid

# ------

param_grid = [{'clf__n_estimators': [10, 100],

               'clf__min_samples_split': [5, 10, 30],

               'clf__min_samples_leaf': [2, 5, 10]}]



param_grids['rf'] = param_grid

# ------

param_grid = [{'clf__eta': [10 ** i for i in range(-6, -1)],

               'clf__gamma': [0, 10, 100],

               'clf__lambda': [10 ** i for i in range(-6, -1)]}]



param_grids['xgb'] = param_grid

# ------

param_grid = [{'clf__C': [10 ** i for i in range(-4, 5)],

               'clf__gamma': ['auto', 'scale']}]



param_grids['svc'] = param_grid

# ------

param_grid = [{'clf__n_neighbors': list(range(1, 11))}]



param_grids['knn'] = param_grid

# ------

param_grid = [{'clf__var_smoothing': [10 ** i for i in range(-12, -4)]}]



param_grids['gnb'] = param_grid
# best parameter (selected after GSCV)



param_grids = {}

# ------

C_range = [0.001]

param_grid = [{'clf__multi_class': ['ovr'],

               'clf__solver': ['sag'],

               'clf__C': C_range}]



param_grids['lr'] = param_grid

# ------

param_grid = [{'clf__hidden_layer_sizes': [(10,)],

               'clf__activation': ['relu']}]



param_grids['mlp'] = param_grid

# ------

param_grid = [{'clf__min_samples_split': [2],

               'clf__min_samples_leaf': [150]}]



param_grids['dt'] = param_grid

# ------

param_grid = [{'clf__n_estimators': [200],

               'clf__min_samples_split': [30],

               'clf__min_samples_leaf': [2]}]



param_grids['rf'] = param_grid

# ------

param_grid = [{'clf__eta': [1e-6],

               'clf__gamma': [10],

               'clf__lambda': [1e-6]}]



param_grids['xgb'] = param_grid

# ------

param_grid = [{'clf__C': [0.1],

               'clf__gamma': ['auto']}]



param_grids['svc'] = param_grid

# ------

param_grid = [{'clf__n_neighbors': 10}]



param_grids['knn'] = param_grid

# ------

param_grid = [{'clf__var_smoothing': [1e-12]}]



param_grids['gnb'] = param_grid

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



# ------

gsRF = GridSearchCV(pipe_clfs['rf'], param_grid=param_grids['rf'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsRF.fit(X, y)

RF_best = gsRF.best_estimator_

print(gsRF.best_score_, gsRF.best_estimator_)

# ------

gsXGB = GridSearchCV(pipe_clfs['xgb'], param_grid=param_grids['xgb'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsXGB.fit(X, y)

XGB_best = gsXGB.best_estimator_

print(gsXGB.best_score_, gsXGB.best_estimator_)

# ------

gsLR = GridSearchCV(pipe_clfs['lr'], param_grid=param_grids['lr'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsLR.fit(X, y)

LR_best = gsLR.best_estimator_

print(gsLR.best_score_, gsLR.best_estimator_)

# ------

gsGNB = GridSearchCV(pipe_clfs['gnb'], param_grid=param_grids['gnb'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsGNB.fit(X, y)

GNB_best = gsGNB.best_estimator_

print(gsGNB.best_score_, gsGNB.best_estimator_)

# ------

# gsSVC = GridSearchCV(pipe_clfs['svc'], param_grid=param_grids['svc'],

#                     cv=StratifiedKFold(n_splits=4, random_state=0),

#                     scoring="roc_auc", n_jobs=-1, verbose=1)

#

# gsSVC.fit(X, y)

# SVC_best = gsSVC.best_estimator_

# print(gsSVC.best_score_, gsSVC.best_estimator_)

# ------

# gsKNN = GridSearchCV(pipe_clfs['knn'], param_grid=param_grids['knn'],

#                     cv=StratifiedKFold(n_splits=4, random_state=0),

#                     scoring="roc_auc", n_jobs=-1, verbose=1)

#

# gsKNN.fit(X, y)

# KNN_best = gsKNN.best_estimator_

# print(gsKNN.best_score_, gsKNN.best_estimator_)

# ------

gsMLP = GridSearchCV(pipe_clfs['mlp'], param_grid=param_grids['mlp'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsMLP.fit(X, y)

MLP_best = gsMLP.best_estimator_

print(gsMLP.best_score_, gsMLP.best_estimator_)

# ------

gsDT = GridSearchCV(pipe_clfs['dt'], param_grid=param_grids['dt'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsDT.fit(X, y)

DT_best = gsDT.best_estimator_

print(gsDT.best_score_, gsDT.best_estimator_)

# ------
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



# The list of [best_score_, best_params_, best_estimator_]

best_score_param_estimators = []



# For each classifier

for name in pipe_clfs.keys():

    # Implement me

    # GridSearchCV

    gs = GridSearchCV(estimator=pipe_clfs[name],

                      param_grid=param_grids[name],

                      scoring='roc_auc',

                      n_jobs=-1,

                      verbose=1,

                      iid=False,

                      cv=StratifiedKFold(n_splits=10,

                                         random_state=0),

                      return_train_score=True)

    # Implement me

    # Fit the pipeline

    gs = gs.fit(X, y)



    # Update best_score_param_estimators

    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    

    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'

    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    

    # Get the important columns in cv_results

    important_columns = ['rank_test_score',

                         'mean_test_score', 

                         'std_test_score', 

                         'mean_train_score', 

                         'std_train_score',

                         'mean_fit_time', 

                         'std_fit_time',                        

                         'mean_score_time', 

                         'std_score_time']

    

    # Move the important columns ahead

    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]



    # Write cv_results file

#     cv_results.to_csv(path_or_buf=name + '_cv_results.csv', index=False)
# Sort best_score_param_estimators in descending order of the best_score_

best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)



# Print best_score_param_estimators

for rank in range(len(best_score_param_estimators)):

    best_score, best_params, best_estimator = best_score_param_estimators[rank]



    print('Top', str(rank + 1))    

    print('%-15s' % 'best_score:', best_score)

    print('%-15s' % 'best_estimator:'.format(20), type(best_estimator.named_steps['clf']))

    print('%-15s' % 'best_params:'.format(20), best_params, end='\n\n')
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



# Create the pipeline with StandardScaler and RandomForestClassifier

pipe_rf = Pipeline([('StandardScaler', StandardScaler()),

                    ('RandomForestClassifier', RandomForestClassifier(class_weight=None, criterion='gini', max_features='auto',

                            min_samples_leaf=2,

                            min_samples_split=30,

                            min_weight_fraction_leaf=0.0,

                            n_estimators=100, n_jobs=-1, random_state=0,

                            verbose=1, warm_start=False))])



pipe_rf.fit(X, y)
roc_auc_score(y_test, pipe_rf.predict_proba(X_test).T[1])
import matplotlib.pyplot as plt



feature_value_names = df.columns[2:]

# Convert the importances into one-dimensional 1darray with corresponding df column names as axis labels

f_importances = pd.Series(pipe_rf.named_steps['RandomForestClassifier'].feature_importances_, feature_value_names)



# Sort the array in descending order of the importances

f_importances = f_importances.sort_values(ascending=False)



# Draw the bar Plot from f_importances 

f_importances[:20].plot(x='Features', y='Importance', kind='bar', figsize=(8,5), rot=45, fontsize=14)



# Show the plot

plt.tight_layout()

plt.show()
# Select out the more important features

X_new = X_raw.loc[:, f_importances[f_importances>0.005].index]
X_new.head()
f_importances[f_importances>0.005].sum()
X_new, X_test, y_new, y_test = train_test_split(X_new, y_raw, test_size=0.2, random_state=0, stratify=y_raw)
# Here just use a simple demo

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

param_grid = [{'clf__min_samples_split': [2],

               'clf__min_samples_leaf': [5000]}]



param_grids['dt'] = param_grid



gsDT = GridSearchCV(pipe_clfs['dt'], param_grid=param_grids['dt'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsDT.fit(X, y)

DT_best = gsDT.best_estimator_

print(gsDT.best_score_, gsDT.best_estimator_)
# # Draw the tree with original features

# from pydotplus import graph_from_dot_data

# from sklearn.tree import export_graphviz

# from IPython.display import Image



# feature_value_names = df.columns[2:]

# dot_data = export_graphviz(DT_best.named_steps['clf'],

#                            filled=True, 

#                            rounded=True,

#                            class_names=['0', 

#                                         '1'],

#                            feature_names=feature_value_names,

#                            out_file=None) 



# graph = graph_from_dot_data(dot_data) 



# Image(graph.create_png())
param_grid = [{'clf__min_samples_split': [2],

               'clf__min_samples_leaf': [20000]}]



param_grids['dt'] = param_grid



gsDT = GridSearchCV(pipe_clfs['dt'], param_grid=param_grids['dt'],

                    cv=StratifiedKFold(n_splits=4, random_state=0),

                    scoring="roc_auc", n_jobs=-1, verbose=1)



gsDT.fit(X_new, y_new)

DT_best = gsDT.best_estimator_

print(gsDT.best_score_, gsDT.best_estimator_)
# # Draw the tree with selected features

# feature_value_names = f_importances[f_importances>0.005].index

# dot_data = export_graphviz(DT_best.named_steps['clf'],

#                            filled=True, 

#                            rounded=True,

#                            class_names=['0', 

#                                         '1'],

#                            feature_names=feature_value_names,

#                            out_file=None) 



# graph = graph_from_dot_data(dot_data) 



# Image(graph.create_png()) 
X, X_test, y, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=0, stratify=y_raw)

# X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=0, stratify=y_test)
# Ensemble - Stacking - Logistic Regression

# something wrong with the sklearn version. Cannot import StackingClassifier

from sklearn.ensemble import StackingClassifier

stackingC = StackingClassifier(estimators=[('rf', RF_best), ('lr', LR_best), ('dt', DT_best), #('knn', KNN_best), ('svc', SVC_best),   ##KNN makes model so large

('mlp', MLP_best), ('xgb', XGB_best), ('gnb', GNB_best)], final_estimator=LogisticRegression(), n_jobs=-1, verbose=1)

stackingC.fit(X, y)

y_pred_stacking = stackingC.predict_proba(X_test).T[1]

# print(confusion_matrix(y_test, y_pred_stacking))

print(roc_auc_score(y_test, y_pred_stacking))
# Ensemble - Stacking - Voting

from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators=[('rf', RF_best), ('lr', LR_best), ('dt', DT_best), #('knn', KNN_best), ('svc', SVC_best),   ##KNN makes model so large

('mlp', MLP_best), ('xgb', XGB_best), ('gnb', GNB_best)], voting='soft', n_jobs=-1)

votingC.fit(X, y)

y_pred_stacking = votingC.predict_proba(X_test).T[1]

# print(confusion_matrix(y_test, y_pred_stacking))

print(roc_auc_score(y_test, y_pred_stacking))
# FOR SUBMISSION

df_test = pd.read_csv(FILEDIR + 'test.csv',

                 header=0)

y_pred_voting = votingC.predict_proba(df_test.iloc[:,1:]).T[1]

df_test['target'] = y_pred_voting

df_sub = df_test.loc[:,['ID_code', 'target']]

df_sub.to_csv('submission.csv', index=False)