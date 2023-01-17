!pip install chart-studio
# load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import warnings 

warnings.filterwarnings('ignore')

# %matplotlib inline
# load dataset

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_data = train_data.sample(n=1000)

test_data = test_data.sample(n=1000)
# print the shape of datasets

print('Shape of train dataset : ', train_data.shape)

print('Shape of test dataset : ', test_data.shape)



# sample entries from the train dataset

train_data.head()
# Now time to handle the missing values

print('Missing values in the train dataset : ', train_data[train_data.isnull()].count().sum())

print('Missing values in the test dataset : ', test_data[test_data.isnull()].count().sum())
# Is the 'target' variable biased?

train_data['target'].hist()
# the counts

majority_class_count, minority_class_count = train_data['target'].value_counts()

print('The majority class count :  ', majority_class_count)

print('The minority class count :  ', minority_class_count)

# majority and minority class dataframes

train_data_majority_class = train_data[train_data['target'] == 0]

train_data_minority_class = train_data[train_data['target'] == 1]



maj_class_percent = round((majority_class_count/minority_class_count)/len(train_data)*100)

min_class_percent = round((minority_class_count/minority_class_count)/len(train_data)*100)



print('Majority class (%): ', maj_class_percent)

print('Minority class (%): ', minority_class_count)
# let's introduce a new plot function for visualizing the impacts

def plot2DClusters(X,y,label='Classes'):

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='Upper right')

    plt.show()





    

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly_express as px



def plot3DClusters(dataset):

    fig = px.scatter_3d(dataset, x='dim-1', y='dim-2', z='dim-3', color='target', opacity=0.8)

    iplot(fig, filename='jupyter-parametric_plot')
# split the dataset into labels and IVs

X = train_data.drop(['ID_code', 'target'], axis=1)

y = train_data['target']



temp_X_holder = X

temp_y_holder = y





# It is not practical to visualize the classes or clusters in the dataset using 2DPlot (as dimensions > 3)

# So we will perform the PCA to reduce the dimension

from sklearn.decomposition import PCA



pca = PCA(n_components = 3)

X = pca.fit_transform(temp_X_holder)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = temp_y_holder.values

plot3DClusters(test)
from sklearn.decomposition import PCA



pca = PCA(n_components = 2)

X = pca.fit_transform(temp_X_holder)



plot2DClusters(X,temp_y_holder.values,label='Imbalanced Dataset (PCA)')
new_train_data_majority_class = train_data_majority_class.sample(minority_class_count, replace=True)



# create new dataset

downsampled_data = pd.concat([train_data_minority_class, new_train_data_majority_class], axis=0)
# check results

print(downsampled_data['target'].value_counts())

downsampled_data['target'].hist()
# split the dataset into labels and IVs

y = downsampled_data['target']

X = downsampled_data.drop(['ID_code', 'target'], axis=1)



temp_X_holder = X

temp_y_holder = y





# It is not practical to visualize the classes or clusters in the dataset using 2DPlot (as dimensions > 3)

# So we will perform the PCA to reduce the dimension

from sklearn.decomposition import PCA



pca = PCA(n_components = 3)

X = pca.fit_transform(temp_X_holder)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = temp_y_holder.values

plot3DClusters(test)
from sklearn.decomposition import PCA



pca = PCA(n_components = 2)

X = pca.fit_transform(temp_X_holder)



plot2DClusters(X,temp_y_holder.values,label='Balanced Dataset (PCA)')
new_train_data_minority_class = train_data_minority_class.sample(majority_class_count, replace=True) \



# concatenate the dataframes to create the new one

upsampled_data = pd.concat([new_train_data_minority_class, train_data_majority_class], axis=0)



# check the results

print(upsampled_data['target'].value_counts())

upsampled_data['target'].hist()
# split the dataset into labels and IVs

y = upsampled_data['target']

X = upsampled_data.drop(['ID_code', 'target'], axis=1)



temp_X_holder = X

temp_y_holder = y





# It is not practical to visualize the classes or clusters in the dataset using 2DPlot (as dimensions > 3)

# So we will perform the PCA to reduce the dimension

from sklearn.decomposition import PCA



pca = PCA(n_components = 3)

X = pca.fit_transform(temp_X_holder)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = temp_y_holder.values

plot3DClusters(test)
from sklearn.decomposition import PCA



pca = PCA(n_components = 2)

X = pca.fit_transform(temp_X_holder)



plot2DClusters(X,temp_y_holder.values,label='Balanced Dataset (PCA)')
from imblearn.under_sampling import TomekLinks



imb_tomek = TomekLinks(return_indices = True, ratio = 'majority')



X_imb_tomek, y_imb_tomek, Id_imb_tomek = imb_tomek.fit_sample(temp_X_holder, temp_y_holder)



print('Number of data points deleted : ', len(Id_imb_tomek))
# let's check the results

X_imb_tomek = pd.DataFrame(X_imb_tomek)

y_imb_tomek = pd.DataFrame(y_imb_tomek)



y_imb_tomek.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_tomek)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_tomek.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_tomek)



plot2DClusters(X,y_imb_tomek[0],label='Balanced Dataset (PCA)')
from imblearn.under_sampling import ClusterCentroids



# imb_cc = ClusterCentroids(ratio={0:100}) # we want to save 100 points from each class

imb_cc = ClusterCentroids()

X_imb_cc, y_imb_cc = imb_cc.fit_sample(temp_X_holder, temp_y_holder)
# let's check the results

X_imb_cc = pd.DataFrame(X_imb_cc)

y_imb_cc = pd.DataFrame(y_imb_cc)



y_imb_cc.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_cc)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_cc.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_cc)



plot2DClusters(X, y_imb_cc[0],label='Down sampling with Cluster Centroids')
from imblearn.under_sampling import NearMiss



imb_nn = NearMiss() # we want to save 100 points from each class



X_imb_nn, y_imb_nn = imb_nn.fit_sample(temp_X_holder, temp_y_holder)



# let's check the results

X_imb_nn = pd.DataFrame(X_imb_nn)

y_imb_nn = pd.DataFrame(y_imb_nn)



y_imb_nn.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_nn)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_nn.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_nn)



plot2DClusters(X, y_imb_nn[0],label='Down sampling with Cluster Centroids')
# SMOTE

from imblearn.over_sampling import SMOTE



imb_smote = SMOTE(ratio='minority')



X_imb_smote, y_imb_smote = imb_smote.fit_sample(temp_X_holder, temp_y_holder)
# let's check the results

X_imb_smote = pd.DataFrame(X_imb_smote)

y_imb_smote = pd.DataFrame(y_imb_smote)



y_imb_smote.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_smote)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_smote.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_smote)



plot2DClusters(X, y_imb_smote[0],label='Oversampling with SMOTE')
# ADASYN

from imblearn.over_sampling import ADASYN



imb_adasyn = ADASYN(ratio='minority')



X_imb_adasyn, y_imb_adasyn = imb_adasyn.fit_sample(temp_X_holder, temp_y_holder)
# let's check the results

X_imb_adasyn = pd.DataFrame(X_imb_adasyn)

y_imb_adasyn = pd.DataFrame(y_imb_adasyn)



y_imb_adasyn.hist()

pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_adasyn)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_adasyn.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_adasyn)



plot2DClusters(X, y_imb_adasyn[0],label='Oversampling with ADASYN')
from imblearn.combine import SMOTETomek



imb_smotetomek = SMOTETomek(ratio='auto')



X_imb_smotetomek, y_imb_smotetomek = imb_smotetomek.fit_sample(temp_X_holder, temp_y_holder)



# let's check the results

X_imb_smotetomek = pd.DataFrame(X_imb_smotetomek)

y_imb_smotetomek = pd.DataFrame(y_imb_smotetomek)



y_imb_smotetomek.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_smotetomek)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_smotetomek.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_smotetomek)



plot2DClusters(X, y_imb_smotetomek[0],label='Balanced Dataset (PCA)')
from imblearn.combine import SMOTEENN



imb_smoteenn = SMOTEENN(random_state=0)



X_imb_smoteenn, y_imb_smoteenn = imb_smoteenn.fit_sample(temp_X_holder, temp_y_holder)



# let's check the results

X_imb_smoteenn = pd.DataFrame(X_imb_smoteenn)

y_imb_smoteenn = pd.DataFrame(y_imb_smoteenn)



y_imb_smoteenn.hist()
pca = PCA(n_components = 3)

X = pca.fit_transform(X_imb_smoteenn)



test = pd.DataFrame(columns=['dim-1', 'dim-2', 'dim-3'], data=X)

test['target'] = y_imb_smoteenn.values

plot3DClusters(test)
pca = PCA(n_components = 2)

X = pca.fit_transform(X_imb_smoteenn)



plot2DClusters(X, y_imb_smoteenn[0],label='Balanced Dataset (PCA)')
from sklearn.model_selection import train_test_split



# load dataset

dataset = pd.read_csv('../input/train.csv')

dataset = dataset.sample(n=500)



y = dataset['target']

X = dataset.drop(['ID_code', 'target'], axis=1)



# time to split into train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

print('Shape of Train data : ', X_train.shape)

print('Shape of Test data : ', X_test.shape)
# helper methods for the dataset preperation and benchmarking

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



def resample(resampler, X, y):

    print("Resamping with {}".format(resampler.__class__.__name__))

    X_resampled, y_resampled = resampler.fit_sample(X, y)

    return resampler.__class__.__name__, pd.DataFrame(X_resampled), pd.DataFrame(y_resampled)



def simulation(resampling_type, X, y):

    lr = LogisticRegression(penalty='l1')

    parameter_grid = {'C':[0.01, 0.1, 1, 10]}

    gs = GridSearchCV(estimator=lr, param_grid=parameter_grid, scoring='accuracy', cv=3, verbose=2) # cv=5

    gs = gs.fit(X.values, y.values.ravel())

    return resampling_type, gs.best_score_, gs.best_params_['C']

%%time

# we will use the random under and over sampling methods of imblearn instead that of pandas

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler



resampled_datasets = []

resampled_datasets.append(("base dataset", X_train, y_train))

resampled_datasets.append(resample(SMOTE(n_jobs=-1),X_train,y_train))

resampled_datasets.append(resample(RandomOverSampler(),X_train,y_train))

resampled_datasets.append(resample(ClusterCentroids(n_jobs=-1),X_train,y_train))

resampled_datasets.append(resample(NearMiss(n_jobs=-1),X_train,y_train))

resampled_datasets.append(resample(RandomUnderSampler(),X_train,y_train))

resampled_datasets.append(resample(SMOTEENN(),X_train,y_train))

resampled_datasets.append(resample(SMOTETomek(),X_train,y_train))
%%time

benchmark_scores = []

for resampling_type, X, y in resampled_datasets:

    print('______________________________________________________________')

    print('{}'.format(resampling_type))

    benchmark_scores.append(simulation(resampling_type, X, y))

    print('______________________________________________________________')
benchmark_scores_df = pd.DataFrame(columns = ['Methods', 'Accuracy', 'Parameter'], data = benchmark_scores)

benchmark_scores_df
%%time



from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, auc,roc_auc_score,roc_curve, precision_recall_curve



scores = []

# train models based on benchmark params

for sampling_type,score,param in benchmark_scores:

    print("Training on {}".format(sampling_type))

    lr = LogisticRegression(penalty = 'l1',C=param)

    for s_type,X,y in resampled_datasets:

        if s_type == sampling_type:

            lr.fit(X.values,y.values.ravel())

            pred_test = lr.predict(X_test.values)

            pred_test_probs = lr.predict_proba(X_test.values)

            probs = lr.decision_function(X_test.values)

            fpr, tpr, thresholds = roc_curve(y_test.values.ravel(),pred_test)

            p,r,t = precision_recall_curve(y_test.values.ravel(),probs)

            scores.append((sampling_type,

                           f1_score(y_test.values.ravel(),pred_test),

                           precision_score(y_test.values.ravel(),pred_test),

                           recall_score(y_test.values.ravel(),pred_test),

                           accuracy_score(y_test.values.ravel(),pred_test),

                           auc(fpr, tpr),

                           auc(p,r,reorder=True),

                           confusion_matrix(y_test.values.ravel(),pred_test)))
sampling_results = pd.DataFrame(scores,columns=['Sampling Type','f1','precision','recall','accuracy','auc_roc','auc_pr','confusion_matrix'])

sampling_results
# let's visulize the confusion metrices



f, axes = plt.subplots(2, 4, figsize=(15, 5), sharex=True)

sns.despine(left=True)



sns.heatmap(sampling_results['confusion_matrix'][0], annot=True, fmt='g', ax=axes[0, 0])

sns.heatmap(sampling_results['confusion_matrix'][1], annot=True, fmt='g', ax=axes[0, 1])

sns.heatmap(sampling_results['confusion_matrix'][2], annot=True, fmt='g', ax=axes[0, 2])

sns.heatmap(sampling_results['confusion_matrix'][3], annot=True, fmt='g', ax=axes[0, 3])

sns.heatmap(sampling_results['confusion_matrix'][4], annot=True, fmt='g', ax=axes[1, 0])

sns.heatmap(sampling_results['confusion_matrix'][5], annot=True, fmt='g', ax=axes[1, 1])

sns.heatmap(sampling_results['confusion_matrix'][6], annot=True, fmt='g', ax=axes[1, 2])

sns.heatmap(sampling_results['confusion_matrix'][7], annot=True, fmt='g', ax=axes[1, 3])