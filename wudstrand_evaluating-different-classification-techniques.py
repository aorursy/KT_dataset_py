# General Imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



# Model Imports

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import tree

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer

import scikitplot as skplt
# This class is used for graphing box and wisker plots

class visualizer:

    def __init__(self, df, colname, fields, dimensions, figsize=None):

        self.df = df

        self.colname = colname

        self.fields = fields

        self.figsz = figsize

        self.dim = dimensions

        

    def data(self, field):

        vals = list(set(self.df[self.colname]))

        return [self.df[self.df[self.colname] == i][field].values for i in vals]

          

    def plot(self, ax, field):

        ax.set_title(field)

        ax.set_xlabel("Classification")

        sns.boxplot(ax=ax, data = self.data(field), showfliers=False)

    

    def graph(self):

        fig, axes = plt.subplots(nrows=self.dim[0], ncols=self.dim[1], figsize=self.figsz)

        for ax, field in zip(axes.reshape(-1), self.fields):

            self.plot(ax, field)

                       
# Reading in the kepler exoplanet data from csv

kepler_df = pd.read_csv("../input/cumulative.csv")



print("Dimensions: ", kepler_df.shape)

kepler_df.head()
agg_fxns = {

    'koi_disposition': ['count'],

    'koi_score': ['mean']

}



kepler_df.groupby(['koi_disposition']).agg(agg_fxns)
agg_fxns = {

    'koi_pdisposition': ['count'],

    'koi_score': ['mean']

}



kepler_df.groupby(['koi_pdisposition']).agg(agg_fxns)
# Replace the enumerated values in the kepler_df

kepler_df = kepler_df.replace({'CONFIRMED': 0, 'CANDIDATE': 1, 'FALSE POSITIVE': 2})



# Clean the Data Frame by removing all labels, and error values

kepler_df = kepler_df.drop(['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname', 

                     'koi_period_err1', 'koi_period_err2', 

                     'koi_time0bk_err1', 'koi_time0bk_err2', 

                     'koi_impact_err1', 'koi_impact_err2',

                     'koi_depth_err1', 'koi_depth_err2', 

                     'koi_prad_err1', 'koi_prad_err2', 

                     'koi_insol_err1', 'koi_insol_err2',

                     'koi_steff_err1', 'koi_steff_err2', 

                     'koi_slogg_err1', 'koi_slogg_err2', 

                     'koi_srad_err1', 'koi_srad_err2', 

                     'koi_duration_err1', 'koi_duration_err2', 

                     'koi_teq_err1', 'koi_teq_err2'], axis=1)



# Dealing with null data fields

# kepler_df.fillna(0, inplace=True)

kepler_df.fillna(kepler_df.mean(), inplace=True)
# Building a correlation matrix

corr = kepler_df.corr()

f, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
agg_fxns = {

    'koi_score': ['mean'],

    'koi_fpflag_nt': ['mean'],

    'koi_fpflag_ss': ['mean'],

    'koi_fpflag_co': ['mean'],

    'koi_fpflag_ec': ['mean'],

    'koi_depth': ['mean'],

    'koi_teq': ['mean'],

    'koi_model_snr': ['mean']

}

kepler_df.groupby('koi_disposition').agg(agg_fxns).transpose()
fields = ['koi_score', 'koi_depth', 'koi_teq', 'koi_model_snr']

id = 'koi_disposition'



v = visualizer(kepler_df, 'koi_disposition', fields, (1,4), (15,5))

v.graph()
kepler_df.groupby('koi_pdisposition').agg(agg_fxns).transpose()
fields = ['koi_score', 'koi_depth', 'koi_teq', 'koi_model_snr']

id = 'koi_pdisposition'



v = visualizer(kepler_df, 'koi_pdisposition', fields, (1,4), (15,5))

v.graph()
# kepler_df.dtypes



data = kepler_df.values



scaler = MinMaxScaler(feature_range=[0, 1])



# Removing first two columns because they are labels

rescaled_data = scaler.fit_transform(data[:, 2:])



pca = PCA().fit(rescaled_data)



#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.xlim((0,15))

plt.title('Variance in KOI Data')

plt.show()
pca = PCA(n_components=8)

dataset = pca.fit_transform(rescaled_data)

dataset.shape
fig = plt.figure()

ax = Axes3D(fig)



ax.scatter(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=data[:,0])
# Plot the differnt clusters

titles = ["Front View", "Top View", "Right View"]

marker = []

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,5))

for i, ax in enumerate(axes.reshape(-1)):

    ax.scatter(dataset[:, i], dataset[:, i+1], c=data[:,0], marker="2")

    ax.set_title(titles[i])  
# classification = data[:, 1:2] # koi_pdisposition

classification = data[:, :1]

train, test, trainC, testC = train_test_split(dataset, classification, test_size=0.33, random_state=0)



print("Training set size: ", train.shape)

print("Test set size: ", test.shape)
# target_names = ['CANDIDATE', 'FALSE POSITIVE'] # koi_pdisposition

target_names = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']



def run_model(model):

    # Train and fit the model

    model.fit(train, trainC.ravel())

    # Predict the test data

    predictions = model.predict(test)

    # Compute model metrics

    overall_precision = np.mean(predictions == testC.ravel()) 

    print("Percentage of correct predictions: ", overall_precision)

    report = classification_report(predictions, testC, target_names=target_names, output_dict=True)

    report = pd.DataFrame(data=report).drop(['micro avg', 'macro avg', 'weighted avg'], axis=1)

    print("\n", report)

    return report.drop(['precision', 'recall', 'support']) 
gc = GaussianNB()

gc_report = run_model(gc)
knn = KNeighborsClassifier(n_neighbors=7)

knn_report = run_model(knn)
svm = SVC(gamma='auto', probability=True)

svm_report = run_model(svm)
dt = tree.DecisionTreeClassifier()

dt_report = run_model(dt)
nn = MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(5, 2), random_state=1)

nn_report = run_model(nn)
def plot_roc(ax, model, title):

    probas = model.predict_proba(test)

    skplt.metrics.plot_roc(y_true=testC, y_probas=probas, ax=ax, title=title)



models = [(gc, "Niave Bayes"), (knn, "knn"), (svm, "SVM"), (dt, "Decision Tree"), (nn, "Neural network")]

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(30,6))

for ax, model in zip(axes, models):

    plot_roc(ax, model[0], model[1])
reports = [gc_report, knn_report, svm_report, dt_report, nn_report]

cumulative = pd.DataFrame()

for report in reports:

    cumulative = pd.concat([cumulative, report])

    

cumulative.set_index(pd.Index(["Naive Bayes f1-score", "KNN f1-score",

                               "SVM f1-score", "Decision Tree f1-score", "Neural Network f1-score"]))