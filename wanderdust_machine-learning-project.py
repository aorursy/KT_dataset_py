import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
# Load the images and the labels

train_raw = pd.read_csv("../input/data-files/x_train_gr_smpl.csv")

labels = pd.read_csv("../input/data-files/y_train_smpl.csv")





# Display top 2 rows for each dataframe

display(train_raw.head(2))

display(labels.head(2))





# Train contains the image data and labels in the same table.

train = pd.read_csv("../input/data-files/x_train_gr_smpl.csv")

train['label'] = labels

fig = sns.countplot(x="label", data=train)

fig.set_title("Frequency Table")
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train_raw, labels, test_size=0.1, shuffle=True, random_state=3)
from sklearn.naive_bayes import MultinomialNB
# Create a function that runs a basic Naive Bayes without any optimization

# that we can run for different combinations of features and compare scores



def basic_Naive_Bayes(x_train, y_train, x_test, y_test = []):

    # initiate the Naive Bayes class

    clf = MultinomialNB()

    # Train on the train data .values.ravel() is added to y_train to avoid getting an error message.

    clf.fit(x_train, y_train.values.ravel())



    # Make predictions on the test data.

    output = clf.predict(x_test)

    

    return output



output = basic_Naive_Bayes(x_train, y_train, x_test)
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix, precision_score



# checking the accuracy

accuracy = accuracy_score(y_true=y_test, y_pred=output)

print("The accuracy is {0:.4f}".format(accuracy))



# checking the f1 score

f1_s = f1_score(y_true=y_test, y_pred=output, average='weighted')

print("The f1 score is {0:.4f}".format(f1_s))



# checking the recall for average= micro

recall = recall_score(y_true=y_test, y_pred=output, average='micro')

print("The avg = micro recall is {0:.16f}".format(recall))



# checking the recall for average= macro

recall = recall_score(y_true=y_test, y_pred=output, average='macro')

print("The avg = macro recall is {0:.16f}".format(recall))



# checking the recall for average= weighted

recall = recall_score(y_true=y_test, y_pred=output, average='weighted')

print("The avg = weighted recall is {0:.16f}".format(recall))





print(classification_report(y_true=y_test, y_pred=output))
from sklearn.metrics import confusion_matrix



# Create confusion matrix from test data

conf_matrix_data = confusion_matrix(y_true=y_test, y_pred=output)



# Make confusion matrix with colors

plt.figure(figsize=(10,5)) # Size of the plot

ax = sns.heatmap(conf_matrix_data, annot=True) # Creating the colors
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=0)

tree.fit(train_raw, labels.values.ravel())



# Look at the most important features according to the tree

importances  = tree.feature_importances_
def extract_top_indices(my_array, n_indices):

    # This gets the indices for the top N highest values in my_array

    feature_idx = my_array.argsort()[-n_indices:][::-1]

    # convert the indices to string to match column names in the pandas dataframe

    str_columns = [str(i) for i in feature_idx]

    

    return str_columns
def naive_bayes_on_top_N (correlations, n_indices, labels, binary=True, full_report=False):

    top_N_indices = extract_top_indices(correlations, n_indices)

    

    train_data = train_raw[top_N_indices]

     

    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.1, shuffle=True, random_state=3)

    

    output = basic_Naive_Bayes(x_train, y_train, x_test)

    

    # Binary when we want binary classification (0 or 1)

    # Micro when we want multiclass classification (0 - 9)

    if binary:

        avg = 'binary'

    else:

        avg='macro'

        

    f1_sc = f1_score(y_true=y_test, y_pred=output, average=avg)  # Ratio between precision and recall

    accuracy = accuracy_score(y_true=y_test, y_pred=output)

    

    # To show the full report for each class set full_report to True

    if full_report:

        cr=classification_report(y_true=y_test, y_pred=output)

    else:

        cr="N/A"



    # More metrics can be added ...

    print("Running Naive bayes on top {} classes".format(n_indices))

    print("* Accuracy: {}".format(accuracy))

    print("* F1 Score: {}". format(f1_sc))

    print("classification_report: \n{}".format(cr))

    print("_________________________________")
def test_importances_scores (path, top_n_array):

        labels = pd.read_csv(path)

        

        for top_n in top_n_array:            

            naive_bayes_on_top_N(importances, top_n, labels, binary=False, full_report=True)





all_class_labels_path = "../input/data-files/y_train_smpl.csv"

test_importances_scores(all_class_labels_path, [20, 50, 100])
# Save the locations of the files in a list to easily loop throgh them

list_classes = [

    "../input/data-files/y_train_smpl_0.csv",

    "../input/data-files/y_train_smpl_1.csv",

    "../input/data-files/y_train_smpl_2.csv",

    "../input/data-files/y_train_smpl_3.csv",

    "../input/data-files/y_train_smpl_4.csv",

    "../input/data-files/y_train_smpl_5.csv",

    "../input/data-files/y_train_smpl_6.csv",

    "../input/data-files/y_train_smpl_7.csv",

    "../input/data-files/y_train_smpl_8.csv",

    "../input/data-files/y_train_smpl_9.csv",

]
def test_importances_scores_all (paths, top_n_array):

    

    # For every file with labels do:

    for i, path in enumerate(paths):

        labels = pd.read_csv(path)

        # Run the tree

        tree = DecisionTreeClassifier(random_state=3)

        tree.fit(train_raw, labels.values.ravel())

        importances  = tree.feature_importances_

        

        print("\nCLASS {}".format(i))

    

        for top_n in top_n_array:

            naive_bayes_on_top_N(importances, top_n, labels)

        

test_importances_scores_all(list_classes, [20, 50, 100])
import matplotlib.pyplot as plt



acc_list = []



for i in range (1, 81) :

    # Top 100-i features

    top_100_indices_i = extract_top_indices(importances, 100)

    top_100_features_i = train_raw[top_100_indices_i[0:-i]]

    

    x_train_100_i, x_test_100_i, y_train_100_i, y_test_100_i = train_test_split(top_100_features_i, labels, test_size=0.1, shuffle=True, random_state=3)

    



    output_100_i = basic_Naive_Bayes(x_train_100_i, y_train_100_i, x_test_100_i)



    # print("=======================================================================")

    # print("Classification Report for top 100-i features :")

    # print(classification_report(y_true=y_test, y_pred=output_100_i))

    accuracy_100_i = accuracy_score(y_true=y_test, y_pred=output_100_i)

    # print("The accuracy for top 100 features is {0:.4f}".format(accuracy_100_i))

    acc_list.append(accuracy_100_i)

    # print(len(acc_list))





#acc_list.reverse() 

plt.plot(acc_list)

plt.ylabel('Accuracy based on number of top attributes')

plt.show()

    

import scipy as sc
def get_correlation_array(path):

    # Read the file with the classes

    class_labels = pd.read_csv(path)

    

    correlations = []

    

    for column in train_raw.columns:

        corr, _ = sc.stats.pearsonr(train_raw[[column]], class_labels[["0"]])

        correlations.append(corr[0])



    correlations = np.asarray(correlations)

    # Get the absolute values (we want the closest values to 1 or -1)

    correlations = np.absolute(correlations)

    

    return correlations

def test_correlation_scores_all(path, top_n_array):

        correlations = get_correlation_array(path)

        labels = pd.read_csv(path)

        

        for top_n in top_n_array:            

            naive_bayes_on_top_N(correlations, top_n, labels, binary=False, full_report=True)

            

all_class_labels_path = "../input/data-files/y_train_smpl.csv"

test_correlation_scores_all(all_class_labels_path, [20, 50, 100])
def test_correlation_scores (paths, top_n_array):

    

    # For every file with labels do:

    for i, path in enumerate(paths):

        correlations = get_correlation_array(path)

        print("\nCLASS {}".format(i))

        

        labels = pd.read_csv(path)

        for top_n in top_n_array:

            naive_bayes_on_top_N(correlations, top_n, labels)

        



test_correlation_scores (list_classes, [20, 50, 100])
# Comment out whichever one you dont want to use.

#scores_array = correlations = get_correlation_array('../input/data-files/y_train_smpl.csv')

scores_array = importances

n_features = 50 # Use whichever number we want...



# I'll use the top n according to tree data.

top_N_indices = extract_top_indices(scores_array, n_features)
# Assign our features and labels to variables.

x_data = pd.DataFrame(train_raw[top_N_indices])

y_data = pd.DataFrame(labels)



# Split the data into train and test

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, shuffle=True, random_state=3)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV



def fit_model(x_train, y_train, clf, params):    

    # The classifier

    estimator = clf()

    

    # Convert metric into scorer

    scorer = make_scorer(accuracy_score)

    

    # (estimator, param_grid, scoring, cv) which have values 'regressor', 'params', 'scoring_fnc', and 'cv_sets' respectively.

    grid = GridSearchCV(estimator=estimator, param_grid=params, scoring=scorer, cv=10)



    # Fit the grid search object to the data to compute the optimal model

    grid = grid.fit(x_train, y_train.values.ravel())

    

    print("Best params: {}".format(grid.best_params_))



    # Return the optimal model after fitting the data

    return grid.best_estimator_
def run_metrics (y_true, output):

    accuracy = accuracy_score(y_true, output)

    f1_sc = f1_score(y_true, output, average='weighted')

    

    print("* Accuracy: {}".format(accuracy))

    print("* F1 Score: {}". format(f1_sc))
from sklearn.naive_bayes import MultinomialNB



# Parameters to try

params = {'alpha': [0.1, 0.2, 0.4, 0.7, 0.9, 1],

          'fit_prior': [True, False]}



clf_MNB = fit_model(x_train, y_train, MultinomialNB, params)

output = clf_MNB.predict(x_test)



run_metrics(y_test, output)
# Create confusion matrix from test data

conf_matrix_data = confusion_matrix(y_true=y_test, y_pred=output)



# Make confusion matrix with colors

plt.figure(figsize=(10,5)) # Size of the plot

ax = sns.heatmap(conf_matrix_data, annot=True) # Creating the colors
from sklearn.naive_bayes import GaussianNB



# Parameters to try

params = {'var_smoothing': [1e-9, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4]}



clf_Gauss = fit_model(x_train, y_train, GaussianNB, params)

output = clf_Gauss.predict(x_test)



run_metrics(y_test, output)
from sklearn.naive_bayes import ComplementNB



# Parameters to try

params = {'alpha': [1e-10, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1],

         'fit_prior': [True, False],

          'norm': [True, False]

         }



clf_Compl = fit_model(x_train, y_train, ComplementNB, params)

output = clf_Compl.predict(x_test)



run_metrics(y_test, output)
from sklearn.naive_bayes import BernoulliNB



# Parameters to try

params = {'alpha': [1e-10, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1],

         'fit_prior': [True, False],

          'binarize': [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1]

         }



clf_Compl = fit_model(x_train, y_train, BernoulliNB, params)

output = clf_Compl.predict(x_test)



run_metrics(y_test, output)
from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
#Function k_means applies k-means clustering alrorithm on dataset and build confusion matrix of cluster and actual labels



def k_means(n_clust, data_frame, true_labels):

    #Initializing parameters 

    k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=10)

    #Compute k-means clustering

    k_means.fit(data_frame)

    c_labels = k_means.labels_

    #Compute cluster centers and predict cluster index for each sample

    y_clust = k_means.predict(data_frame)

    # Create confusion matrix from data

    conf_matrix_data = confusion_matrix(y_true=true_labels, y_pred=c_labels)

    plt.figure(figsize=(10,5))

    ax = sns.heatmap(conf_matrix_data, annot=True)

    print('Adjusted rand score :',adjusted_rand_score(true_labels, y_clust))
Data = pd.DataFrame(train_raw)

Data['label'] = labels

Labels = Data['label']

print('Kmeans result : All Data')

k_means(n_clust=10, data_frame=Data, true_labels=Labels)
def test_k_means(top_n_array):

        #Top features

        scores_array = importances

        labels = pd.read_csv("../input/data-files/y_train_smpl.csv")

        for top_n in top_n_array:

            top_N_indices = extract_top_indices(scores_array, top_n)

            Data = pd.DataFrame(train_raw[top_N_indices])

            Data['label'] = labels

            Labels = Data['label']

            print('Kmeans result : Features',top_n)

            k_means(n_clust=10, data_frame=Data, true_labels=Labels)

        



test_k_means ([10, 50, 100])
from time import time

import numpy as np

from scipy import ndimage

from matplotlib import pyplot as plt

from sklearn import manifold

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import confusion_matrix, adjusted_rand_score

from sklearn.model_selection import train_test_split





# Load the images and the labels

train_raw = pd.read_csv("../input/data-files/x_train_gr_smpl.csv")

labels = pd.read_csv("../input/data-files/y_train_smpl.csv")

print(labels.shape)


Data = pd.DataFrame(train_raw)

Data['label'] = labels

Labels = Data['label']



y_test_1d = []

for i in range(0, y_test.shape[0]):

    y_test_1d.append(y_test[i])



from sklearn.feature_extraction import image

from sklearn.cluster import spectral_clustering



from sklearn.cluster import Birch

from sklearn.cluster import AffinityPropagation



def Birch_clustering(n_clust, data, label, branching_factor, threshold):

    brc = Birch(branching_factor=50, n_clusters=10, threshold=0.5,compute_labels=True)

    pred_labels = brc.fit_predict(data)

    c_labels = brc.labels_   

    

    # Make confusion matrix with colors

    conf_matrix_data = confusion_matrix(y_true=label, y_pred=pred_labels)    

    plt.figure(figsize=(20, 10))  # Size of the plot

    ax = sns.heatmap(conf_matrix_data, annot=True)  # Creating the colors

    print('Adjusted rand score :', adjusted_rand_score(label, pred_labels))

    

def Affinity_clustering(data, label):

    Affinity = AffinityPropagation()

    pred_labels = Affinity.fit_predict(data)

    c_labels = Affinity.labels_

    conf_matrix_data = confusion_matrix(y_true=label, y_pred=pred_labels)

    # Make confusion matrix with colors

    plt.figure(figsize=(20, 10))  # Size of the plot

    ax = sns.heatmap(conf_matrix_data, annot=True)  # Creating the colors

    print('Adjusted rand score :', adjusted_rand_score(label, pred_labels))

    print(c_labels)

    print(pred_labels)

    

def Hierarchical_clustering(n_clust, data, label, linkage):

    

    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=n_clust)

    #clustering.fit(data)

    pred_labels = clustering.fit_predict(data)

    c_labels = clustering.labels_ 

    conf_matrix_data = confusion_matrix(y_true=label, y_pred=pred_labels)

    # Make confusion matrix with colors

    plt.figure(figsize=(20,10)) # Size of the plot

    ax = sns.heatmap(conf_matrix_data, annot=True) # Creating the colors

    print('Adjusted rand score :',adjusted_rand_score(label, pred_labels))

    

def Spectral_clustering(n_clust, data, label, AssignLabels, RandomState):

    clustering = SpectralClustering(n_clusters=n_clust,assign_labels=AssignLabels,random_state=RandomState).fit(Data)

    pred_labels =  clustering.fit_predict(data)

    c_labels = clustering.labels_

    conf_matrix_data = confusion_matrix(y_true=label, y_pred=pred_labels)

    # Make confusion matrix with colors

    plt.figure(figsize=(20, 10))  # Size of the plot

    ax = sns.heatmap(conf_matrix_data, annot=True)  # Creating the colors

    print('Adjusted rand score :', adjusted_rand_score(label, pred_labels))    



Spectral_clustering(n_clust=10, data = Data, label = Labels, AssignLabels ="discretize" , RandomState = 0)
Hierarchical_clustering(n_clust = 10,data = Data, label = Labels, linkage = "ward")

Hierarchical_clustering(n_clust = 10,data = Data, label = Labels, linkage = "complete")

Hierarchical_clustering(n_clust = 10,data = Data, label = Labels, linkage = "average")

Hierarchical_clustering(n_clust = 10,data = Data, label = Labels, linkage = "single")
Hierarchical_clustering(n_clust = 10,data = x_test, label = y_test_1d, linkage = "single")
Birch_clustering(n_clust = 10, data = Data, label = Labels, branching_factor =50, threshold =1)
Birch_clustering(n_clust = 10, data = Data, label = Labels, branching_factor =50, threshold =0.05)
Birch_clustering(n_clust = 10, data = Data, label = Labels, branching_factor =10, threshold =1)
Birch_clustering(n_clust = 10, data = Data, label = Labels, branching_factor =10, threshold =0.05)
Affinity_clustering(data = Data, label = Labels)