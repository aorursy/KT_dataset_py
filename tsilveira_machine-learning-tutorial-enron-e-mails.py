# Udacity statements
import pickle

# Data analysis packages:
import pandas as pd
import numpy as np
#from datetime import datetime as dt

# Visualization packages:
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## Importing "manually" some functions provided by Udacity and available at 
## https://github.com/tbnsilveira/DAND-MachineLearning/blob/master/tools/feature_format.py
def featureFormat( dictionary, features, remove_NaN=True, remove_all_zeroes=True, remove_any_zeroes=False, sort_keys = False):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """
    return_list = []
    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value=="NaN" and remove_NaN:
                value = 0
            tmp_list.append( float(value) )

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'poi':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list
        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append( np.array(tmp_list) )
    return np.array(return_list)

def targetFeatureSplit( data ):
    """ given a numpy array like the one returned from featureFormat, separate out the first feature and put it into its own list
    (this should be the quantity you want to predict) return targets and features as separate lists (sklearn can generally 
    handle both lists and numpy arrays as input formats when training/predicting) """
    target = []
    features = []
    for item in data:
        target.append( item[0] )
        features.append( item[1:] )
    return target, features
## Forcing pandas to display any number of elements
pd.set_option('display.max_columns', None)
pd.options.display.max_seq_items = 2000
data_dict = pd.read_pickle("../input/final_project_dataset.pkl")
## What is the data type and length?
print('Dataset type: ',type(data_dict))
print('Dataset length: ',len(data_dict))
## Exploring the dataset through pandas.Dataframe
dataset = pd.DataFrame.from_dict(data_dict, orient='index')
dataset.head()
dataset.describe()
## Checking the feature data type:
features_to_check = []
for col in dataset.columns:
    datatype = type(dataset[col][0])
    ## Uncomment the line below for a verbose mode:
    # print '{} has type {}'.format(col,datatype)
    ## Here we select those attributes which have string type data:
    if datatype is str:
        features_to_check.append(col)
## Printing out the features that must be checked (string types are not iterable!)
features_to_check
dataset['loan_advances'].unique()
dataset[dataset['loan_advances']!='NaN']
dataset['director_fees'].unique()
dataset[dataset['director_fees']!='NaN']
for column in dataset.columns:
    dataset[column] = dataset[column].apply(lambda x: np.NaN if x == 'NaN' else x)
## Checking the dataset information:
dataset.info()
notNullDataset = dataset.dropna(thresh=15)
notNullDataset.info()
## Only numerical features are being considered here
financialFeatures = ['salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
                     'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive']
behavioralFeatures = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other']
allFeatures = ['poi','salary','bonus', 'exercised_stock_options', 'restricted_stock', 'shared_receipt_with_poi',
               'total_payments', 'expenses', 'total_stock_value', 'deferred_income', 'long_term_incentive',
               'to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'other']
dataset.fillna(0,inplace=True)
def visualizeFeat(series, figsize):
    ''' series = pandas.series, which can be inputed as "dataframe['feature']
        figsize = (width,length)'''
    fig, axes = plt.subplots(2,1,figsize=figsize, sharex=True)
    series.plot(kind='kde', ax=axes[0])
    sns.boxplot(x=series, ax=axes[1])
    plt.xlim(series.min(), series.max()*1.1)
    return
def visualize3Feats(dataset, features):
    '''Shows the distribution and the boxplot for the given features of a pandas.Dataframe:
        dataset = pandas dataframe.
        features = list of features of interest'''
    ## Building the Figure:
    fig, axes = plt.subplots(2,3,figsize=(15,6), sharex=False)
    for col, feat in enumerate(features):
        dataset[feat].plot(kind='kde', ax=axes[0,col])
        sns.boxplot(x=dataset[feat], ax=axes[1,col])
        axes[0,col].set_xlim(dataset[feat].min(), dataset[feat].max()*1.1);
        axes[1,col].set_xlim(dataset[feat].min(), dataset[feat].max()*1.1);
    return
### Visualizing financial features:
numPlots = int(np.ceil(len(financialFeatures)/3.))
for i in range(numPlots):
    shift = i*3
    visualize3Feats(dataset,financialFeatures[0+shift:3+shift])
dataset.drop('TOTAL',inplace=True)  #Removing the anomalous instance
## Counting gender classes
dataset['poi'].value_counts()
from sklearn.cross_validation import train_test_split
## For pandas.Dataframe the train_test_split is given in a straight way:
trainData, testData = train_test_split(dataset, test_size=0.3, random_state=42, stratify=dataset['poi'])
## Converting boolean data into int:
dataset['poi'] = dataset['poi'].apply(lambda x: int(x))
trainData['poi'] = trainData['poi'].apply(lambda x: int(x))
testData['poi'] = testData['poi'].apply(lambda x: int(x))
## Evaluating the class distribution:
fig2, axes2 = plt.subplots(1,3,figsize=(15,3), sharex=False);
dataset['poi'].plot(kind='hist', ax=axes2[0], title='Total dataset');
trainData['poi'].plot(kind='hist', ax=axes2[1], title='Train subset');
testData['poi'].plot(kind='hist', ax=axes2[2], title='Test subset');
## Calculating the correlation among features by Pearson method
correlationDataframe = dataset[allFeatures].corr()

# Drawing a heatmap with the numeric values in each cell
fig1, ax = plt.subplots(figsize=(14,10))
fig1.subplots_adjust(top=.945)
plt.suptitle('Features correlation from the Enron POI dataset', fontsize=14, fontweight='bold')

cbar_kws = {'orientation':"vertical", 'pad':0.025, 'aspect':70}
sns.heatmap(correlationDataframe, annot=True, fmt='.2f', linewidths=.3, ax=ax, cbar_kws=cbar_kws);
from sklearn.decomposition import PCA
## Listing the financial features
financialFeatures
## Defining only one resulting component:
pca = PCA(n_components=1)
pca.fit(dataset[financialFeatures])
pcaComponents = pca.fit_transform(dataset[financialFeatures])
dataset['financial'] = pcaComponents
sns.pairplot(dataset,hue='poi',vars=['salary','bonus'], diag_kind='kde');
sns.pairplot(dataset,hue='poi',vars=['salary','financial'], diag_kind='kde');
## Adding up the new 'financial' feature to the 'allFeatures' list:
allFeatures.append('financial')
financialFeatures.append('financial')
allFeatures
from sklearn.feature_selection import SelectPercentile, f_classif

selectorDataset = dataset[financialFeatures]
selectorLabel = dataset['poi']

# #############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 5% most significant features
selector = SelectPercentile(f_classif, percentile=5)
selector.fit(selectorDataset, selectorLabel)
## Plotting the features selection: 
X_indices = np.arange(selectorDataset.shape[-1])
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',
        edgecolor='black')
len(scores)
## Printing out the selected financial features: 
selectedFeatures = ['poi']  #'poi' must be the first one due to the evaluation methods defined by Udacity.
for ix, pval in enumerate(scores):
    print(financialFeatures[ix],': ',pval)
    if (pval >= 0.45):
        selectedFeatures.append(financialFeatures[ix])
selectedFeatures
strategicFeatures = ['poi'] + behavioralFeatures + ['financial']
strategicFeatures
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(dataset[allFeatures])
dataset[allFeatures] = scaler.transform(dataset[allFeatures])
## Converting back the pandas Dataframe to the dictionary structure, in order to use the Udacity evaluating code.
my_dataset = dataset.to_dict(orient='index')
features_list = selectedFeatures
#features_list = strategicFeatures

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
## Splitting the data:
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
## Defining an evaluation metric based on (http://scikit-learn.org/stable/modules/model_evaluation.html)
from sklearn.metrics import classification_report
def evaluateClassif(clf):
    classes=['Non-POI','POI']  ## Defining the classes labels
    predTrain = clf.predict(features_train)
    print('################### Training data ##################')
    print(classification_report(labels_train, predTrain, target_names=classes))
    
    predTest = clf.predict(features_test)
    print('################### Testing data ###################')
    print(classification_report(labels_test, predTest, target_names=classes))
    
    return
## Importing GridSearch algorithm for parameter selection:
from sklearn.model_selection import GridSearchCV
#%%## Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_parameters = {}
clf_nb = GridSearchCV(nb, nb_parameters)
clf_nb.fit(features_train, labels_train)
evaluateClassif(clf_nb)
### Adaboost Classifier
### http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
## Defining the Adaboost parameters for GridSearch:
abc_parameters = {"learning_rate" : [0.5, 1., 2., 5., 10., 100.],
                 "n_estimators": [10,50,100,200,500,900,2000],
                 "algorithm": ['SAMME','SAMME.R']}

dtc = DecisionTreeClassifier(random_state = 42, max_features = "auto", max_depth = None)
abc = AdaBoostClassifier(base_estimator=dtc)

# run grid search
clf_adaboost = GridSearchCV(abc, param_grid=abc_parameters)
clf_adaboost.fit(features_train, labels_train)
evaluateClassif(clf_adaboost)
from sklearn import svm
svm_parameters = {'kernel':['linear','rbf','poly','sigmoid'], 
                  'C':[0.5,1.,5.,10.,50.,100.,1000.], 'gamma':['scale']}
svr = svm.SVC()
clf_svc = GridSearchCV(svr, svm_parameters);
clf_svc.fit(features_train, labels_train)
evaluateClassif(clf_svc)
