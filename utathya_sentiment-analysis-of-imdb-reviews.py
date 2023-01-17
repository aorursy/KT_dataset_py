# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
################################################# import libraries ###########################################



import pandas as pd

import os

from nltk.corpus import stopwords

import string

import re

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

import plotly.plotly as py

import operator

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud

import time

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.cluster import AgglomerativeClustering
def rem_sw(df):

    # Downloading stop words

    stop_words = set(stopwords.words('english'))



    # Removing Stop words from training data

    count = 0

    for sentence in df:

        sentence = [word for word in sentence.lower().split() if word not in stop_words]

        sentence = ' '.join(sentence)

        df.loc[count] = sentence

        count+=1

    return(df)
def rem_punc(df):

    count = 0

    for s in df:

        cleanr = re.compile('<.*?>')

        s = re.sub(r'\d+', '', s)

        s = re.sub(cleanr, '', s)

        s = re.sub("'", '', s)

        s = re.sub(r'\W+', ' ', s)

        s = s.replace('_', '')

        df.loc[count] = s

        count+=1

    return(df)
def lemma(df):



    lmtzr = WordNetLemmatizer()



    count = 0

    stemmed = []

    for sentence in df:    

        word_tokens = word_tokenize(sentence)

        for word in word_tokens:

            stemmed.append(lmtzr.lemmatize(word))

        sentence = ' '.join(stemmed)

        df.iloc[count] = sentence

        count+=1

        stemmed = []

    return(df)
def stemma(df):



    stemmer = SnowballStemmer("english") #SnowballStemmer("english", ignore_stopwords=True)



    count = 0

    stemmed = []

    for sentence in df:

        word_tokens = word_tokenize(sentence)

        for word in word_tokens:

            stemmed.append(stemmer.stem(word))

        sentence = ' '.join(stemmed)

        df.iloc[count] = sentence

        count+=1

        stemmed = []

    return(df)
def get_feature(df, number):

    

    feature_list = []

    # create an instance for tree feature selection

    tree_clf = ExtraTreesClassifier()



    # first create arrays holding input and output data



    # Vectorizing Train set

    cv = CountVectorizer(analyzer='word')

    x_train = cv.fit_transform(df['review'])



    # Creating an object for Label Encoder and fitting on target strings

    le = LabelEncoder()

    y = le.fit_transform(df['label'])



    # fit the model

    tree_clf.fit(x_train, y)

    

    # Preparing variables

    importances = tree_clf.feature_importances_

    feature_names = cv.get_feature_names()

    feature_imp_dict = dict(zip(feature_names, importances))

    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

    indices = np.argsort(importances)[::-1]



    # Create the feature list

    for f in range(number):

        feature_list.append(sorted_features[f][0])

    

    return(feature_list)
def print_feature(df):

    

    # create an instance for tree feature selection

    tree_clf = ExtraTreesClassifier()



    # first create arrays holding input and output data



    # Vectorizing Train set

    cv = CountVectorizer(analyzer='word')

    x_train = cv.fit_transform(df['review'])



    # Creating an object for Label Encoder and fitting on target strings

    le = LabelEncoder()

    y = le.fit_transform(df['label'])



    # fit the model

    tree_clf.fit(x_train, y)



    # Preparing variables

    importances = tree_clf.feature_importances_

    feature_names = cv.get_feature_names()

    feature_imp_dict = dict(zip(feature_names, importances))

    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")

    for f in range(20):

        print("feature %d : %s (%f)" % (indices[f], sorted_features[f][0], sorted_features[f][1]))



    # Plot the feature importances of the forest

    plt.figure(figsize = (20,20))

    plt.title("Feature importances")

    plt.bar(range(100), importances[indices[:100]],

           color="r", align="center")

    plt.xticks(range(100), sorted_features[:100], rotation=90)

    plt.xlim([-1, 100])

    plt.show()



    return()
def get_bestrf(X, y):

    parameters = [

        {

            "n_estimators":[5, 10, 20, 50, 100],

            "criterion":['gini', 'entropy']

        }

    ]



    best_clf = GridSearchCV(clf, parameters, scoring="accuracy", verbose=5, n_jobs=4)



    best_clf.fit(X, y)

    

    return(best_clf.best_estimator_.n_estimators, best_clf.best_estimator_.criterion)
############################## Loading Data #########################################

df_master = pd.read_csv("../input/imdb_master.csv", encoding='latin-1', index_col = 0)



##################### Seperating the data in to train and test set #############################

imdb_train = df_master[["review", "label"]][df_master.type.isin(['train'])].reset_index(drop=True)

imdb_test = df_master[["review", "label"]][df_master.type.isin(['test'])].reset_index(drop=True)



##################################### Removing Stop words from training data ##################################



imdb_train['review'] = rem_sw(imdb_train['review'])

##################################### Removing Stop words from testing data ###################################



imdb_test['review'] = rem_sw(imdb_test['review'])

###################################### Removing punctuations from Train set ##################################



imdb_train['review'] = rem_punc(imdb_train['review'])

###################################### Removing punctuations from Test set ###################################



imdb_test['review'] = rem_punc(imdb_test['review'])

############################################### Stemming Train set ##########################################



imdb_train['review'] = lemma(imdb_train['review'])

imdb_train['review'] = stemma(imdb_train['review'])

############################################### Stemming Test set ###########################################



imdb_test['review'] = lemma(imdb_test['review'])

imdb_test['review'] = stemma(imdb_test['review'])



################################# Visualising the best features ################################

print_feature(imdb_train)
################################# Training Set ################################

print_feature(imdb_train)
############################################# Test set #############################################



print_feature(imdb_test)
###################################### Negative set frequency of train and test combined ################################



# Creating a frequency dataframe of stemmed train and test data set

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



# Vectorizing negative reviews set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(df_freq[df_freq.label.isin(['neg'])].review)



# Visualising the high frequency words for negative set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

df_freq.nlargest(10, 'frequency')
###################################### Positive set frequency of train and test combined ################################



# Creating a frequency dataframe of stemmed train and test data set

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



# Vectorizing pos reviews set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(df_freq[df_freq.label.isin(['pos'])].review)



# Visualising the high frequency words for positive set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

df_freq.nlargest(10, 'frequency')
######################### Lowest and highest frequency words ###########################



# Creating a frequency dataframe of stemmed train and test data set

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



# Vectorizing complete review set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(df_freq.review)



# Visualising the high and low frequency words for complete set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T

print(df_freq.nlargest(1, 'frequency'), sep='\n')

print(df_freq.nsmallest(1, 'frequency'), sep='\t')
########################## WordCloud Positive Train & Test set ##################################



# Creating a list of train and test data to analyse

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)

imdb_list = df_freq["review"][df_freq.label.isin(['pos'])].unique().tolist()

imdb_bow = " ".join(imdb_list)



# Create a word cloud for psitive words

imdb_wordcloud = WordCloud().generate(imdb_bow)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
########################## WordCloud Negative Train & Test set ##################################



# Creating a list of train and test data to analyse

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)

imdb_list = df_freq["review"][df_freq.label.isin(['neg'])].unique().tolist()

imdb_bow = " ".join(imdb_list)



# Create a word cloud for negative words

imdb_wordcloud = WordCloud().generate(imdb_bow)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
########################## Histogram Positive Train & Test set ##################################



#Combining cleaned train and test data

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



# Creating an object for Count vectorizer and fitting it to positive dataset

hist_cv = CountVectorizer(stop_words = 'english', analyzer='word')

hist_pos = hist_cv.fit_transform(df_freq[df_freq.label.isin(['pos'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_pos.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data[0], bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Zooming in on below 100 frequency words



zoom_data = [f for f in data[0] if f <= 100]

binwidth = 5

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.xlim(0, 100)

plt.show()
# Having a look at above 100 frequency words more closely



zoom_data = [f for f in data[0] if f > 100]

binwidth = 2500

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
########################## Histogram Negative Train & Test set ##################################



#Combining cleaned train and test data

df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



# Creating an object for Count vectorizer and fitting it to positive dataset

hist_cv = CountVectorizer(stop_words = 'english', analyzer='word')

hist_neg = hist_cv.fit_transform(df_freq[df_freq.label.isin(['neg'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_neg.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data, bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at less than 100 frequency words more closely



zoom_data = [f for f in data[0] if f <= 100]

binwidth = 5

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at above 100 frequency words more closely



zoom_data = [f for f in data[0] if f > 100]

binwidth = 2500

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
df_freq = pd.concat([imdb_train, imdb_test], ignore_index = True)



word_list = get_feature(df_freq, 1000)



# Removing non prefered words from training and test combined data

count = 0

for sentence in df_freq['review']:

    sentence = [word for word in sentence.lower().split() if word in word_list]

    sentence = ' '.join(sentence)

    df_freq.loc[count, 'review'] = sentence

    count+=1
########################## WordCloud Positive Train & Test set post feature selection ##################################



# Creating a list of train and test data to analyse

imdb_list = df_freq["review"][df_freq.label.isin(['pos'])].unique().tolist()

imdb_bow = " ".join(imdb_list)



# Create a word cloud for psitive words

imdb_wordcloud = WordCloud().generate(imdb_bow)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
########################## WordCloud Negative Train & Test set post feature selection ##################################



# Creating a list of ham data only to analyse

imdb_list = df_freq["review"][df_freq.label.isin(['neg'])].unique().tolist()

imdb_bow = " ".join(imdb_list)



# Create a word cloud for ham

imdb_wordcloud = WordCloud().generate(imdb_bow)



# Show the created image of word cloud

plt.figure(figsize=(20, 20))

plt.imshow(imdb_wordcloud)

plt.show()
########################## Histogram Positive Train & Test set post feature selection ##################################



# Creating an object for Count vectorizer and fitting it to positive dataset

hist_cv = CountVectorizer(stop_words = 'english', analyzer='word')

hist_pos = hist_cv.fit_transform(df_freq[df_freq.label.isin(['pos'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_pos.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data, bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at less than 100 frequency words more closely



zoom_data = [f for f in data[0] if f <= 100]

binwidth = 5

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=False)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at above 100 frequency words more closely



zoom_data = [f for f in data[0] if f > 100]

binwidth = 2500

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=False)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
########################## Histogram Negative Train & Test set post feature selection ##################################



# Creating an object for Count vectorizer and fitting it to positive dataset

hist_cv = CountVectorizer(stop_words = 'english', analyzer='word')

hist_pos = hist_cv.fit_transform(df_freq[df_freq.label.isin(['neg'])].review)



# Visualising the histogram for positive reviews only from train and dataset

data = hist_pos.sum(axis=0).tolist()

binwidth = 2500

plt.hist(data, bins=range(min(data[0]), max(data[0]) + binwidth, binwidth), log=True)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at less than 100 frequency words more closely



zoom_data = [f for f in data[0] if f <= 100]

binwidth = 5

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=False)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
# Having a look at above 100 frequency words more closely



zoom_data = [f for f in data[0] if f > 100]

binwidth = 2500

plt.hist(zoom_data, bins=range(min(zoom_data), max(zoom_data) + binwidth, binwidth), log=False)

plt.title("Gaussian Histogram")

plt.xlabel("Frequency")

plt.ylabel("Number of instances")

plt.show()
imdb_unsup = df_master[["review", "label"]][df_master.label.isin(['unsup'])].reset_index(drop=True)



# Cleaning Unlabelled data



imdb_unsup['review'] = rem_sw(imdb_unsup['review'])

imdb_unsup['review'] = rem_punc(imdb_unsup['review'])

imdb_unsup['review'] = lemma(imdb_unsup['review'])

imdb_unsup['review'] = stemma(imdb_unsup['review'])



# Vectorizing unlabelled reviews set

vect = CountVectorizer(stop_words = 'english', analyzer='word')

vect_pos = vect.fit_transform(imdb_unsup.review)



# Creating a dataframe for the high frequency words for unlabelled reviews set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T



# Removing high frequency and low frequency data for more accuracy

word_list = df_freq.nlargest(100, 'frequency').index

word_list = word_list.append(df_freq.nsmallest(43750, 'frequency').index)



# Removing unwanted words based on word_list from unlabelled data

count = 0

for sentence in imdb_unsup['review']:

    sentence = [word for word in sentence.lower().split() if word not in word_list]

    sentence = ' '.join(sentence)

    imdb_unsup.loc[count, 'review'] = sentence

    count+=1



################################## Preparing dataframe for model ##############################



# Creating df_algo dataframe which will be used for hypothesis testing

df_algo = pd.concat([imdb_train, imdb_test], keys=['train', 'test'])

df_algo = df_algo.reset_index(col_level=1).drop(['level_1'], axis=1)



# Cleaning the dataset

df_algo['review'] = rem_sw(df_algo['review'])

df_algo['review'] = rem_punc(df_algo['review'])

df_algo['review'] = lemma(df_algo['review'])

df_algo['review'] = stemma(df_algo['review'])



# df_algo = pd.read_csv("clean_algo.csv", encoding='latin-1', index_col = 0) # Uncomment this line to load from csv



################################### Removing non feature words ###############################



# Creating the feature word_list

# Selecting 14440 feature selected words based on 80-20 rule

word_list = get_feature(df_algo[['review', 'label']], 14440)



# Removing non prefered words from training and test combined data

count = 0

for sentence in df_algo['review']:

    sentence = [word for word in sentence.lower().split() if word in word_list]

    sentence = ' '.join(sentence)

    df_algo.loc[count, 'review'] = sentence

    count+=1



################################## Splitting with feature selection data ###############################a



# Vectorising the required data

vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')

vect_algo.fit(df_algo.review)

Xf_train = vect_algo.transform(df_algo[df_algo['level_0'].isin(['train'])].review)

Xf_test = vect_algo.transform(df_algo[df_algo['level_0'].isin(['test'])].review)



# Encoding target data

# Creating an object and fitting on target strings

le = LabelEncoder()

yf_train = le.fit_transform(df_algo[df_algo['level_0'].isin(['train'])].label)

yf_test = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)



########################################### Naive Bayes #########################################



# Fit the Naive Bayes classifier model to the object

clf = MultinomialNB()

clf.fit(Xf_train, yf_train)



# predict the outcome for testing data

predictions = clf.predict(Xf_test)



# check the accuracy of the model

accuracy = accuracy_score(yf_test, predictions)

print("Observation: Naive Bayes Classification gives an accuracy of %.2f%% on the testing data" %(accuracy*100))
##################################### Using K-means to create two clusters ##################################### 



# Vectorizing dataset

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(imdb_unsup.review)

 

# Creating a k-means object and fitting it to target variable

true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

model.fit(X)

 

# Visualising the 2 clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :10]:

        print(' %s' % terms[ind])
# Prediction for test set using Kmeans clusters

Y = vectorizer.transform(imdb_test.review)

prediction = model.predict(Y)



# Actual results of test sets for comparison

le = LabelEncoder()

y = le.fit_transform(imdb_test.label)



# check the accuracy of the model

accuracy = accuracy_score(y, prediction)

if accuracy < 0.5:

    accuracy = 1 - accuracy

print("Observation: The unsupervised learning gives an accuracy of %.2f%% on the testing data" %(accuracy*100))
imdb_unsup = df_master[["review", "label"]][df_master.label.isin(['unsup'])].reset_index(drop=True)



# Cleaning Unlabelled data



imdb_unsup['review'] = rem_sw(imdb_unsup['review'])

imdb_unsup['review'] = rem_punc(imdb_unsup['review'])

imdb_unsup['review'] = lemma(imdb_unsup['review'])

imdb_unsup['review'] = stemma(imdb_unsup['review'])



# Vectorizing unlabelled reviews set

vect = CountVectorizer(analyzer='word')

vect_pos = vect.fit_transform(imdb_unsup.review)



# Creating a dataframe for the high frequency words for unlabelled reviews set

df_freq = pd.DataFrame(vect_pos.sum(axis=0), columns=list(vect.get_feature_names()), index = ['frequency']).T



# Removing high frequency and low frequency data for more accuracy



word_list = df_freq.nlargest(100, 'frequency').index

word_list = word_list.append(df_freq.nsmallest(43750, 'frequency').index)



# Removing unwanted words based on word_list from unlabelled data

count = 0

for sentence in imdb_unsup['review']:

    sentence = [word for word in sentence.lower().split() if word not in word_list]

    sentence = ' '.join(sentence)

    imdb_unsup.loc[count, 'review'] = sentence

    count+=1

    

##################################### Using K-means to create clusters ##################################### 



# Vectorizing dataset

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(imdb_unsup.review)

 

# Creating a k-means object and fitting it to target variable

true_k = 9

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=13)

model.fit(X)

 

# Visualising the clusters

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()

for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :10]:

        print(' %s' % terms[ind])