# import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling



# data visualization

import seaborn as sns

from matplotlib import pyplot as plt

from matplotlib import style  



# Tools for preprocessing input data

import re #reg expressions for find & replace

from bs4 import BeautifulSoup

import nltk

from nltk import word_tokenize

import nltk.data

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



import logging



# Tools for creating ngrams and vectorizing input data

from gensim.models import Word2Vec, Phrases

from gensim.models import word2vec



# Tools for building a model

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer #for bag of words feature extraction

from sklearn.ensemble import RandomForestClassifier 



# ignore warnings

import warnings

warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# create data handles

train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)

unlabeled_train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)

test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3 )
# here we create a useful function to display all important confusion matrix metrics

def display_confusion_matrix(target, prediction, score=None):

    cm = metrics.confusion_matrix(target, prediction)

    plt.figure(figsize=(6,6))

    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    if score:

        score_title = 'Accuracy Score: {0}'.format(round(score, 5))

        plt.title(score_title, size = 14)

    classification_report = pd.DataFrame.from_dict(metrics.classification_report(target, prediction, output_dict=True))

    display(classification_report.round(2))
# In this case we do not need extensive data cleaning & feature engineering. 

#instead of profile report we can simply use head() and info()



#labeled_train.profile_report()
train.head()
train.columns.values
train['review'][0]
train.info()
unlabeled_train.head()
test.head()
# Initialize the BeautifulSoup object on a single movie review

# provides review without tags or markup

example1 = BeautifulSoup(train["review"][0])  
example1.get_text()
# Use regular expressions to do a find-and-replace

letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for

                      " ",                   # The pattern to replace it with

                      example1.get_text() )  # The text to search

letters_only
lower_case = letters_only.lower()        # Convert to lower case

lower_case
words = lower_case.split()               # Split into words

words
# stopword list from nltk.corpus

stopwords.words("english") 
# Remove stop words from "words"

words = [w for w in words if not w in stopwords.words("english")]

words
# function for reusable code



# example raw_review = train["review"] or raw_review = test["review"]



def review_to_words(raw_review):

    # Function to convert a raw review to a string of words

    # The input is a single string (a raw movie review), and 

    # the output is a single string (a preprocessed movie review)

    #

    # 1. Remove HTML

    review_text = BeautifulSoup(raw_review).get_text() 

    #

    # 2. Remove non-letters        

    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))   
# loop through and clean all of the training set at once



# Get the number of reviews based on the dataframe column size

num_reviews = train["review"].size



# Initialize an empty list to hold the clean reviews

clean_train_reviews = []



# Loop over each review; create an index i that goes from 0 to the length

# of the movie review list 

for i in range(0, num_reviews):

    # Call our function for each one, and add the result to the list of

    # clean reviews

    clean_train_reviews.append(review_to_words(train["review"][i]))
clean_train_reviews
# use CountVectorizer scikit-learn object to create bag of words

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, 

                             preprocessor = None, stop_words = None, max_features = 5000) 
# fit-transform learns the vocabulary and transforms training data into feature vectors

train_data_features = vectorizer.fit_transform(clean_train_reviews)



# transform list of strings to numpy array for more efficiency

train_data_features = train_data_features.toarray()
train_data_features.shape
# Take a look at the words in the vocabulary

vocab = vectorizer.get_feature_names()

vocab
# print the counts of each word in vocabulary



# Sum up the counts of each vocabulary word

dist = np.sum(train_data_features, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

for tag, count in zip(vocab, dist):

    print(count, tag)
# Initialize a Random Forest classifier with 100 trees

rf = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

rf = rf.fit(train_data_features, train["sentiment"])



# Score

#rf.score(train_data_features, train["sentiment"])

#acc_random_forest = round(rf.score(train_data_features, train["sentiment"]) * 100, 2)



#acc_random_forest
def review_to_wordlist(review, remove_stopwords=False):

    # Function to convert a document to a sequence of words,

    # optionally removing stop words. Returns a list of words.

    

    # 1. Remove HTML

    review_text = BeautifulSoup(review).get_text()

      

    # 2. Remove non-letters

    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    

    # 3. Convert words to lower case and split them

    words = review_text.lower().split()

    

    # 4. Optionally remove stop words (false by default)

    if remove_stopwords:

        stops = set(stopwords.words("english"))

        words = [w for w in words if not w in stops]



    # 5. Return a list of words

    return(words)
# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



# Define a function to split a review into parsed sentences

def review_to_sentences( review, tokenizer, remove_stopwords=False ):

    # Function to split a review into parsed sentences. Returns a 

    # list of sentences, where each sentence is a list of words

    

    # 1. Use the NLTK tokenizer to split the paragraph into sentences

    raw_sentences = tokenizer.tokenize(review.strip())

    #

    # 2. Loop over each sentence

    sentences = []

    for raw_sentence in raw_sentences:

        # If a sentence is empty, skip it

        if len(raw_sentence) > 0:

            # Otherwise, call review_to_wordlist to get a list of words

            sentences.append( review_to_wordlist( raw_sentence, \

              remove_stopwords ))

    #

    # Return the list of sentences (each sentence is a list of words,

    # so this returns a list of lists

    return sentences
# Now we can apply this function to prepare our data for input to Word2Vec



sentences = []  # Initialize an empty list of sentences



print("Parsing sentences from training set")

for review in train["review"]:

    sentences += review_to_sentences(review, tokenizer)



print("Parsing sentences from unlabeled set")

for review in unlabeled_train["review"]:

    sentences += review_to_sentences(review, tokenizer)

len(sentences)
sentences[0]
# Import the built-in logging module and configure it so that Word2Vec creates nice output messages

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 40   # Minimum word count                        

num_workers = 4       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model (this will take some time)

print ("Training model...")

model = word2vec.Word2Vec(sentences, workers=num_workers, \

            size=num_features, min_count = min_word_count, \

            window = context, sample = downsampling)



# If you don't plan to train the model any further, calling 

# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)



# It can be helpful to create a meaningful model name and 

# save the model for later use. You can load it later using Word2Vec.load()

model_name = "300features_40minwords_10context"

model.save(model_name)

print("Model saved")
#The "doesnt_match" function will try to deduce which word in a set is most dissimilar from the others:

model.doesnt_match("man woman child kitchen".split())
model.most_similar("man")
model.most_similar("queen")
# Setting the minimum word count to 40 gave us a total vocabulary of 16,492 words with 300 features apiece

# model["flower"] # returns 1*300 numpy array
def makeFeatureVec(words, model, num_features):

    # Function to average all of the word vectors in a given paragraph



    # Pre-initialize an empty numpy array (for speed)

    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0



    # Index2word is a list that contains the names of the words in 

    # the model's vocabulary. Convert it to a set, for speed 

    index2word_set = set(model.wv.index2word)

    

    # Loop over each word in the review and, if it is in the model's

    # vocaublary, add its feature vector to the total

    for word in words:

        if word in index2word_set: 

            nwords = nwords + 1

            featureVec = np.add(featureVec,model[word])

            

    # Divide the result by the number of words to get the average

    featureVec = np.divide(featureVec,nwords)

    return featureVec
def getAvgFeatureVecs(reviews, model, num_features):

    # Given a set of reviews (each one a list of words), calculate 

    # the average feature vector for each one and return a 2D numpy array 

    

    # Initialize a counter

    counter = 0

    

    # Preallocate a 2D numpy array, for speed

    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")

    

    # Loop through the reviews

    for review in reviews:

        # Print a status message every 1000th review

        if counter%1000 == 0:

            print ("Review %d of %d" % (counter, len(reviews)))

        

       # Call the function (defined above) that makes average feature vectors

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

       

       # Increment the counter

        counter = counter + 1

    return reviewFeatureVecs
# Calculate average feature vectors for training and testing sets,

# using the functions we defined above. Notice that we now use stop word removal



clean_train_reviews = []

for review in train["review"]:

    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))



trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)



print("Creating average feature vecs for test reviews")

clean_test_reviews = []

for review in test["review"]:

    clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))



testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
# Fit a random forest to the training data, using 100 trees

from sklearn.ensemble import RandomForestClassifier

forest1 = RandomForestClassifier(n_estimators = 100)



print("Fitting a random forest to labeled training data...")

forest1 = forest1.fit(trainDataVecs, train["sentiment"])



# Test & extract results 

pred_f1 = forest1.predict(testDataVecs)
#accuracy_f1 = round(forest1.score(trainDataVecs, train["sentiment"]) * 100, 2)



#accuracy_f1
'''

# Write the test results 

output = pd.DataFrame(data={"id":test["id"], "sentiment":pred_f1})

output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

'''
from sklearn.cluster import KMeans

import time



start = time.time() # Start time



# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or avg of 5 words per cluster

word_vectors = model.wv.syn0 

num_clusters = int(word_vectors.shape[0] / 5)



# Initalize a k-means object and use it to extract centroids

kmeans_clustering = KMeans(n_clusters = num_clusters)

idx = kmeans_clustering.fit_predict(word_vectors)



# Get the end time and print how long the process took

end = time.time()

elapsed = end - start

print("Time taken for K Means clustering: ", elapsed, "seconds.")
# Create a Word / Index dictionary, mapping each vocabulary word to

# a cluster number                                                                                            

word_centroid_map = dict(zip(model.wv.index2word, idx))
# For the first 10 clusters

for cluster in range(0,10):

    # Print the cluster number  

    print("\nCluster %d" % cluster)

    

    # Find all of the words for that cluster number, and print them out

    words = [k for k, v in word_centroid_map.items() if v == cluster]

    print(words)
def create_bag_of_centroids(wordlist, word_centroid_map):

    #

    # The number of clusters is equal to the highest cluster index

    # in the word / centroid map

    num_centroids = max(word_centroid_map.values()) + 1

    #

    # Pre-allocate the bag of centroids vector (for speed)

    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    #

    # Loop over the words in the review. If the word is in the vocabulary,

    # find which cluster it belongs to, and increment that cluster count 

    # by one

    for word in wordlist:

        if word in word_centroid_map:

            index = word_centroid_map[word]

            bag_of_centroids[index] += 1

    #

    # Return the "bag of centroids"

    return bag_of_centroids
# Pre-allocate an array for the training set bags of centroids (for speed)

train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")



# Transform the training set reviews into bags of centroids

counter = 0

for review in clean_train_reviews:

    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)

    counter += 1



# Repeat for test reviews 

test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")



counter = 0

for review in clean_test_reviews:

    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)

    counter += 1
# Fit a random forest and extract predictions 

forest2 = RandomForestClassifier(n_estimators = 100)



# Fitting the forest may take a few minutes

print("Fitting a random forest to labeled training data...")

forest2 = forest2.fit(train_centroids,train["sentiment"])

pred_f2 = forest2.predict(test_centroids)
'''

# Write the test results 

output = pd.DataFrame(data={"id":test["id"], "sentiment":pred_f2})

output.to_csv("BagOfCentroids.csv", index=False, quoting=3)

'''
# Verify that there are 25,000 rows and 2 columns

test.shape
# EXPORTING PREDICTION FROM FIRST RANDOM FOREST MODEL



#Create an empty list and append the clean reviews one by one

num_reviews = len(test["review"])

clean_test_reviews = [] 



for i in range(0,num_reviews):

    if((i+1) % 1000 == 0):

        print("Review {} of {}".format(i+1, num_reviews))

    clean_review = review_to_words(test["review"][i])

    clean_test_reviews.append(clean_review)



# Get a bag of words for the test set, and convert to a numpy array

test_data_features = vectorizer.transform(clean_test_reviews)

test_data_features = test_data_features.toarray()



# Use the random forest to make sentiment label predictions

result = rf.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})



# Use pandas to write the comma-separated output file

output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

print("Output exported to csv!")