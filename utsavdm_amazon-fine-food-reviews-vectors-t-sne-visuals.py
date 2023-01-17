# importing libraries:

import sqlite3                      # to save/ load the .sqlite files and perform SQL operations

import pandas as pd                 # dataframe ops

import numpy as np                  # array ops

from IPython.display import display # to view dataframe in a tabular format



# creating the connect object to connect with the database:

con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')



# visualizing the table:

dataset = pd.read_sql("""

SELECT

    *

FROM

    Reviews;

""", con)



# let us only keep the rows where 'Score' is either greater than 3 or less than 3.

# The 'Score' equal to 3 would mean neutral reviews and can be tricky to classify it as a negative/ positive review:

dataset = pd.read_sql("""

SELECT

    *

FROM 

    Reviews

WHERE

    Score <> 3;

""", con)



# Also, let us change the 'Score' from numbers to ratings as follows:

# if Score > 3 then 'Positive' rating

# if Score < 3 then 'Negative' rating and eliminate the rows where Score = 3, for simplicity:

def rate(x):

    if x < 3:

        return "Negative"

    return "Positive"



# replacing the numbers in the Score column with "Positive"/ "Negative" values:

positive_negative = dataset['Score']

positive_negative = positive_negative.map(rate)

dataset['Score'] = positive_negative

display(dataset)
# ignoring 'Id' column as it has unique values throughout the table:

duplicates = pd.read_sql("""

SELECT

    UserId,

    ProfileName,

    Time,

    Text,

    count(*)

FROM

    Reviews

GROUP BY

    UserId,

    ProfileName,

    Time,

    Text

HAVING

    count(*) > 1;

""", con)



con.close # closing the connection

print("Duplicates:\n")

display(duplicates)
# here, we sort ProductID and then remove the duplicates except the first occurrence.

sorted_data = dataset.sort_values('ProductId', axis=0, inplace=False, ascending=True)



# here - 'UserId', 'ProfileName', 'Time' & 'Text' get concatenated and all the rows (except the first occurence) are dropped

# where this concatenated record is repeated.

final_dataset = sorted_data.drop_duplicates(['UserId', 'ProfileName', 'Time', 'Text'], inplace=False, keep='first')

print("Duplicates eliminated!")
# only keeping the rows where helpfulness numerator is less than or equal to the helpfulness denominator:

final_dataset = final_dataset[final_dataset.HelpfulnessNumerator <= final_dataset.HelpfulnessDenominator]



# resetting the index because many of the rows are deleted and their corresponding indices are missing.

# drop=True means to drop/ delete the exisiting indices:

final_dataset.reset_index(drop=True, inplace=True)

display(final_dataset)
# randomly sampling 5k rows:

sampled_data = final_dataset.sample(n=5000, random_state=0)



# resetting the index as we have sampled random rows:

sampled_data.reset_index(drop=True)



# creating 2 obejects to store the sampled values of 'Text' & 'Score'

sampled_rows = sampled_data['Text']

sampled_label = sampled_data['Score']



# reshaping the sampled_label

sampled_label = sampled_label.values.reshape(5000,1)



print("Sampling performed successfully, size of the sample is {} rows.".format(sampled_data.shape[0]))
# importing CountVectorizer from sklearn to implement Bag of Words & bi-gram/ n-gram:

# importing TfidfVectorizer from sklearn to implement TF-IDF: to implement Bag of Words:

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# We apply these algorithms on the 'Text' column (which is the raw text and not the cleaned/ processed text) 

# in the dataframe 'final_dataset':



# Also, we'll skip the words that occur less than 5 times in the whole set of reviews/ documents i.e. min_df=5



# Bag of Words (BoW):

bow_model = CountVectorizer(min_df=5)   # default ngram_range = 1 i.e. uni-gram

bow_vector = bow_model.fit_transform(sampled_rows)

print("Shape of the vector obtained by using Bag of Words: ", bow_vector.shape)



# bi-gram/ n-grams:

gram_model = CountVectorizer(ngram_range=(1,2), min_df=5)     # calculating both uni-gram & bi-grams

gram_vector = gram_model.fit_transform(sampled_rows)

print("Shape of the vector obtained by using uni-gram & bi-grams: ", gram_vector.shape)



# TF-IDF:

tfidf_model = TfidfVectorizer(ngram_range=(1,2), min_df=5)     # calculating both uni-gram & bi-grams

tfidf_vector = tfidf_model.fit_transform(sampled_rows)

print("Shape of the vector obtained by using TF-IDF (uni-gram & bi-grams): ", tfidf_vector.shape)
# import statements:

import re   # to search for html tags, punctuations & special characters



# importing gensim.models to implement Word2Vec:

import gensim

from gensim import models

from gensim.models import Word2Vec

from gensim.models import KeyedVectors



# Remove HTML tags - getting all the HTML tags and replacing them with blank spaces:

def cleanhtml(sentence):

    clean_text = re.sub('<.*?>', ' ', sentence)

    return clean_text



# Remove punctuations & special characters - getting all the punctuations and replacing them with blank spaces:

def cleanpunc(sentence):

    clean_text = re.sub(r'[@#$%\^&\*+=]', r'', sentence) # removing special characters

    clean_text = re.sub(r'[,.;\'"\-\!?:\\/|\[\]{}()]', r' ', clean_text) # removing punctuations

    return clean_text



final_clean_sentences = []



for sentence in sampled_data['Text'].values:

    sentence = cleanhtml(sentence)

    sentence = cleanpunc(sentence)

    clean_sentence = []

    

    # for each word in the sentence, if it is alphabetic, we append it to the new list

    for word in sentence.split():

        if word.isalpha():

            clean_sentence.append(word.lower())

    

    # for each review in the 'Text' column, we create a list of words that appear in that sentence and store it in another list. 

    # basically, a list of lists - because that's how the model takes the input while training:

    final_clean_sentences.append(clean_sentence)

    

print("Sentence cleaning completed")
# training the model:

my_w2v_model = gensim.models.Word2Vec(final_clean_sentences, min_count=5, size=30, workers=4, iter=8)

print("Custom-built model has completed the training!")
# testing the Custom-built model:

print("Custom-built model results:")

print("*"*30)

print('Words most similar to "book":\n', my_w2v_model.wv.most_similar(positive='book'))

print("-"*80)

print('Words opposite to "tasty":\n', my_w2v_model.wv.most_similar(negative='tasty'))
# below imports are used only to show the progress bar:

import tqdm

import time

from tqdm import notebook



vectored_sentences = []

not_converted = set()  # used to keep track of how many words are not converted to vector

print("Average Word2Vec calculations started...")



# its ok to ignore tqdm_notebook in the for loop below, just kept it for ETA:

for sentence in notebook.tqdm(final_clean_sentences):

    vec_sent = np.zeros(30)        # since the size of custom-built model is 30

    for word in sentence:

        if word in my_w2v_model:

            vectored_word = my_w2v_model.wv[word]

            vec_sent += vectored_word

        else:

             not_converted.add(word)

                

    vec_sent/=len(sentence)

    vectored_sentences.append(vec_sent)



print("\nAvg. Word2Vec calculations completed!")

print("First couple of elements of Avg. Word2Vec vectored sentences:\n", vectored_sentences[0:2])

print("-"*60)

print("***Info. - There were {} words that were NOT converted to vector because their occurrences were less than the 'min_count' value provided during training.".format(len(not_converted)))
# we'll store the TF values & IDF values so that we can use them directly in the for loop:

# we can also calculate them in the for loop but it takes too long:

tfidf_dict = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))



# TF-IDF-Word2Vec calculation using the custom-built model:

tfidf_w2v_vectors = []

tfidf_features = tfidf_model.get_feature_names()

row = 0

not_converted = set()

print("TF-IDF weighted Word2Vec calculations started...")



# its ok to ignore tqdm_notebook in the for loop below, just kept it for ETA:

for sentence in notebook.tqdm(final_clean_sentences):

    vec_sent = np.zeros(30)       # since the size of custom-built model is 20

    sum_tfidf_val = 0

    sum_tfidf_w2v = 0

    sentence_length = len(sentence)

    

    for word in sentence:

        if word in my_w2v_model and word in tfidf_dict.keys():

            tfidf_val = tfidf_dict[word] * (sentence.count(word) / sentence_length)

            vectored_word = my_w2v_model[word]

            prod = tfidf_val * vectored_word

            sum_tfidf_w2v+=prod

            sum_tfidf_val+=tfidf_val

        else:

             not_converted.add(word)

        

    try:

        tfidf_w2v_vectors.append(sum_tfidf_w2v / sum_tfidf_val)

    except:

        pass

    row+=1

    

print("\nTF-IDF weighted Word2Vec calculations completed!")

print("First couple of elements of TF-IDF weighted Word2Vec vectored sentences:\n", tfidf_w2v_vectors[0:2])

print("-"*60)

print("***Info. - There were {} words that were NOT converted to vector because their occurrences were less than the 'min_count' value provided during training.".format(len(not_converted)))
# import statements:

import re       # to search for html tags, punctuations & special characters

import nltk     # to import stopwords & SnowballStemmer

from nltk.corpus import stopwords       # to get all the stop words (words that don't add meaning to the sentences)

from nltk.stem import SnowballStemmer   # to get the stem words of the similar words



# 1. Removing the noise (HTML/ XML tags, punctuations & special characters)

def cleanhtml(sentence):

    clean_text = re.sub('<.*?>', ' ', sentence) # removing HTML/ XML tags

    return clean_text



def cleanpunc(sentence):

    clean_text = re.sub(r'[@#$%\^&\*+=]', r'', sentence) # removing special characters

    clean_text = re.sub(r'[,.;\'"\-\!?:\\/|\[\]{}()]', r' ', clean_text) # removing punctuations

    #print("In func:", clean_text)

    return clean_text



# to view all the stop words in English language:

print("All the stopwords in English language:\n", (stopwords.words('english')))



# 3. Remove the Stopwords - storing all the English stopwords:

all_stopwords = set(stopwords.words('english'))     # getting the full set of stopwords in English language



# 4. Perform Stemming:

snow_stemmer = SnowballStemmer('english')



row = 0

stemmed_sentence = ''

final_stemmed_text = []

all_positive_words = []

all_negative_words = []



# its ok to ignore tqdm_notebook in the for loop below, just kept it for ETA:

# for each sentence/ review in the 'Text' column, we remove HTML tags, puctuations & special characters:

for sentence in notebook.tqdm(sampled_data['Text'].values):

    stemmed_list = []

    sentence = cleanhtml(sentence)  # removing the HTML tags 

    sentence = cleanpunc(sentence)  # removing the punctuations & special characters



    # for each word in the sentence, we make sure that:

    # the length of word > 2 AND they are ALPHABETIC and are converted to lower case

    for word in sentence.split():

        

        # 2. Words should be alphanumeric & length of words > 2

        if len(word)>2 and word.isalpha() and word.lower() not in all_stopwords:

                

                # 4. perform stemming, convert to lower case & encode using standard format 'utf8'

                stemmed_word = snow_stemmer.stem(word.lower()).encode('utf8')  

                stemmed_list.append(stemmed_word)

                

    # creating the sentence out of the stemmed words:

    stemmed_sentence = " ".join(str(stemmed_list))



    # storing all the words that resulted in the positive OR negative score in 2 different lists:

    if final_dataset['Score'][row].lower() == 'positive':

        all_positive_words.append(i for i in stemmed_sentence)     # list of all words that resulted in the POSITIVE review/ score

    if final_dataset['Score'][row].lower() == 'negative':

        all_negative_words.append(i for i in stemmed_sentence)     # list of all words that resulted in the NEGATIVE review/ score



    # appending the sentences to a new list that stores the processed/ cleaned sentences:

    final_stemmed_text.append(stemmed_sentence)

    row+=1



print("No. of words that make up positive reviews in the sampled dataset: ", len(all_positive_words))

print("No. of words that make up negative reviews in the sampled dataset: ", len(all_negative_words))
# the import stataments:

from sklearn.preprocessing import StandardScaler # to standardize the data

from sklearn.manifold import TSNE  # to create the TSNE model

import seaborn as sn  # to plot the visualization

import matplotlib.pyplot as plt  # to plot the visualization



# creating the TSNE model:

tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, random_state=0)

print("t-SNE model created successfully")
# Standardizing the data:

std_scaler = StandardScaler(with_mean=False)

std_bow_dataset = std_scaler.fit_transform(bow_vector)



# to apply the TSNE model, it requires dense matrix, but the one we have (std_bow_dataset) is a sparse matrix:

print(type(std_bow_dataset))

std_bow_dataset_dense = std_bow_dataset.todense()

print("Converted the sparse matrix to dense matrix")



# applying the TSNE model:

print("Applying the t-SNE model...")

tsne_bow_dataset = tsne_model.fit_transform(std_bow_dataset_dense)

print("t-SNE dataset prepared!")



# appending the 'Score' to the obtained TSNE dataset:

tsne_bow_dataset = np.hstack((tsne_bow_dataset, sampled_label))



# creating the dataframe for the plot:

tsne_bow_dataframe = pd.DataFrame(tsne_bow_dataset, columns=['Dimension 1','Dimension 2','Score'])



# plotting the t-SNE for BoW vectors:

sn.FacetGrid(tsne_bow_dataframe, hue='Score', height=8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.title('Bag of Words (uni-gram) vectors')

plt.show()
# Standardizing the data:

std_gram_dataset = std_scaler.fit_transform(gram_vector)



# to apply the TSNE model, it requires dense matrix, but the one we have (std_gram_dataset) is a sparse matrix:

print(type(std_gram_dataset))

std_gram_dataset_dense = std_gram_dataset.todense()

print("Converted the sparse matrix to dense matrix")



# applying the TSNE model:

print("Applying the t-SNE model...")

tsne_gram_dataset = tsne_model.fit_transform(std_gram_dataset_dense)

print("t-SNE dataset prepared!")



# appending the 'Score' to the obtained TSNE dataset:

tsne_gram_dataset = np.hstack((tsne_gram_dataset, sampled_label))



# creating the dataframe for the plot:

tsne_gram_dataframe = pd.DataFrame(tsne_gram_dataset, columns=['Dimension 1','Dimension 2','Score'])



# plotting the t-SNE:

sn.FacetGrid(tsne_gram_dataframe, hue='Score', height=8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.title('bi-gram vectors')

plt.show()
# Standardizing the data:

std_tfidf_dataset = std_scaler.fit_transform(tfidf_vector)



# to apply the TSNE model, it requires dense matrix, but the one we have (std_tfidf_dataset) is a sparse matrix:

print(type(std_tfidf_dataset))

std_tfidf_dataset_dense = std_tfidf_dataset.todense()

print("Converted the sparse matrix to dense matrix")



# applying the TSNE model:

print("Applying the t-SNE model...")

tsne_tfidf_dataset = tsne_model.fit_transform(std_tfidf_dataset_dense)

print("t-SNE dataset prepared!")



# appending the 'Score' to the obtained TSNE dataset:

tsne_tfidf_dataset = np.hstack((tsne_tfidf_dataset, sampled_label))



# creating the dataframe for the plot:

tsne_tfidf_dataframe = pd.DataFrame(tsne_tfidf_dataset, columns=['Dimension 1','Dimension 2','Score'])



# plotting the t-SNE:

sn.FacetGrid(tsne_tfidf_dataframe, hue='Score', height=8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.title('TF-IDF vectors')

plt.show()
# Standardizing the data:

std_w2v_dataset = std_scaler.fit_transform(vectored_sentences)



# applying the TSNE model:

print("Applying the t-SNE model...")

tsne_w2v_dataset = tsne_model.fit_transform(std_w2v_dataset)

print("t-SNE dataset prepared!")



# appending the 'Score' to the obtained TSNE dataset:

tsne_w2v_dataset = np.hstack((tsne_w2v_dataset, sampled_label))



# creating the dataframe for the plot:

tsne_w2v_dataframe = pd.DataFrame(tsne_w2v_dataset, columns=['Dimension 1','Dimension 2','Score'])



# plotting the t-SNE:

sn.FacetGrid(tsne_w2v_dataframe, hue='Score', height=8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.title('Average weighted Word2Vec')

plt.show()
# Standardizing the data:

std_ifidf_w2v_dataset = std_scaler.fit_transform(tfidf_w2v_vectors)



# applying the TSNE model:

print("Applying the t-SNE model...")

tsne_ifidf_w2v_dataset = tsne_model.fit_transform(std_ifidf_w2v_dataset)

print("t-SNE dataset prepared!")



# appending the 'Score' to the obtained TSNE dataset:

tsne_ifidf_w2v_dataset = np.hstack((tsne_ifidf_w2v_dataset, sampled_label))



# creating the dataframe for the plot:

tsne_ifidf_w2v_dataframe = pd.DataFrame(tsne_ifidf_w2v_dataset, columns=['Dimension 1','Dimension 2','Score'])



# plotting the t-SNE:

sn.FacetGrid(tsne_ifidf_w2v_dataframe, hue='Score', height=8).map(plt.scatter, 'Dimension 1', 'Dimension 2').add_legend()

plt.title('TF-IDF weighted Word2Vec')

plt.show()