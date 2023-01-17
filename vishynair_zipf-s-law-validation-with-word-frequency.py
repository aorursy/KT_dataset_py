#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Importing the String module
import string
from nltk import FreqDist
from nltk.corpus import stopwords
#To ignore warning messages
import warnings
warnings.filterwarnings('ignore')
#To display the full text column instead of truncating one
pd.set_option('display.max_colwidth', -1)
#Reading the train file
df = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")
#Head of the dataframe
df.head()
#Dimension of the dataframe
df.shape
#Python provides a constant called string.punctuation that provides a great list of punctuation characters. 
print(string.punctuation)
def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans('','',string.punctuation)
    return input_col.translate(table)
#Applying the remove_punctuation function
df['text'] = df['text'].apply(remove_punctuations)
def build_corpus(text_col):
    """To build a text corpus by stitching all the records together.Input the text column"""
    corpus = ""
    for sent in text_col:
        corpus += sent
    return corpus
#Building the corpus
corpus = build_corpus(df['text'])
#Converting all the words into lowercase
corpus = corpus.lower()
#Some part of the Text Corpus
corpus[:1000]
#Splitting the entire corpus
corpus = corpus.split()
#Observing the first few words
print(corpus[:50])
def plot_word_frequency(words,top_n=10):
    """Function to plot the word frequencies"""
    word_freq = FreqDist(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]
    plt.figure(figsize=(15,5))
    plt.title("Most Frequent Words in the Corpus - Including STOPWORDS")
    plt.ylabel("Count")
    plt.xlabel("Word")
    plot = sns.barplot(labels,counts)
    return plot
plot_word_frequency(corpus,20)
corpus_without_stop = [word for word in corpus if word not in stopwords.words("english")]
plot_word_frequency(corpus_without_stop,20)
#Creating a FreqDist object
fd=FreqDist()
#Creating ranks and frequencies
ranks = []
freqs = []
for i in corpus:
    fd[i] +=1
for rank,word in enumerate(fd):
    ranks.append(rank+1)
    freqs.append(fd[word])
#Plotting the distribution
plt.figure(figsize=(20,7))
plt.loglog(freqs,ranks)
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.title("Zipf's Distribution")
plt.show()