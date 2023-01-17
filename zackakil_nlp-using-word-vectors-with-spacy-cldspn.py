import spacy 

import numpy as np
import matplotlib.pyplot as plt

# this line makes any plots display in the notebook
%matplotlib inline
# YOUR CODE GOES HERE


# YORU CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


words = ['car', 'truck', 'dragon', 'data', 'horse', 'fish' , 'lion']

#YOUR CODE GOES HERE


sentence = 'why is the cat on the boat'

# numpy array with the dimensions (300,), filled with zeros
total = np.zeros(300)

# words from the text split into a list
words = sentence.split(' ')

# number of words in the sentence
n = len(words)

# the variable that the average word vector should be stored in 
average = None


# YOUR CODE GOES HERE




# YOUR CODE ENDS HERE


if average is not None:
    print(average.sum())
# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


sentence_to_compare = 'why is my cat on the car'

sentences = ["where did my dog go", 
             "dude where's my car",
             "i've lost my cat in the car",
             "get that boat back",
             "find my cat",
             "why is my dog on the drugs"]

# YOURE CODE GOES HERE


# import the list of stop words from the spacy library
from spacy.lang.en.stop_words import STOP_WORDS

def remove_stop_words(text):
    return ' '.join([word for word in text.split(' ') if word.lower() not in STOP_WORDS])


# YOUR CODE GOES HERE


# import the PCA module from sklearn
from sklearn.decomposition import PCA

# this is just making sure we have loaded in our word vectors
if 'nlp' not in locals():
    nlp = spacy.load('en_core_web_lg')

def get_word_vectors(words):
    # converts a list of words into their word vectors
    return [nlp(word).vector for word in words]

words = ['car', 'truck', 'dragon', 'data', 'horse', 'fish' , 'lion']

# intialise pca model and tell it to project data down onto 2 dimensions
pca = PCA(n_components=2)

# fit the pca model to our 300D data, this will work out which is the best 
# way to project the data down that will best maintain the relative distances 
# between data points. It will store these intructioons on how to transform the data.
pca.fit(get_word_vectors(words))

# Tell our (fitted) pca model to transform our 300D data down onto 2D using the 
# instructions it learnt during the fit phase.
word_vecs_2d = pca.transform(get_word_vectors(words))

# let's look at our new 2D word vectors
word_vecs_2d
# create a nice big plot 
plt.figure(figsize=(20,15))

# plot the scatter plot of where the words will be
plt.scatter(word_vecs_2d[:,0], word_vecs_2d[:,1])

# for each word and coordinate pair: draw the text on the plot
for word, coord in zip(words, word_vecs_2d):
    x, y = coord
    plt.text(x, y, word, size= 15)

# show the plot
plt.show()
# YOUR CODE GOES HERE 


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE

