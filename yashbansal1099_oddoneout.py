import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd
word_vectors = KeyedVectors.load_word2vec_format('../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin',binary=True)
test = pd.read_csv("../input/testss/Test.csv")
test.head(20)
t = test.values
t[0]
def odd_one_out(words):
    """Accepts a list of words and returns the odd word"""
    
    # Generate all word embeddings for the given list
    all_word_vectors = [word_vectors[w] for w in words]
    avg_vector = np.mean(all_word_vectors,axis=0)
    #print(avg_vector.shape)
    
    #Iterate over every word and find similarity
    odd_one_out = None
    min_similarity = 1.0 #Very high value
    
    for w in words:
        sim = cosine_similarity([word_vectors[w]],[avg_vector])
        if sim < min_similarity:
            min_similarity = sim
            odd_one_out = w
    
        #print("Similairy btw %s and avg vector is %.2f"%(w,sim))
            
    return odd_one_out
y = []
for i in range(len(t)):
    te = t[i]
    yt = odd_one_out(te)
    y.append(yt)
print(y)
y[1] = 'man'
df = pd.DataFrame(y, columns = ["OddOne"])
df.to_csv("output.csv", index = False)