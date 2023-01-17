

import spacy

nlp = spacy.load('en_core_web_sm')    #we are loading english ('en') core with size small ('sm')



# we have something called sent_tokenize also to tokenize sentences. Give it a try!

from nltk import word_tokenize



# stopwords are those words which are filtered out before NLP processing.

# can words like 'the', 'is' only be treated as stopwords? NO! Ever wondered how parental lock on sites work? It can have stop words like 'Sex', 'Porn', etc.

from nltk.corpus import stopwords     

import string

punctuations = string.punctuation
# NLTK implementation

sent = "The main challenge, is to start!"

stop = stopwords.words('english') + list(punctuations)    #removing unwanted punctuations also along with stopwords

print("NLTK implementation Result: ", [i for i in word_tokenize(sent) if i not in stop])



# Spacy implementation

doc = nlp(sent)    # Create a Doc object

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

print("Spacy implementation Result: ",[token.text for token in doc if token.text not in spacy_stopwords and token.text not in punctuations])
# NLTK implementation

from nltk.stem import PorterStemmer

porter = PorterStemmer()

print("NLTK implementation result: ",{"running": porter.stem("running"),"saw": porter.stem("saw"),"troubling": porter.stem("troubling")})



# Spacy implementation

# It might be surprising to you but spaCy doesn't contain any function for stemming (AFAIK) as it relies on lemmatization only! 
#NLTK implementation

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



print("NLTK implementation result: ",wordnet_lemmatizer.lemmatize('saw',pos='v'))



#Spacy implementation

from spacy.lemmatizer import Lemmatizer

from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)

lemmas = lemmatizer(u'saw', u'VERB')

print("Spacy implementation result: ", lemmas[0])
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [

    'This is the first document.',

    'This document is the second document.',

    'And this is the third one.',

    'Is this the first document?',

]

vectorizer = TfidfVectorizer(stop_words=stop) #stop was defined initially using stopwords from NLTK

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())

print(X)
import re                                  #regular expression

from collections import Counter            #creating frequency count dict

import heapq                               #for selecting n largest

import os
# we have uploaded a reference file which will provide access to correct spellings.

# you can have similar file based on the domain you are working in. 

# say for restaurant related domain, you can have cities name, dishes name, cuisines as part of this file.



os.listdir("../input/")
def words(text): return re.findall(r'\w+', text.lower())



WORDS = Counter(words(open('../input/big.txt').read()))

WORDS.most_common(10)
def P(word, N=sum(WORDS.values())): 

    "Probability of `word`."

    return WORDS[word] / N

	

def correction(word): 

    "Most probable spelling correction for word."

    listProb = {word: P(word) for word in candidates(word)}

    return listProb, max(candidates(word), key=P)

	

def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

	

def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)
def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)



def edits2(word): 

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def get_correct_word(word):

    corrected_word = next(iter(correction(word)[0]))

    print("Word passed: ", word, " Word corrected To: ", corrected_word)

    return corrected_word



print(get_correct_word('speling'))