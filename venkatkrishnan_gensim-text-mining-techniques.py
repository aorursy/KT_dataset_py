import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', -1)

# dataset
from sklearn.datasets import fetch_20newsgroups

# Gensim packages
from gensim.parsing import strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
from gensim.parsing import preprocess_string


# loading dataset
news_group = fetch_20newsgroups(subset='train')

news_group_data = news_group.data
news_group_target_names = news_group.target_names
news_group_target = news_group.target
# Creating a dataframe from the loaded data
news_df = pd.DataFrame({'news': news_group_data, 
                        'class': news_group_target})
news_extracts = news_df.sample(2000)

news_extracts.reset_index(drop=True, inplace=True)
news_extracts.head(2)
# Custom filter method
transform_to_lower = lambda s: s.lower()

remove_single_char = lambda s: re.sub(r'\s+\w{1}\s+', '', s)

# Filters to be executed in pipeline
CLEAN_FILTERS = [strip_tags,
                strip_numeric,
                strip_punctuation, 
                strip_multiple_whitespaces, 
                transform_to_lower,
                remove_stopwords,
                remove_single_char]

# Method does the filtering of all the unrelevant text elements
def cleaning_pipe(document):
    # Invoking gensim.parsing.preprocess_string method with set of filters
    processed_words = preprocess_string(document, CLEAN_FILTERS)
    
    return processed_words
# Apply the cleaning pipe on the news data

news_extracts['clean_text'] = news_extracts['news'].apply(cleaning_pipe)
news_extracts['clean_text'][0:2]
# import stemmer from gensim
from gensim import parsing
from gensim.parsing.porter import PorterStemmer
from gensim.summarization import textcleaner

# Initialize PorterStemmer
porter = PorterStemmer()

def basic_stemming(text):
    return parsing.stem_text(text)

# Stem the incoming word
def get_stemword(stemmer, word):    
    return stemmer.stem(word)
# stem all the words in the passed sentence
def get_stem_sentence(stemmer, sentence):
    return stemmer.stem_sentence(sentence)

# stem all the sentences given as a document
def get_stem_documents(stemmer, document):
    return stemmer.stem_documents(document)


document = """A computer is a machine that can be instructed to carry out sequences of arithmetic or logical operations automatically via computer programming. 
Modern computers have the ability to follow generalized sets of operations, called programs. 
These programs enable computers to perform an extremely wide range of tasks. 
A complete computer including the hardware, the operating system (main software), and peripheral equipment required and used for full operation can be referred to as a computer system. 
This term may as well be used for a group of computers that are connected and work together, in particular a computer network or computer cluster."""
# Stem the given paragraph text 
stemmed_text = basic_stemming(document)

print(stemmed_text)
# Break the paragraph into sentences
sentences = textcleaner.get_sentences(document)

# Sentences will be parsed by stem method
stem_doc = get_stem_documents(porter, sentences)

print(stem_doc)
