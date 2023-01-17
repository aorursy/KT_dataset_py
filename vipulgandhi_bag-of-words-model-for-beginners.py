from sklearn.feature_extraction.text import CountVectorizer

# Multiple documents

text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"] 

# create the transform

vectorizer = CountVectorizer()

# tokenize and build vocab

vectorizer.fit(text)

# summarize

print(sorted(vectorizer.vocabulary_))
# encode document

vector = vectorizer.transform(text)

# summarize encoded vector

print(vector.shape)

print(vector.toarray())
# encode another document

text2 = ["the the the times"]

vector = vectorizer.transform(text2)

print(vector.toarray())
from sklearn.feature_extraction.text import TfidfVectorizer

# list of text documents

text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]

# create the transform

vectorizer = TfidfVectorizer()

# tokenize and build vocab

vectorizer.fit(text)

# summarize

print(sorted(vectorizer.vocabulary_))

# encode document

vector = vectorizer.transform([text[0]])

print(vectorizer.idf_)
# summarize encoded vector

print(vector.shape)

print(vector.toarray())
from sklearn.feature_extraction.text import HashingVectorizer

# list of text documents

text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]

# create the transform small number of "n_features"  may result in hash collisions

vectorizer = HashingVectorizer(n_features=6)

# encode document

vector = vectorizer.transform(text)

# summarize encoded vector

print(vector.shape)

print(vector.toarray())
from keras.preprocessing.text import text_to_word_sequence

# define the document

# text = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"]

text = 'The quick brown fox jumped over the lazy dog.'

# tokenize the document

result = text_to_word_sequence(text)

print(result)
from keras.preprocessing.text import hashing_trick



text = 'The quick brown fox jumped over the lazy dog.'

# estimate the size of the vocabulary

words = set(text_to_word_sequence(text))

vocab_size = len(words)

print(vocab_size)

# integer encode the document

result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')

print(result)
from keras.preprocessing.text import Tokenizer # define 5 documents

docs = ["It was the best of times", "it was the worst of times", "it was the age of wisdom", "it was the age of foolishness"] 

# create the tokenizer

tokenizer = Tokenizer()

# fit the tokenizer on the documents

tokenizer.fit_on_texts(docs)
tokenizer.word_counts, tokenizer.document_count, tokenizer.word_index, tokenizer.word_docs
encoded_docs = tokenizer.texts_to_matrix(docs, mode='count')

encoded_docs