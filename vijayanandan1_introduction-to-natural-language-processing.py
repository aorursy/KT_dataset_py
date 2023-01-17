import nltk

from nltk.corpus import movie_reviews

# movie_reviews.readme()
raw = movie_reviews.raw()

print(raw[:3000])
print(raw[0])
corpus = movie_reviews.words()

print(corpus)
print(corpus[0])
freq_dist = nltk.FreqDist(corpus)

print(freq_dist)

print(freq_dist.most_common(50))

freq_dist.plot(50)
reviews = []

for i in range (0,len(movie_reviews.fileids())):

    reviews.append(movie_reviews.raw(movie_reviews.fileids()[i]))
print(reviews[0])
from nltk.tokenize import word_tokenize, sent_tokenize

sentences = nltk.sent_tokenize(reviews[0])

words = nltk.word_tokenize(reviews[0])

print(sentences[0])

print(words[0])
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

tokens = tokenizer.tokenize(reviews[0].lower())

print(tokens)
from nltk.corpus import stopwords

tokens = [token for token in tokens if token not in stopwords.words('english')]

print(tokens)
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()



test_word = "worrying"

word_stem = stemmer.stem(test_word)

word_lemmatise = lemmatizer.lemmatize(test_word)

word_lemmatise_verb = lemmatizer.lemmatize(test_word, pos="v")

word_lemmatise_adj = lemmatizer.lemmatize(test_word, pos="a")

print(word_stem, word_lemmatise, word_lemmatise_verb, word_lemmatise_adj)
# nltk.help.upenn_tagset()
from nltk import pos_tag

pos_tokens = nltk.pos_tag(tokens) 

print(pos_tokens)
corpus_tokens = tokenizer.tokenize(raw.lower())

vocab = sorted(set(corpus_tokens))
print("Tokens:", len(corpus_tokens))

print("Vocabulary:", len(vocab))