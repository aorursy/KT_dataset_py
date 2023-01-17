from nltk.corpus import brown
import nltk

print ('Total Categories:'), len(brown.categories())

print (brown.categories())

# tokenized sentences
brown.sents(categories='mystery')

# POS tagged sentences
brown.tagged_sents(categories='mystery')

# get sentences in natural form
sentences = brown.sents(categories='mystery')

# get tagged words
tagged_words = brown.tagged_words(categories='mystery')

# get nouns from tagged words
nouns = [(word, tag) for word, tag in tagged_words if any(noun_tag in tag for noun_tag in ['NP', 'NN'])]

print (nouns[0:10]) # prints the first 10 nouns

# build frequency distribution for nouns
nouns_freq = nltk.FreqDist([word for word, tag in nouns])

# print top 10 occuring nouns
print (nouns_freq.most_common(10))