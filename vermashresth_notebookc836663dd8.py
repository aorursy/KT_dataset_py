import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from nltk.corpus import stopwords
ex="Hello there, how you doing? Im awesome"
a=(sent_tokenize(ex))

stop=set(stopwords.words("english"))

b=[i for i in a if not i in stop]

print(b)
print(word_tokenize(ex))