import nltk
#nltk.download()
data = "All work and no play makes jack a dull boy, all work and no play"
from nltk.tokenize import word_tokenize,sent_tokenize
print(word_tokenize(data))
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))
from nltk.corpus import stopwords
stopWords=set(stopwords.words('english'))
print(stopWords)
words=word_tokenize(data)
wordfilter=[]
for word in words:

    if word not in stopWords:

        wordfilter.append(word)
print(wordfilter)
words = ["game","gaming","gamed","games"]
from nltk.stem import PorterStemmer
ps=PorterStemmer()
for word in words:

    print(word,':',ps.stem(word))
sentence = "gaming, the gamers play games"

words = word_tokenize(sentence)

 

for word in words:

    print(word + ":" + ps.stem(word))
from nltk.tokenize import PunktSentenceTokenizer
document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
sentences = nltk.sent_tokenize(document)   

for sent in sentences:

    print(nltk.pos_tag(nltk.word_tokenize(sent)))
from nltk.corpus import state_union

document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'

sentences = nltk.sent_tokenize(document)   

 

data = []

for sent in sentences:

    data = data + nltk.pos_tag(nltk.word_tokenize(sent))

 

for word in data: 

    if 'NNP' in word[1]: 

        print(word)
import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import names
def gender_features(word): 

    return {'last_letter': word[-1]} 
# Load data and training 

name = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), g) for (n,g) in name] 

train_set = featuresets

classifier = nltk.NaiveBayesClassifier.train(train_set) 
#predict

print(classifier.classify(gender_features('James')))
#We start by defining 3 classes: positive, negative and neutral.

#Each of these is defined by a vocabulary:

positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]

negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]

neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
#Every word is converted into a feature using a simplified bag of words model:

def word_feats(words):

    return dict([(word, True) for word in words])


positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]

negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
#Our training set is then the sum of these three feature sets:

train_set = negative_features + positive_features + neutral_features
classifier=NaiveBayesClassifier.train(train_set)
# Predict

def Review(sentence):

    neg = 0

    pos = 0

    sentence = sentence.lower()

    words = sentence.split(' ')

    for word in words:

        classResult = classifier.classify( word_feats(word))

    if classResult == 'neg':

        neg = neg + 1

    if classResult == 'pos':

        pos = pos + 1

 

    Positive=(float(pos)/len(words))

    Negative=(float(neg)/len(words))

    

    if Positive>Negative:

        print('Positive'+ str(Positive))

    if Positive<Negative:

        print('Negative'+ str(Negative))

    if Positive==Negative:

        print('Mixed Review'+ str(Positive),str(Negative))

    
a='The movie was bad'
Review(a)