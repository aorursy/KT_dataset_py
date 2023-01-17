import nltk
#downloading the NLTK book for basic study

from nltk.book import *
#searching a particular word in the text:

#apply concordance function to do that in the text data you want to find.

text1.concordance('love')
#Finding  text/ words that is similar to a word

text1.similar('fear')
#texts shared by two different words:

text1.common_contexts(['love', 'world'])
#finding positions and frequency of words using dispersion plots in a text

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
#counting  the vocabulary:

len(text1)
#just taking out the unique values

len(sorted(set(text1)))
#counting number of times a word or text is repeated in the data

text3.count('every')
#freqency distributions of words in the text data:

fdist_1 = FreqDist(text1)

print(fdist_1)

print(fdist_1.most_common(20))

print(fdist_1['every'])
#plotting the frequency distribution plot

fdist_1.plot(50, cumulative = True)
#words occuring only once (hapaxes):

fdist_1.hapaxes()[1:10]
#bigrams: a n-grams with n = 2

list(nltk.bigrams(['for', 'love', 'wubba', 'lubba']))
#collations: collocations are essentially just frequent bigrams,

#except that we want to pay more attention to the cases that involve rare words

text4.collocations()
## let's dive further. The Text Corpora

#the large body og lingistic data

nltk.corpus.gutenberg.fileids()
#lest pick up our 1st text data emma

emma = nltk.corpus.gutenberg.words(nltk.corpus.gutenberg.fileids()[0])

len(emma)
for fileid in gutenberg.fileids():

    num_chars = len(gutenberg.raw(fileid)) 

    num_words = len(gutenberg.words(fileid))

    num_sents = len(gutenberg.sents(fileid))

    print(num_chars, num_words, num_words, fileid)
#web chat and texting corpora

from nltk.corpus import webtext

for fileid in webtext.fileids():

    print(fileid, webtext.raw(fileid)[:65], '\n')
#brown corpus

from nltk.corpus import brown

print(brown.categories())

print(len(brown.fileids()))
#reuters corpus

from nltk.corpus import reuters

print(reuters.fileids()[:10])

print(reuters.categories())
#inaugural address corpus

from nltk.corpus import inaugural

print(inaugural.fileids()[:10])
#cumulative frequency distribution for the inaugural corpus:

freq_dist = nltk.ConditionalFreqDist( 

    (target, fileid[:4])

    for fileid in inaugural.fileids()

    for w in inaugural.words(fileid)

    for target in ['america', 'citizen']

    if w.lower().startswith(target))

freq_dist.plot()
#word list corpus:

from nltk.corpus import words

print(words.words()[1547:1557])
#let's see the corpus for stopwords in english:

from nltk.corpus import stopwords

print(stopwords.words('english'))
#lookinh at some pronouncing dictionary

from nltk.corpus import cmudict

entries = cmudict.entries()

print(len(entries))

for entry in entries[78541:78549]:

    print(entry)
#wordNet: a semantically-oriented dictionary of English(thesaurus)

#finding a words synsets(the synonym sets)

from nltk.corpus import wordnet as wn

print(wn.synsets('table'),'\n')

#accessing different synonms in different sets:

for value in wn.synsets('table'):

    file = str(value)[8:-2]

    print(wn.synset(file).definition())

    print('example:', wn.synset(file).examples())

    print(wn.synset(file).lemma_names(), '\n')
#hyponyms using wordNet

file = str(wn.synsets('motorcar')[0])[8:-2]

car_1 = wn.synset(file)

types_car_1 = car_1.hyponyms()

print(types_car_1[0],'\n')

print(sorted(lemma.name() for synset in types_car_1 for lemma in synset.lemmas()))
import re

#detecting word patterns

#words ending with

wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]

w = ['played', 'character','devoted', 'purple']

print([word for word in wordlist if re.search('rley$', word)])
#words inbetween and we know the length of the text

w = ['abjectly', 'adjuster', 'detected', 'dejectly', 'injector', 'majestic']

print([word for word in wordlist if re.search('^..j..t..$', word)])
#ranges and closures of words

print([w for w in wordlist if re.search('^[ivor][esiy][opes][sexy][rac]$', w)])
#extracting the word pieces

word = 'whatsupyo!what are you even doing??'

print(re.findall(r'[aeiou]', word))
import nltk

example = sorted(set(nltk.corpus.treebank.words()))

freq_dist = nltk.FreqDist(freq for word in example

                          for freq in re.findall(r'[aeiou]{2,}',word))

print(freq_dist.most_common(12))
example = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'

def compress_words(word):

    pieces = re.findall(example, word)

    return ''.join(pieces)

eng_udhr = nltk.corpus.udhr.words('English-Latin1')

print(nltk.tokenwrap(compress_words(w) for w in eng_udhr[:25]))
#searching tokenized texts

#let's find all the data between 'a' and 'man'

from nltk.corpus import gutenberg, nps_chat

moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))

moby.findall(r"<a> (<.*>) <man>")
# normalizing a text

raw = 'penelope: is this the cowardly man? That used to strangle every piece if information running in the servers, that was so gross, spooky and artificial'

tokens = nltk.word_tokenize(raw)
#stemming the tokens:

porter = nltk.PorterStemmer()

lancaster = nltk.LancasterStemmer()

print([porter.stem(t) for t in tokens],'\n')

print([lancaster.stem(t) for t in tokens])
#as you can see it stemming is not that better

#lemmatization

word_netlemmatizer = nltk.WordNetLemmatizer()

print([word_netlemmatizer.lemmatize(t) for t in tokens])
#segmentation:

#sentence segmentation: dividing the whole rawdata into sentences

text = nltk.corpus.gutenberg.raw('chesterton-thursday.txt')

sentences = nltk.sent_tokenize(text)

print(sentences[5:8])
#NetworkX visualisation of wordnet(used for graphs):

import networkx as nx

import matplotlib.pyplot as plt

from nltk.corpus import wordnet as wn

def traverse(graph, start, node):

    graph.depth[node.name] = node.shortest_path_distance(start)

    for child in node.hyponyms():

        graph.add_edge(node.name, child.name)

        traverse(graph, start, child)

def hyponym_graph(start):

    G = nx.Graph()

    G.depth = {}

    traverse(G, start, start)

    return G

def graph_draw(graph):

    plt.figure(figsize = (12,5))

    nx.draw(graph,

         node_size = [16 * graph.degree(n) for n in graph],

         node_color = [graph.depth[n] for n in graph],

         with_labels = False)

    plt.show()

dog = wn.synset('dog.n.01')

graph = hyponym_graph(dog)

graph_draw(graph)
#POS tagging

text = nltk.word_tokenize('I hate you because you are making me not me')

print(nltk.pos_tag(text), '\n')

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

print(text.similar('boy'))
#automatic tagging

from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories = 'news')

brown_sents = brown.sents(categories = 'news')
#1. the default tagger

raw = 'I do not like green eggs and ham, I do not like them Sam I am!'

tokens = nltk.word_tokenize(raw)

default_tagger = nltk.DefaultTagger('NN')

print(default_tagger.tag(tokens))
#2.regular expression tagger



patterns = [

    (r'.*ing$', 'VBG'),                # gerunds

    (r'.*ed$', 'VBD'),                 # simple past

    (r'.*es$', 'VBZ'),                 # 3rd singular present

    (r'.*ould$', 'MD'),                # modals

    (r'.*\'s$', 'NN$'),                # possessive nouns

    (r'.*s$', 'NNS'),                  # plural nouns

    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers

    (r'.*', 'NN')]                   # nouns (default)



regexp_tagger = nltk.RegexpTagger(patterns)

print(regexp_tagger.tag(brown_sents[3]))
# N- gram tagger

#1. unigram tagging

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)

print(unigram_tagger.tag(brown_sents[2007]))