import nltk

nltk.corpus.gutenberg.fileids()
nltk.corpus.gutenberg.raw("shakespeare-hamlet.txt")
nltk.corpus.machado.fileids()
nltk.corpus.machado.raw("contos/macn001.txt")
nltk.corpus.nps_chat.fileids()
nltk.corpus.nps_chat.posts("11-06-adults_706posts.xml")
from nltk.corpus import brown

brown.categories()
brown.fileids()
from nltk import pos_tag



words = "He will race the car. When will the race start?".split(" ")



pos_tags = pos_tag(words)



print(pos_tags)
!python -m spacy download pt_core_news_sm
import spacy



from spacy.lang.pt import Portuguese



nlp = spacy.load('pt_core_news_sm')

tokens = nlp("Eles não vão trabalhar hoje para estudar para a prova. Espero que não seja em vão")

for t in tokens:

    print(t.text, t.pos_)
from nltk.corpus import wordnet as wn



wn.synsets('dog')
wn.synsets('dog', pos=wn.NOUN)
wn.synset('dog.n.01').definition()
wn.synset('dog.n.01').examples()
wn.synset('dog.n.01').lemma_names('por')
wn.synsets('cão', lang='por')
print("Hypernyms:")

print(wn.synset('dog.n.01').hypernyms())



print("Hyponyms:")

print(wn.synset('dog.n.01').hyponyms())



print("Holonyms")

print(wn.synset('dog.n.01').member_holonyms())
print("Hypernyms:")

print(wn.synset('canine.n.02').lemma_names())



print("Hyponyms:")

print(wn.synset('hunting_dog.n.01').lemma_names())



print("Holonyms")

print(wn.synset('canis.n.01').lemma_names())

dog = wn.synset('dog.n.01')

cat = wn.synset('cat.n.01')

fork = wn.synset('fork.n.01')

print(dog.path_similarity(cat))

print(dog.path_similarity(fork))
nlp = spacy.load('pt_core_news_sm')

tokens = nlp("O machado esquece, a arvore não.")

for t in tokens:

    for s in wn.synsets(t.text, lang='por'):

        print(s.lemma_names())