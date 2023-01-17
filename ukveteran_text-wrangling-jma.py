import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
text = 'The quick brown fox jumped over The Big Dog'
text
text.lower()
text.upper()
text.title()
sample_text = ("US unveils world's most powerful supercomputer, beats China. " 
               "The US has unveiled the world's most powerful supercomputer called 'Summit', " 
               "beating the previous record-holder China's Sunway TaihuLight. With a peak performance "
               "of 200,000 trillion calculations per second, it is over twice as fast as Sunway TaihuLight, "
               "which is capable of 93,000 trillion calculations per second. Summit has 4,608 servers, "
               "which reportedly take up the size of two tennis courts.")
sample_text
import nltk

nltk.sent_tokenize(sample_text)
print(nltk.word_tokenize(sample_text))
import spacy
nlp = spacy.load('en')

text_spacy = nlp(sample_text)
[obj.text for obj in text_spacy.sents]
print([obj.text for obj in text_spacy])
import requests

data = requests.get('http://www.gutenberg.org/cache/epub/8001/pg8001.html')
content = data.text
print(content[2745:3948])
import re
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

clean_content = strip_html_tags(content)
print(clean_content[1163:1957])
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
s = 'Sómě Áccěntěd těxt'
s
remove_accented_chars(s)
import re

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text
s = "Well this was fun! See you at 7:30, What do you think!!? #$@@9318@ 🙂🙂🙂"
s
remove_special_characters(s, remove_digits=True)
remove_special_characters(s)
from nltk.stem import PorterStemmer
ps = PorterStemmer()

ps.stem('jumping'), ps.stem('jumps'), ps.stem('jumped')
ps.stem('lying')
ps.stem('strange')
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
help(wnl.lemmatize)
print(wnl.lemmatize('cars', 'n'))
print(wnl.lemmatize('boxes', 'n'))
print(wnl.lemmatize('running', 'v'))
print(wnl.lemmatize('ate', 'v'))
print(wnl.lemmatize('saddest', 'a'))
print(wnl.lemmatize('fancier', 'a'))
print(wnl.lemmatize('ate', 'n'))
print(wnl.lemmatize('fancier', 'v'))
print(wnl.lemmatize('fancier'))
import spacy
nlp = spacy.load('en', parse=False, tag=False, entity=False)

def spacy_lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
s
spacy_lemmatize_text(s)
def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words[:10])
s
remove_stopwords(s, is_lower_case=False)