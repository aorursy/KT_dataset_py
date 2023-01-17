import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import sent_tokenize, word_tokenize 
import gensim 
from gensim.models import Word2Vec 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#  Reads ‘alice.txt’ file 
import pandas as pd
df = pd.read_csv("/kaggle/input/grimms-fairy-tales/grimms_fairytales.csv") 
f = df['Text'][1]
df.head()
tokenized_text = []
# iterate through each sentence in the file 
for sentence in sent_tokenize(f): 
    sentence_tokenized = [] 
      
    # tokenize the sentence into words 
    for word in word_tokenize(sentence): 
        sentence_tokenized.append(word.lower()) 
  
    tokenized_text.append(sentence_tokenized) 
strings = ['hans','luck']
models_quantity = 4
sizes = []
windows = []
for i in range(1,models_quantity+1):
    sizes.append(i*100)
    windows.append(i*5)
import matplotlib.pyplot as plt
cbow_fixed_size_results = []
for i in range(0,models_quantity):
    cbow_model = gensim.models.Word2Vec(tokenized_text, min_count = 1,  size = sizes[0], window = windows[i])
    cbow_fixed_size_results.append(cbow_model.wv.similarity(strings[0], strings[1]))

plt.plot(windows, cbow_fixed_size_results)
plt.ylabel('cosine similarity')
plt.xlabel('windows')
plt.show()

cbow_fixed_window_results = []
for i in range(0,models_quantity):
    cbow_model = gensim.models.Word2Vec(tokenized_text, min_count = 1,  size = sizes[i], window = windows[0])
    cbow_fixed_window_results.append(cbow_model.wv.similarity(strings[0], strings[1]))

plt.plot(sizes, cbow_fixed_window_results)
plt.ylabel('cosine similarity')
plt.xlabel('sizes')
plt.show()
# Create Skip Gram model 
skip_gram_fixed_size_results = []
for i in range(0,models_quantity):
    skip_gram_model = gensim.models.Word2Vec(tokenized_text, min_count = 1,  size = sizes[0], window = windows[i], sg = 1)
    skip_gram_fixed_size_results.append(skip_gram_model.wv.similarity(strings[0], strings[1]))

plt.plot(windows, skip_gram_fixed_size_results)
plt.ylabel('cosine similarity')
plt.xlabel('windows')
plt.show()

skip_gram_fixed_window_results = []
for i in range(0,models_quantity):
    skip_gram_model = gensim.models.Word2Vec(tokenized_text, min_count = 1,  size = sizes[i], window = windows[0], sg = 1)
    skip_gram_fixed_window_results.append(skip_gram_model.wv.similarity(strings[0], strings[1]))

plt.plot(sizes, skip_gram_fixed_window_results)
plt.ylabel('cosine similarity')
plt.xlabel('sizes')
plt.show()
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]
text_index = 29
sequence = df['Text'][text_index].replace('\n',' ').replace('‘','').replace("’","").translate(str.maketrans('', '', string.punctuation))
#sequence = "On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further."
sequence = (sequence[:2000] + ' ..') if len(sequence) > 2000 else sequence
# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs)[0]
predictions = torch.argmax(outputs, dim=2)
print(df['Title'][text_index])
for token, prediction in zip(tokens, predictions[0].numpy()):
    if 'O' not in label_list[prediction]:
        print(token, label_list[prediction])