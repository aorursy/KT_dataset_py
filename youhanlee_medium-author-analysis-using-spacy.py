#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import os
#print(os.listdir("../input"))
import spacy
import random 
from collections import Counter #for counting
import seaborn as sns #for visualization
import matplotlib.pyplot as plt

plt.style.use('seaborn')
sns.set(font_scale=2)
articles = pd.read_csv('../input/articles.csv')
nlp = spacy.load('en')
doc = nlp(articles['text'][0][:500]) 
df_token = pd.DataFrame()

for i, token in enumerate(doc):
    df_token.loc[i, 'text'] = token.text
    df_token.loc[i, 'lemma'] = token.lemma_,
    df_token.loc[i, 'pos'] = token.pos_
    df_token.loc[i, 'tag'] = token.tag_
    df_token.loc[i, 'dep'] = token.dep_
    df_token.loc[i, 'shape'] = token.shape_
    df_token.loc[i, 'is_alpha'] = token.is_alpha
    df_token.loc[i, 'is_stop'] = token.is_stop
df_token
from spacy import displacy
sentence_spans = list(doc.sents)
displacy.render(sentence_spans, style='dep', jupyter=True)
spacy.displacy.render(doc, style='ent',jupyter=True)
articles['author'].value_counts()
from nltk.corpus import stopwords
import string
stopwords = stopwords.words('english')
punctuations = string.punctuation
# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 100 == 0:
            print('Processed {} out of {}'.format(counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)
def make_barplot_for_author(Author):
    author_text = [text for text in articles.loc[articles['author'] == Author]['text']]

    author_clean = cleanup_text(author_text)
    author_clean = ' '.join(author_clean).split()
    author_clean = [word for word in author_clean if word not in '\'s']
    author_counts = Counter(author_clean)

    NUM_WORDS = 25
    author_common_words = [word[0] for word in author_counts.most_common(NUM_WORDS)]
    author_common_counts = [word[1] for word in author_counts.most_common(NUM_WORDS)]

    plt.figure(figsize=(15, 12))
    sns.barplot(x=author_common_counts, y=author_common_words)
    plt.title('Words that {} use frequently'.format(Author), fontsize=20)
    plt.show()
Author = 'Adam Geitgey'
make_barplot_for_author(Author)
for title in articles.loc[articles['author'] == 'Adam Geitgey']['title']:
    print(title)
Author = 'Slav Ivanov'
make_barplot_for_author(Author)
for title in articles.loc[articles['author'] == 'Slav Ivanov']['title']:
    print(title)
Author = 'Arthur Juliani'
make_barplot_for_author(Author)
for title in articles.loc[articles['author'] == 'Arthur Juliani']['title']:
    print(title)
Author = 'Milo Spencer-Harper'
make_barplot_for_author(Author)
for title in articles.loc[articles['author'] == 'Milo Spencer-Harper']['title']:
    print(title)
Author = 'Dhruv Parthasarathy'
make_barplot_for_author(Author)
Author = 'William Koehrsen'
make_barplot_for_author(Author)