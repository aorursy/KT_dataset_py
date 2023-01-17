import re

import spacy

import string

import datetime

import pandas as pd

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from sklearn.manifold import TSNE
# nltk.download()

STOPWORDS = set(stopwords.words("english"))
# Read the data.

print("Started: ", datetime.datetime.now())

df = pd.read_csv("/kaggle/input/medium-articles/articles.csv")

print("Data Read: ", datetime.datetime.now())
def clean_text(text):

    """

    Function to clean the text by performing:

    1) Lowercase Operation.

    2) Removing words with square brackets.

    3) Removing punctuations.

    4) Removing stopwords.

    :param text: Raw Text.

    :return text: Clean Text.

    """



    text = text.lower()

    text = re.sub(r"\[.*?\]", "", text)

    text = re.sub(r"\w*\d\w*", "", text)

    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

    if len(text) > 3:

        text = " ".join([t for t in text.split() if t not in STOPWORDS])



        return text

    else:



        return ""
# Clean the text.

df["text"] = df["text"].apply(lambda text_value: clean_text(text_value))

print("Cleaned Text: ", datetime.datetime.now())
def lemmatize_text_tokens(text, nlp):

    """

    Funciton to tokenize a sentence, lemmatize the tokens and return lemmatized sentence back.

    :param text: Raw Text.

    :param nlp: Spacy Object with "en" corpus loaded.

    :return lemmatized_text: Lemmatized sentence.

    """



    tokens = nlp(text)

    lemmatized_tokens = list()

    for token in tokens:

        lemmatized_token = token.lemma_

        lemmatized_tokens.append(lemmatized_token)

    lemmatized_text = " ".join(lemmatized_tokens)



    return lemmatized_text
# Lemmatize the tokens.

nlp = spacy.load("en")

df["lemmatized_text"] = df["text"].apply(lambda text_value: lemmatize_text_tokens(text_value, nlp))

print("Lemmatized the Documents' Tokens: ", datetime.datetime.now())
# Split the documents into tokens.

doc_sentences = [text.split() for text in list(df["lemmatized_text"])]

print("Split Token of Documents: ", datetime.datetime.now())
w2v_model = Word2Vec(min_count=200,

                         window=5,

                         size=100,

                         workers=7)
print("Building Vocabulary: ", datetime.datetime.now())

w2v_model.build_vocab(doc_sentences)
print("Training Word2Vec Model: ", datetime.datetime.now())

w2v_model.train(doc_sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

print("Training Done: ", datetime.datetime.now())
w2v_model.init_sims(replace=True)

most_similar_words = w2v_model.wv.most_similar(positive=['computer'])

for similar_word in most_similar_words:

    print(similar_word)
def plt_tsne(word2vec_model):

    """

    Function to plot the words using t-SNE from the models vocabulary and the probability associations.

    :param word2vec_model: Word2Vec Model.

    """



    labels = list()

    words = list()



    for word in word2vec_model.wv.vocab:

        words.append(word2vec_model[word])

        labels.append(word)



    tsne_model = TSNE(perplexity=25, n_components=2, init="pca", n_iter=2000, random_state=0)

    new_values = tsne_model.fit_transform(words)



    x, y = list(), list()



    for value in new_values:

        x.append(value[0])

        y.append(value[1])



    plt.figure(figsize=(18, 18))



    for i in range(len(x)):

        plt.scatter(x[i], y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords="offset points",

                     ha="right",

                     va="bottom")

    plt.savefig("./tsne_plot_word2vec.png")

    plt.show(True)
plt_tsne(w2v_model)