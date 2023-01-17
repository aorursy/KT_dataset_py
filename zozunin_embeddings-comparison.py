from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.downloader as api

from gensim.models import word2vec, KeyedVectors
import os

print(os.listdir("../input/added-glove"))
twitter_w2v = KeyedVectors.load_word2vec_format('../input/word2vec-twitter-model/word2vec_twitter_model.bin', binary=True, encoding = "ISO-8859-1")
len(twitter_w2v.vocab)
glove_twitter = KeyedVectors.load_word2vec_format('../input/added-glove/glove_twitter_27B_200d.txt',binary=False)
len(glove_twitter.vocab)
train = pd.read_csv('../input/lemmatized/clean_data_lem_nosw.csv', encoding='utf-8').drop('Unnamed: 0', axis=1)

train.fillna('', inplace=True)
np_x_train=[comment.split() for comment in train.cleaned]
my_model = word2vec.Word2Vec(np_x_train, min_count=1, window=5, size=300, workers=4)
len(my_model.wv.vocab)
#twitter_w2v.most_similar(positive='stupid')
#glove_twitter.most_similar(positive='stupid')
#my_model.wv.most_similar(positive='stupid')
#twitter_w2v.most_similar(positive='hate')
#glove_twitter.most_similar(positive='hate')
#my_model.wv.most_similar(positive='hate')
#twitter_w2v.most_similar(positive='pride')
#glove_twitter.most_similar(positive='pride')
#my_model.wv.most_similar(positive='pride')
def words_for_tsne(comments):

    vectorizer = TfidfVectorizer(min_df=5000)

    vectorizer.set_params(ngram_range=(1, 1), lowercase=False,tokenizer=lambda x: x, preprocessor=lambda x: x, norm='l2')

    count = vectorizer.fit(comments)

    bow = vectorizer.transform(comments)

    all_words = vectorizer.get_feature_names()

    

    

    return all_words
from sklearn.manifold import TSNE

from bokeh.models import ColumnDataSource, LabelSet

from bokeh.plotting import figure, show, output_file

from bokeh.io import output_notebook

output_notebook()



def visual(model, temp_l):

    words_top_vec_ted = []

    names = []

    for i in temp_l:

        try: 

            words_top_vec_ted.append(model[i])

            names.append(i)

        except:

            pass

    words_top_vec_ted = np.array(words_top_vec_ted)

    

    tsne = TSNE(n_components=2, random_state=0)

    words_top_ted_tsne = tsne.fit_transform(words_top_vec_ted)

    p = figure(tools="pan,wheel_zoom,reset,save",

               toolbar_location="above",

               title="word2vec T-SNE for most common words")

    

    source = ColumnDataSource(data=dict(x1=words_top_ted_tsne[:,0],

                                        x2=words_top_ted_tsne[:,1],

                                        names=list(names)))



    p.scatter(x="x1", y="x2", size=8, source=source)



    labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,

                      text_font_size="8pt", text_color="#555555",

                      source=source, text_align='center')

    p.add_layout(labels)



    show(p)
visual(my_model, words_for_tsne(np_x_train))
visual(glove_twitter, words_for_tsne(np_x_train))
visual(twitter_w2v, words_for_tsne(np_x_train))