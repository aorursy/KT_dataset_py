import pandas as pd

import numpy as np

# LDA, tSNE

from sklearn.manifold import TSNE

from gensim.models.ldamodel import LdaModel

from sklearn.metrics.pairwise import pairwise_distances

# NLTK

from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk import pos_tag

from nltk.corpus import stopwords

import re

# Bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider

from bokeh.layouts import column

from bokeh.palettes import all_palettes

output_notebook()
an_author = 1211

df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df.text[an_author][:500])
# Removing numerals:

df['text_tokens'] = df.text.map(lambda x: re.sub(r'\d+', '', x))

# Lower case:

df['text_tokens'] = df.text_tokens.map(lambda x: x.lower())

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:500])
df['text_tokens'] = df.text_tokens.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:25])
lemma = WordNetLemmatizer()

df['tags'] = df.text_tokens.map(lambda x: list(zip(*pos_tag(x)))[1])



def recode_tag(tag):

    if tag[0].lower() in ['n', 'r', 'v', 'j']:

        if tag[0].lower() == 'j': return 'a'

        else: return tag[0].lower()

    else: return None



df['tags'] = df.tags.map(lambda x: list(map(recode_tag, x)))

df['tags'] = df.apply(lambda x: list(zip(x.text_tokens, x.tags)), axis=1)



def lemmatize_tokens(pairs):

    return [lemma.lemmatize(tok, pos=tag) if tag != None else tok 

            for (tok, tag) in pairs]



df['text_tokens'] = df.tags.map(lemmatize_tokens)

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:25])

print(df['tags'][an_author][:25])
stop_en = stopwords.words('english')

df['text_tokens'] = df.text_tokens.map(lambda x: [t for t in x if t not in stop_en])

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:25])
df['text_tokens_bigrams'] = df.text_tokens.map(lambda x: [' '.join(x[i:i+2]) 

                                                          for i in range(len(x)-1)])

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens_bigrams'][an_author][:25])
df['text_tokens'] = df.text_tokens.map(lambda x: [t for t in x if len(t) > 1])

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:25])
df['text_tokens'] = df.text_tokens + df.text_tokens_bigrams

print("Author: {}, ID: {}".format(df.author[an_author], df.id[an_author]))

print(df['text_tokens'][an_author][:100])
from gensim import corpora, models

T = 4 # number of topics

np.random.seed(2017)

texts = df['text_tokens'].values

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, 

                                    num_topics=T, passes=7, minimum_probability=0)
ldamodel.print_topics(num_topics=3, num_words=5)
# Matrix with topics probabilities for every text:

hm = np.array([[y for (x,y) in ldamodel[corpus[i]]] for i in range(len(corpus))])

# Computing pairwise cosine distance between texts:

precomp_cosine = pairwise_distances(hm, metric='cosine')
tsne = TSNE(random_state=2017, perplexity=25, metric='precomputed', early_exaggeration=4)

tsne_rep = tsne.fit_transform(precomp_cosine)

tsne_rep = pd.DataFrame(tsne_rep, columns=['x','y'])

tsne_rep['hue'] = [['EAP', 'HPL', 'MWS'].index(x) for x in df.author.values]
source = ColumnDataSource(

        data=dict(

            x = tsne_rep.x,

            y = tsne_rep.y,

            colors = [all_palettes['Inferno'][4][i] for i in tsne_rep.hue],

            author = df.author,

            text = df.text,

            alpha = [0.7] * tsne_rep.shape[0],

            size = [7] * tsne_rep.shape[0]

        )

    )



hover_tsne = HoverTool(names=["df"], tooltips="""

    <div style="margin: 10">

        <div style="margin: 0 auto; width:300px;">

            <span style="font-size: 12px; font-weight: bold;">Author:</span>

            <span style="font-size: 12px">@author</span>

        </div>

        <div style="margin: 0 auto; width:300px;">

            <span style="font-size: 12px; font-weight: bold;">Text:</span>

            <span style="font-size: 12px">@text</span>

        </div>

    </div>

    """)



tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']

plot_tsne = figure(plot_width=700, plot_height=700, tools=tools_tsne, title='Spooky')



plot_tsne.circle('x', 'y', size='size', fill_color='colors', 

                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df")



layout = column(plot_tsne)
show(layout)