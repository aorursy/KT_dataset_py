import numpy as np

import pandas as pd





# Bokeh

from bokeh.io import output_notebook

from bokeh.plotting import figure, show

from bokeh.models import HoverTool, CustomJS, ColumnDataSource, Slider, Range1d

from bokeh.layouts import column

from bokeh.palettes import all_palettes

output_notebook()
np.random.seed(42)

df = pd.read_csv("../input/papers.csv")

print(df.paper_text[0][:500] + ' ...')
%%time

import spacy



nlp = spacy.load('en', disable=['parser', 'ner'])

df['paper_text_lemma'] = df.paper_text.map(lambda x: [token.lemma_ for token in nlp(x) if token.lemma_ != '-PRON-' and token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'}])



# Final cleaning

df['paper_text_lemma'] = df.paper_text_lemma.map(lambda x: [t for t in x if len(t) > 1])



# Example

print(df['paper_text_lemma'].iloc[0][:25], end='\n\n')
%%time

from sklearn.feature_extraction.text import TfidfVectorizer



n_features=2000

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, ngram_range=(1,2), stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(df.paper_text_lemma.map(lambda x: ' '.join(x)))
%%time

import umap



umap_embr = umap.UMAP(n_neighbors=10, metric='cosine', min_dist=0.1, random_state=42)

embedding = umap_embr.fit_transform(tfidf.todense())

embedding = pd.DataFrame(embedding, columns=['x','y'])
%%time

from nltk.corpus import stopwords



stop_en = stopwords.words('english')

df['paper_text_lemma'] = df.paper_text_lemma.map(lambda x: [t for t in x if t not in stop_en]) 

print(df['paper_text_lemma'].iloc[0][:25])
%%time

from gensim import corpora, models

np.random.seed(42)



# Create a corpus from a list of texts

texts = df.sample(n=1500, random_state=43)['paper_text_lemma'].values

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
%%time

from gensim.models.ldamodel import LdaModel

from gensim.models.coherencemodel import CoherenceModel



max_topics = 30

coh_list = []

for n_topics in range(3,max_topics+1):

    # Train the model on the corpus

    my_lda = LdaModel(corpus, num_topics=n_topics, id2word=dictionary, random_state=42, alpha=0.1)

    # Estimate coherence

    cm = CoherenceModel(model=my_lda, texts=texts, dictionary=dictionary, coherence='c_v', topn=20)

    coherence = cm.get_coherence_per_topic() # get coherence value

    coh_list.append(coherence)
# Coherence scores:

coh_means = np.array([np.mean(l) for l in coh_list])

coh_stds = np.array([np.std(l) for l in coh_list])



import matplotlib.pyplot as plt

%matplotlib inline

plt.xticks(np.arange(3, max_topics+1, 3.0));

plt.plot(range(3,max_topics+1), coh_means);

plt.fill_between(range(3,max_topics+1), coh_means-coh_stds, coh_means+coh_stds, color='g', alpha=0.05);

plt.vlines([8, 9], 0.24, 0.26, color='red', linestyles='dashed',  linewidth=1);

plt.hlines([0.253], 3, max_topics, color='black', linestyles='dotted',  linewidth=0.5);
%%time

from gensim import corpora, models

np.random.seed(42)



# Create a corpus from a list of texts

texts = df['paper_text_lemma'].values

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
%%time

n_topics=9

n_top_words = 25

my_lda = LdaModel(corpus, num_topics=n_topics, id2word=dictionary, random_state=42, minimum_probability=0)

for index, topic in my_lda.show_topics(formatted=False, num_words= n_top_words):

        print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
topics = ['T{}'.format(i) for i in range(n_topics)]
hm = np.array([[y for (x,y) in my_lda[corpus[i]]] for i in range(len(corpus))])
embedding['hue'] = hm.argmax(axis=1)

my_colors = [(all_palettes['Category20'][20] + all_palettes['Category20'][20])[i] for i in embedding.hue]

source = ColumnDataSource(

        data=dict(

            x = embedding.x,

            y = embedding.y,

            colors = my_colors,

            topic = [topics[i] for i in embedding.hue],

            title = df.title,

            year = df.year,

            alpha = [0.7] * embedding.shape[0],

            size = [7] * embedding.shape[0]

        )

    )

hover_emb = HoverTool(names=["df"], tooltips="""

    <div style="margin: 10">

        <div style="margin: 0 auto; width:300px;">

            <span style="font-size: 12px; font-weight: bold;">Topic:</span>

            <span style="font-size: 12px">@topic</span>

            <span style="font-size: 12px; font-weight: bold;">Title:</span>

            <span style="font-size: 12px">@title</span>

            <span style="font-size: 12px; font-weight: bold;">Year:</span>

            <span style="font-size: 12px">@year</span>

        </div>

    </div>

    """)

tools_emb = [hover_emb, 'pan', 'wheel_zoom', 'reset']

plot_emb = figure(plot_width=700, plot_height=700, tools=tools_emb, title='Papers')

plot_emb.circle('x', 'y', size='size', fill_color='colors', 

                 alpha='alpha', line_alpha=0, line_width=0.01, source=source, name="df", legend='topic')



plot_emb.legend.location = "bottom_left"

plot_emb.legend.label_text_font_size= "8pt"

plot_emb.legend.spacing = -5

plot_emb.x_range = Range1d(-9, 7)

plot_emb.y_range = Range1d(-9, 7)



callback = CustomJS(args=dict(source=source), code=

    """

    var data = source.data;

    var f = cb_obj.value

    x = data['x']

    y = data['y']

    colors = data['colors']

    alpha = data['alpha']

    title = data['title']

    year = data['year']

    size = data['size']

    for (i = 0; i < x.length; i++) {

        if (year[i] <= f) {

            alpha[i] = 0.9

            size[i] = 7

        } else {

            alpha[i] = 0.05

            size[i] = 4

        }

    }

    source.change.emit();

    """)



slider = Slider(start=df.year.min()-1, end=df.year.max(), value=2016, step=1, title="Before year")

slider.js_on_change('value', callback)



layout = column(slider, plot_emb)

show(layout)
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



legend_list = []

for color in all_palettes['Category20'][20][:n_topics]:   

    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))

    

fig,ax = plt.subplots(figsize=(12,13))

ax.scatter(embedding.x, embedding.y, c=my_colors, alpha=0.7)

ax.set_title('6 topics found via NMF');

fig.legend(legend_list, topics, loc=(0.18,0.87), ncol=3)

plt.subplots_adjust(top=0.82)

plt.suptitle("NIPS clustered by topic", **{'fontsize':'14','weight':'bold'});

plt.figtext(.51,0.95, 'topic modeling with NMF + 2D-embedding with UMAP', 

            **{'fontsize':'12','weight':'light'}, ha='center');
import pyLDAvis.gensim

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(my_lda, corpus, dictionary)