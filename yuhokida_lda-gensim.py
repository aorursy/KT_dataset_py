from time import time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from IPython.core.display import display

from gensim.parsing.preprocessing import preprocess_string

pd.set_option('display.max_colwidth', 200)

%matplotlib inline



# load data

df = pd.read_json('../input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)

df.sample(10)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# length of texts

display(df['headline'].str.len().describe())

display(df['short_description'].str.len().describe())



# df = df.head(100000)
# combine two columns to use all information

combined_text = df['headline'] + df['short_description']
import seaborn as sns

df.category.value_counts().plot.bar(figsize=(14, 4))
processed_text = combined_text.map(preprocess_string)



display(combined_text.head())

display(processed_text.head())
# Create a corpus from a list of texts

from gensim.corpora.dictionary import Dictionary



dictionary = Dictionary(processed_text)

dictionary.filter_extremes(no_below=10, no_above=0.7, keep_n=100000)

print(dictionary)



corpus = [dictionary.doc2bow(text) for text in processed_text]

corpus[10]
doc = corpus[10]

for i in range(len(doc)):

    print("Word {} (\"{}\") appears {} time.".format(

        doc[i][0], dictionary[doc[i][0]], doc[i][1]))
%%time

from gensim.models import LdaModel, LdaMulticore

# lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, random_state=42)

lda_model = LdaMulticore(corpus, num_topics=20, id2word=dictionary, random_state=42, workers=3)



print(lda_model)
print(lda_model.log_perplexity(corpus))

print()
# Get the most significant topics 

for idx, topic in lda_model.print_topics(-1): # -1 for all topics

    print('Topic: {} Word: {}'.format(idx, topic))
from wordcloud import WordCloud

from PIL import Image

 

fig, axs = plt.subplots(ncols=4, nrows=int(lda_model.num_topics/4), figsize=(15,10))

axs = axs.flatten()

 

def color_func(word, font_size, position, orientation, random_state, font_path):

    return 'darkblue'



 

for i, t in enumerate(range(lda_model.num_topics)):

 

    x = dict(lda_model.show_topic(t, 30))

    im = WordCloud(

        background_color='white',

#         color_func=color_func,

        random_state=0

    ).generate_from_frequencies(x)

    axs[i].imshow(im)

    axs[i].axis('off')

    axs[i].set_title('Topic '+ str(t))

        

plt.tight_layout()

plt.savefig('/kaggle/working/wordcloud.png')
# pyLDAvis

import pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()



# Vis PCoA

vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)

vis_pcoa



# save as html

# pyLDAvis.save_html(vis_pcoa, 'pyldavis_output_pcoa.html')
# coherence

# Each element in the list is a pair of a topic representation and its coherence score.

# Topic representations are distributions of words, represented as a list of pairs of word IDs and their probabilities.

topics = lda_model.top_topics(corpus, topn=10)



print('coherence: top words')

for topic in topics:

    print('{:.3f}: {}'.format(topic[1], ' '.join([i[1] for i in topic[0]])))