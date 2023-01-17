#data wrangling packages

import pandas as pd

import numpy as np

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF

import random 

random.seed(13)



#visualization packages

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib

%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/papers.csv")

df.head()
n_features = 1000

n_topics = 8

n_top_words = 10





def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print("\nTopic #%d:" % topic_idx)

        print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()





tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,max_features=n_features,stop_words='english')



tfidf = tfidf_vectorizer.fit_transform(df['paper_text'])





nmf = NMF(n_components=n_topics, random_state=0,alpha=.1, l1_ratio=.5).fit(tfidf)



print("Topics found via NMF:")

tfidf_feature_names = tfidf_vectorizer.get_feature_names()

print_top_words(nmf, tfidf_feature_names, n_top_words)
nmf_embedding = nmf.transform(tfidf)



top_idx = np.argsort(nmf_embedding,axis=0)[-3:]



count = 0

for idxs in top_idx.T: 

    print("\nTopic {}:".format(count))

    for idx in idxs:

        print(df.iloc[idx]['title'])

    count += 1
topics = ['optimization algorithms',

          'neural network application',

          'reinforcement learning',

          'bayesian methods',

          'image recognition',

          'artificial neuron design',

          'graph theory',

          'kernel methods'

         ]
tsne = TSNE(random_state=3211)

tsne_embedding = tsne.fit_transform(nmf_embedding)

tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])

tsne_embedding['hue'] = nmf_embedding.argmax(axis=1)
###code used to create the first plot for getting the colors 

#plt.style.use('ggplot')



#fig, axs = plt.subplots(1,1, figsize=(5, 5), facecolor='w', edgecolor='k')

#fig.subplots_adjust(hspace = .1, wspace=.001)



#legend_list = []



#data = tsne_embedding

#scatter = plt.scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")

#plt.axis('off')

#plt.show()



#colors = []

#for i in range(len(topics)):

#    idx = np.where(data['hue']==i)[0][0]

#    color = scatter.get_facecolors()[idx]

#    colors.append(color)

#    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))

 

colors = np.array([[ 0.89411765,  0.10196079,  0.10980392,  1. ],

 [ 0.22685121,  0.51898501,  0.66574396,  1. ],

 [ 0.38731259,  0.57588621,  0.39148022,  1. ],

 [ 0.7655671 ,  0.38651289,  0.37099578,  1. ],

 [ 1.        ,  0.78937332,  0.11607843,  1. ],

 [ 0.75226453,  0.52958094,  0.16938101,  1. ],

 [ 0.92752019,  0.48406   ,  0.67238756,  1. ],

 [ 0.60000002,  0.60000002,  0.60000002,  1. ]])



legend_list = []



for i in range(len(topics)):   

    color = colors[i]

    legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
matplotlib.rc('font',family='monospace')

plt.style.use('ggplot')





fig, axs = plt.subplots(3,2, figsize=(10, 15), facecolor='w', edgecolor='k')

fig.subplots_adjust(hspace = .1, wspace=0)



axs = axs.ravel()



count = 0

legend = []

for year, idx in zip([1991,1996,2001,2006,2011,2016], range(6)):

    data = tsne_embedding[df['year']<=year]

    scatter = axs[idx].scatter(data=data,x='x',y='y',s=6,c=data['hue'],cmap="Set1")

    axs[idx].set_title('published until {}'.format(year),**{'fontsize':'10'})

    axs[idx].axis('off')



plt.suptitle("all NIPS proceedings clustered by topic",**{'fontsize':'14','weight':'bold'})

plt.figtext(.51,0.95,'unsupervised topic modeling with NMF based on textual content + 2D-embedding with t-SNE:', **{'fontsize':'10','weight':'light'}, ha='center')





fig.legend(legend_list,topics,loc=(0.1,0.89),ncol=3)

plt.subplots_adjust(top=0.85)



plt.show()