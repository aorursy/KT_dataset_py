# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

plt.style.use('ggplot')

import glob

import json

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import time

import warnings 
root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)



    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])

df_covid.head()
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
df_covid.info()
df_covid.describe(include='all')
df_covid.drop_duplicates(['abstract'], inplace=True)

df_covid.describe(include='all')
df_covid.head()
import re



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
df_covid.head(4)
text = df_covid.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count"], axis=1)
text.head(5)
text_arr = text.stack().tolist()

len(text_arr)
words = []

for ii in range(0,len(text)):

    words.append(str(text.iloc[ii]['body_text']).split(" "))
print(words[0][0])
n_gram_all = []



for word in words:

    # ??????n-gram?????????

    n_gram = []

    for i in range(len(word)-2+1):

        n_gram.append("".join(word[i:i+2]))

    n_gram_all.append(n_gram)
n_gram_all[0][2]
from sklearn.feature_extraction.text import HashingVectorizer



# hash vectorizer instance

hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)



# features matrix X

X = hvec.fit_transform(n_gram_all)
X.shape
from sklearn.model_selection import train_test_split



X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)



print("X_train size:", len(X_train))

print("X_test size:", len(X_test), "\n")
from sklearn.manifold import TSNE



tsne = TSNE(verbose=1, perplexity=5)

X_embedded = tsne.fit_transform(X_train)
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", 1)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)



plt.title("t-SNE Covid-19 Articles")

#plt.savefig("plots/t-sne_covid19.png")

plt.show()
from sklearn.cluster import KMeans



k = 10

kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)

y_pred = kmeans.fit_predict(X_train)
y_train = y_pred
y_test = kmeans.predict(X_test)
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y_pred)))



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered")

plt.show()
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



# ?????????????????????

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)



# ????????????

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=3, n_jobs=4)



# ????????????????????????

print("Accuracy: ", '{:,.3f}'.format(float(forest_scores.mean()) * 100), "%")
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_score, recall_score



# ??????

forest_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3, n_jobs=4)



# ??????????????????

print("Precision: ", '{:,.3f}'.format(float(precision_score(y_train, forest_train_pred, average='macro')) * 100), "%")

print("   Recall: ", '{:,.3f}'.format(float(recall_score(y_train, forest_train_pred, average='macro')) * 100), "%")
# ????????????

forest_clf.fit(X_train, y_train)



# ????????????????????????

forest_pred = forest_clf.predict(X_test)
# ???????????????????????????

def classification_report(model_name, test, pred):

    from sklearn.metrics import precision_score, recall_score

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import f1_score

    

    print(model_name, ":\n")

    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")

    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='micro')) * 100), "%")

    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='micro')) * 100), "%")

    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")



# ??????????????????

classification_report("Random Forest Classifier Report (Test Set)", y_test, forest_pred)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=2**12)

X = vectorizer.fit_transform(text_arr)
X.shape
from sklearn.model_selection import train_test_split



# test set size of 20% of the data and the random seed 42 <3

X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)



print("X_train size:", len(X_train))

print(" X_test size:", len(X_test), "\n")
k = 10

kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)

y_pred = kmeans.fit_predict(X_train)
# ?????????

y_train = y_pred



# ?????????

y_test = kmeans.predict(X_test)
from sklearn.manifold import TSNE



tsne = TSNE(verbose=1)

X_embedded = tsne.fit_transform(X_train)
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y_pred)))



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")

plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca_result = pca.fit_transform(X_train)
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y_pred)))



# plot

sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("PCA Covid-19 Articles - Clustered (K-Means) - Tf-idf with Plain Text")

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_result[:,0], 

    ys=pca_result[:,1], 

    zs=pca_result[:,2], 

    c=y_pred, 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.title("PCA Covid-19 Articles (3D) - Clustered (K-Means) - Tf-idf with Plain Text")

plt.show()
k = 20

kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)

y_pred = kmeans.fit_predict(X_train)
from matplotlib import pyplot as plt

import seaborn as sns

import random 



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# let's shuffle the list so distinct colors stay next to each other

palette = sns.hls_palette(20, l=.4, s=.9)

random.shuffle(palette) 



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered(K-Means 20) - Tf-idf with Plain Text")

#plt.savefig("plots/t-sne_covid19_20label_TFID.png")

plt.show()