# all imports

import numpy as np #linear algebra

import pandas as pd # data processing I/O



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize

from nltk.stem import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import sent_tokenize, word_tokenize



import re



import matplotlib.pyplot as plt #plots

%matplotlib inline

import seaborn as sns#advanced plots



import warnings

warnings.filterwarnings("ignore")#ignore warnings



from tqdm import tqdm



from sklearn.feature_extraction.text import TfidfVectorizer

import collections
tr_var = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants', sep=',')

tr_var.head()
tr_text = pd.read_csv('../input/msk-redefining-cancer-treatment/training_text', sep='\|\|',engine="python",names=["ID","TEXT"],skiprows=1)

tr_text.head(1)
tr_var.describe(include='all')
tr_text.describe(include='all')
print('Is there any null values in training variants:', tr_var.isnull().values.any())

print('Is there any null values in training text:', tr_text.isnull().values.any())
print("get indexes of null values in all training text")

print(tr_text.loc[tr_text.isnull().any(axis=1)])
plt.figure(figsize=(12,8))

plt.title("classes distribution: training Data")

#(1-9 the class this genetic mutation has been classified on)

sns.countplot(x="Class", data=tr_var)

plt.xlabel("Classes")

plt.show()
plt.figure(figsize=(12,8))

plt.title("classes distribution: training Data")

#(1-9 the class this genetic mutation has been classified on)

sns.countplot(x="Gene", data=tr_var)

plt.xlabel("Classes")

plt.show()
plt.figure(figsize=(12,8))

plt.title('frequency dist of gene')

sns.distplot(sorted(tr_var["Gene"].value_counts().tolist(), reverse=True), hist = False, kde_kws=dict(cumulative=True))

plt.xlabel('Gene')

plt.grid()

plt.minorticks_on()

plt.grid(b=True, which='minor', color='r', linestyle='--')

plt.show()
tr_var["Gene"].value_counts()[:20]
def selectTopGene(top):

  tmp_list = tr_var["Gene"].value_counts()[:top].index.tolist()

  tmp_df = tr_var[tr_var['Gene'].isin(tmp_list)]

  #print(tmp_df)

  return tmp_df
def drawClassVsTopGeneFacet(top):

  tmp_df = selectTopGene(top)

  fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(18,15))

  for i in range(3):

    for j in range(3):

      tmp_df_1 = tmp_df[tmp_df['Class'] == ((i*3+j)+1)].groupby('Gene').size().reset_index(name='counts')

      tmp_df_1 = tmp_df_1.sort_values('counts', ascending=False)

      tmp_df_top_7 = tmp_df_1[:]

      axs[i][j].set_title('for Class ' + str((i*3+j)+1))

      plt.sca(axs[i][j])

      plt.xticks(rotation=30)

      sns.barplot(x="Gene", y="counts", data=tmp_df_top_7, ax=axs[i][j])
drawClassVsTopGeneFacet(20)
tr_text.columns
print(tr_text['TEXT'].iloc[0])
print(tr_text['TEXT'].iloc[1])
def plotWordCloud():

  combText = tr_text['TEXT'].agg(lambda x: ' '.join(x.dropna()))

  wordcloud = WordCloud().generate(combText)

  # Display the generated image:

  print("word cloud for text ")

  plt.figure(figsize=(12,8))

  plt.imshow(wordcloud, interpolation='bilinear')

  plt.axis("off")

  plt.show()
plotWordCloud()
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()

stemmer = SnowballStemmer("english")
def removeStopWords(sentence):

  sentence = sentence.replace('\\r', ' ')

  sentence = sentence.replace('\\"', ' ')

  sentence = sentence.replace('\\n', ' ')

  sentence = re.sub('\(.*?\)', ' ', sentence)

  sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)

  list_of_words = [i.lower() for i in wordpunct_tokenize(sentence) if i.lower() not in stop_words]

  list_of_words = [stemmer.stem(i.lower()) for i in list_of_words if stemmer.stem(i.lower()) not in stop_words]

  sentence = ' '.join(list_of_words).lower().strip()



  return sentence

stop_words.update(['line', 'fig','figure', 'author','find',

                   'et', 'al', 'evaluate', 'show', 'demonstrate', 'conclusion', 'study', 'analysis', 'method'])
def preProcessText():

  tmp_sen = []

  for i in tqdm(tr_text['TEXT']):

    i = removeStopWords(i)

    tmp_sen.append(i)

  

  tr_text['TEXT'] = tmp_sen
tr_text = tr_text.replace(np.nan, '', regex=True)

preProcessText()

plotWordCloud()
df = tr_var.join(other=tr_text.set_index('ID'), on='ID')

df.head()
def jaccard_similarity(document1, document2):

  intersection = set(document1).intersection(set(document2))

  union = set(document1).union(set(document2))

  return len(intersection)/len(set(union))
#https://stackoverflow.com/a/17841321

tmp_df = df.groupby('Class')['TEXT'].agg(lambda col: ' '.join(col)).reset_index()
tmp_df.head()
similarity = np.zeros((9, 9))

def calculateSimilrity():

  for i in tqdm(tmp_df["Class"].values):

    for j in tmp_df["Class"].values:

     if(i < j):

       sim = jaccard_similarity(tmp_df['TEXT'].iloc[i - 1].split(), tmp_df['TEXT'].iloc[j - 1].split())

       similarity[i - 1][j - 1] = sim

       similarity[j - 1][i - 1] = sim

calculateSimilrity()
#https://stackoverflow.com/a/58165593

#https://indianaiproduction.com/seaborn-heatmap/

plt.figure(figsize=(12, 8))

up_matrix = np.triu(similarity)

ax = sns.heatmap(similarity, xticklabels=range(1,10), yticklabels=range(1,10), annot=True, mask=up_matrix)

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
def calculateTF(document):

  tmp_tfs = dict()

  words = document.split()

  len_doc = len(words)

  for word in words:

    if word in tmp_tfs:

      tmp_tfs[word] += 1

    else:

      tmp_tfs[word] = 1

  

  for word in tmp_tfs:

    tmp_tfs[word] = tmp_tfs[word] / len_doc

  return tmp_tfs
def calculateIDF(documents):

  vectorizer = TfidfVectorizer()

  X = vectorizer.fit_transform(documents)

  tmp_dct = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_ ))

  return tmp_dct
def calculateTfIdf(documents):

  tmp_tfIdf = dict()

  tmp_idfs = calculateIDF(documents)

  count = 0

  for i in documents:

    tmp_tfs = calculateTF(i)

    tmp_dict = dict()

    for j in set(i.split()):

      try:

        tmp_dict[j] = tmp_idfs[j] * tmp_tfs[j]

      except:

        continue

    tmp_tfIdf[count] = tmp_dict

    count += 1

  return tmp_tfIdf
tfIdfs = calculateTfIdf(df['TEXT'].values)
sorted(tfIdfs[0].items(), key=lambda kv: kv[1])[:2]
plt.figure(figsize=(12,15))

plt.title("word frequency for each class")

for i in tqdm(range(9)):

  tmp_arr = sorted(tfIdfs[i].items(), key=lambda kv: kv[1], reverse=True)[:10]

  plt.subplot(3, 3, (i + 1))

  plt.title('class ' + str(i + 1))

  tfidfseries = pd.Series(data=[p[1] for p in tmp_arr],name='TFIDF')

  names = pd.Series(data=[p[0] for p in tmp_arr], name='Words')

  frame = {'words': names, 'tfidfs': tfidfseries}

  tmp_df_tfidf = pd.DataFrame(frame)

  plt.xticks(rotation=30)

  ax = sns.barplot(x='words', y="tfidfs", data=tmp_df_tfidf)



plt.show()