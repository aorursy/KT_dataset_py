# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Text analysis helper libraries
from gensim.summarization import summarize
from gensim.summarization import keywords

# Text analysis helper libraries for word frequency etc..
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation

# Word cloud visualization libraries
from scipy.misc import imresize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from collections import Counter


# Any results you write to the current directory are saved as output.
train_variants_df = pd.read_csv("../input/training_variants")
test_variants_df = pd.read_csv("../input/test_variants")
train_txt_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test_txt_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
print("Train and Test variants shape : ",train_variants_df.shape, test_variants_df.shape)
print("Train and Test text shape : ",train_txt_df.shape, test_txt_df.shape)
train_df = pd.merge(train_variants_df, train_txt_df, on='ID')
test_df = pd.merge(test_variants_df, test_txt_df, on='ID')
#train_df.head(5)
#test_df.head(5)
#train_df.describe()
train_df.head(5)
print("For training data, there are a total of", len(train_df.ID.unique()), "IDs,")
print(len(train_df.Gene.unique()), "unique genes,")
print(len(train_df.Variation.unique()), "unique variations and ")
print(len(train_df.Class.unique()),  "classes")
plt.figure(figsize=(12,8))
sns.countplot(x="Class", data=train_df)
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Distribution of genetic mutation classes", fontsize=18)
plt.show()
train_genes = train_df.groupby("Gene")['Gene'].count()
fewest_genes = train_genes.sort_values(ascending=True)[:10]
print("Genes with most occurences\n", train_genes.sort_values(ascending=False)[:10])
print("\nGenes with fewest occurences\n", fewest_genes)
fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True, figsize=(12,12))

# Normalize value counts for better comparison
def normalize_group(x):
    label, repetition = x.index, x
    t = sum(repetition)
    r = [n/t for n in repetition]
    return label, r

for idx, g in enumerate(train_df.groupby('Class')):
    label, val = normalize_group(g[1]["Gene"].value_counts())
    ax = axes.flat[idx]
    ax.bar(np.arange(5), val[:5],
           tick_label=label[:5]) 
    ax.set_title("Class {}".format(g[0]))
    
fig.text(0.5, 0.97, 'Normalized Top 5 Gene Frequency for each Class', ha='center', fontsize=14, fontweight='bold')
fig.text(0.5, 0, 'Gene', ha='center', fontweight='bold')
fig.text(0, 0.5, 'Frequency', va='center', rotation='vertical', fontweight='bold')
fig.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
train_df.head(5)
train_df['Text_count']  = train_df["Text"].apply(lambda x: len(str(x).split()))
train_df.head()
plt.figure(figsize=(12, 8))
sns.distplot(train_df.Text_count.values, bins=50, kde=False, color='blue')
plt.xlabel('Number of words in text', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Frequency of number of words", fontsize=15)
plt.show()
plt.figure(figsize=(12,8))
gene_count_grp = train_df.groupby('Gene')["Text_count"].sum().reset_index()
sns.violinplot(x="Class", y="Text_count", data=train_df, inner=None)
sns.swarmplot(x="Class", y="Text_count", data=train_df, color="y", alpha=.5);
plt.ylabel('Text Count', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.title("Text length distribution", fontsize=18)
plt.show()
custom_words = ["fig", "figure", "et", "al", "al.", "also",
                "data", "analyze", "study", "table", "using",
                "method", "result", "conclusion", "author", 
                "find", "found", "show", '"', "’", "“", "”"]

stop_words = set(stopwords.words('english') + list(punctuation) + custom_words)
wordnet_lemmatizer = WordNetLemmatizer()

class_corpus = train_df.groupby('Class').apply(lambda x: x['Text'].str.cat())
# this cell takes a long time

class_corpus = class_corpus.apply(lambda x: Counter(
    [wordnet_lemmatizer.lemmatize(w) 
     for w in word_tokenize(x) 
     if w.lower() not in stop_words and not w.isdigit()]
))
class_corpus
whole_text_freq = class_corpus.sum()

fig, ax = plt.subplots()

label, repetition = zip(*whole_text_freq.most_common(25))

ax.barh(range(len(label)), repetition, align='center')
ax.set_yticks(np.arange(len(label)))
ax.set_yticklabels(label)
ax.invert_yaxis()

ax.set_title('Word Distribution Over Whole Text')
ax.set_xlabel('# of repetitions')
ax.set_ylabel('Word')

plt.tight_layout()
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
train_df['Text'].head()
tfidf = TfidfVectorizer(
                    min_df=5, max_features=16000, strip_accents='unicode', lowercase=True,
                    analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
                    smooth_idf=True, sublinear_tf=True, stop_words = 'english')
# this cell takes a long time
tfidf.fit(train_df["Text"].values.astype('U'))
X_train_tfidf = tfidf.transform(train_df['Text'].values.astype('U'))
X_test_tfidf = tfidf.transform(test_df['Text'].values.astype('U'))

y_train = train_df['Class'].values
def evaluate(X, y, clf=None):
    probas = cross_val_predict(clf, X, y, cv=StratifiedKFold(n_splits=5, random_state=8), 
                              n_jobs=-1, method='predict_proba', verbose=2)
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print('Log loss: {}'.format(log_loss(y, probas)))
    print('Accuracy: {}'.format(accuracy_score(y, preds)))
# this cell takes a long time
clf=svm.SVC(probability=True)
clf.fit(X_train_tfidf, y_train)
y_test_predicted = clf.predict_proba(X_test_tfidf)
submission_df = pd.DataFrame(y_test_predicted, columns=['class' + str(c + 1) for c in range(9)])
submission_df.insert(0, 'ID', value=test_df['ID'])
submission_df.head()
submission_df.to_csv('submission.csv', index=False)

