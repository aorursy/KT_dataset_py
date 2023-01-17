# Load libraries
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy import sparse
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import gensim
import gc
import plotly
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
from os import path
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import seaborn as sns
from collections import defaultdict
%matplotlib inline
# Load dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
test.head()
print(train.info())
print(test.info())
train.shape
test.shape
train.columns
test.columns
train.isnull().sum()
test.isnull().sum()
train_target = train['target'].values
np.unique(train_target)
train_target.mean()
train.describe() 
## target count ##
cnt_target = train['target'].value_counts()
count = go.Bar(
    x=cnt_target.index,
    y=cnt_target.values,
    marker=dict(
        color=cnt_target.values,
        colorscale = 'Viridis',
        reversescale = True),)

layout = go.Layout(
    title='Target Count',
    font=dict(size=18))

data = [count]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="TargetCount")
# Distributions
f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['target']==0].question_text.apply(lambda x: len(str(x).split())).plot.hist(ax=ax[0],bins=20,edgecolor='black',color='blue')
ax[0].set_title('target = Sincere Questions')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['target']==1].question_text.apply(lambda x: len(str(x).split())).plot.hist(ax=ax[1],color='yellow',bins=20,edgecolor='black')
ax[1].set_title('target = Insincere Questions')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.suptitle("Target Questions Distribution", fontsize=30)
plt.show()
# Wordcloud
def plot_wordcloud(text, mask=None, max_words=300, max_font_size=100, figure_size=(30.0,15.0), 
                   title = None, title_size=50, image_color=False):

    wordcloud = WordCloud(background_color='white',
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=1000, 
                    height=500,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(train["question_text"], title="Questions Wordcloud")
# Word frequency plot
train1 = train[train["target"]==1]
train0 = train[train["target"]==0]

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace
## Get the bar chart from sincere questions ##
freq_dict = defaultdict(int)
for sent in train0["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from insincere questions ##
freq_dict = defaultdict(int)
for sent in train1["question_text"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')
frames = [train,test]
data = pd.concat(frames, axis=0)
data.head()
data.tail()
# number of words 
data["num_words"]= data.question_text.apply(lambda x: len(str(x).split()))
# Number of unique words 
data["num_unique_words"] = data["question_text"].apply(lambda x: len(set(str(x).split())))
# length based features
data['len_text'] = data.question_text.apply(lambda x: len(str(x)))
# character length based features
data['len_char_question_text'] = data.question_text.apply(lambda x: 
                  len(''.join(set(str(x).replace(' ', '')))))
# word length based features
data['len_word_question_text'] = data.question_text.apply(lambda x: 
                                         len(str(x).split()))

# Number of stopwords 
data["num_stopwords"] = data["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
# common words in the dataset
data['common_words'] = data.apply(lambda x: 
                        len(set(str(x['question_text'])
                        .lower().split())), axis=1)
# Average length of the words 
data["len_mean_words"] = data["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
data.head()
fs_1 = ['num_words','num_unique_words','len_question_text', 'len_char_question_text', 
        'len_word_question_text','num_stopwords','common_words','len_mean_words']
tfv_data = TfidfVectorizer(min_df=3, 
                         max_features=None, 
                         strip_accents='unicode', 
                         analyzer='word', 
                         token_pattern=r'\w{1,}',
                         ngram_range=(1, 2), 
                         use_idf=1, 
                         smooth_idf=1, 
                         sublinear_tf=1,
                         stop_words='english')
data_tfidf = tfv_data.fit_transform(data.question_text.fillna(""))
svd_data = TruncatedSVD(n_components=180)
question_data_vectors = svd_data.fit_transform(data_tfidf)
data['skew_question_text_vec'] = [skew(x) for x in np.nan_to_num(question_data_vectors)]
data['kur_question_text_vec'] = [kurtosis(x) for x in np.nan_to_num(question_data_vectors)]
fs_2 = ['skew_question_text_vec', 'kur_question_text_vec']
del([tfv_data, data_tfidf,svd_data,question_data_vectors])
gc.collect()
data.head()
model1 = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
model2 = gensim.models.KeyedVectors.load_word2vec_format('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec', binary=False)
stop_words = set(stopwords.words('english'))
def sent2vec(s, model):  
    M = []
    words = word_tokenize(str(s).lower())
    for word in words:
        #It shouldn't be a stopword
        if word not in stop_words:
            #nor contain numbers
            if word.isalpha():
                #and be part of Word2Vec
                if word in model:
                    M.append(model[word])
    M = np.array(M)
    if len(M) > 0:
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())
    else:
        return model.get_vector('null')

w2v_qt1 = np.array([sent2vec(q, model1) 
                   for q in data.question_text])
w2v_qt2 = np.array([sent2vec(q, model2) 
                   for q in data.question_text])


data['cosine_distance'] = [cosine(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['cityblock_distance'] = [cityblock(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['jaccard_distance'] = [jaccard(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['canberra_distance'] = [canberra(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['euclidean_distance'] = [euclidean(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['minkowski_distance'] = [minkowski(x,y,3) for (x,y) in zip(w2v_qt1,w2v_qt2)]
data['braycurtis_distance'] = [braycurtis(x,y) for (x,y) in zip(w2v_qt1,w2v_qt2)]

fs_3 = ['cosine_distance', 'cityblock_distance', 
         'jaccard_distance', 'canberra_distance', 
         'euclidean_distance', 'minkowski_distance',
         'braycurtis_distance']

del([w2v_qt1, w2v_qt2, model1,model2])
gc.collect()
data.head()
data.shape
y = train.iloc[:,2].values
columns = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
X = data.iloc[0:1306122,columns].values
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=42)
import psutil
psutil.virtual_memory()

logres = linear_model.LogisticRegression(C=0.1, solver='sag', max_iter=1000)
logres.fit(X_train, y_train)

lr_preds = logres.predict(X_test)
from sklearn.metrics import f1_score
F1_score = f1_score(y_test, lr_preds, average='weighted')
print("Logistic regression F1 score: %0.3f" % F1_score)