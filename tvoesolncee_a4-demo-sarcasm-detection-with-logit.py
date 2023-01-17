!ls ../input/sarcasm/
# some necessary imports
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
train_df = pd.read_csv('../input/sarcasm/train-balanced-sarcasm.csv')
train_df.head()
train_df.info()
train_df.dropna(subset=['comment'], inplace=True)
train_df['label'].value_counts()
train_texts, valid_texts, y_train, y_valid = \
        train_test_split(train_df['comment'], train_df['label'], random_state=17)
train_texts = train_texts.to_frame()
valid_texts = valid_texts.to_frame()
y_train = y_train.to_frame()
y_valid = y_valid.to_frame()
## Number of words in the text ##
train_texts["num_words"] = train_texts["comment"].apply(lambda x: len(str(x).split()))
valid_texts["num_words"] = valid_texts["comment"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_texts["num_unique_words"] = train_texts["comment"].apply(lambda x: len(set(str(x).split())))
valid_texts["num_unique_words"] = valid_texts["comment"].apply(lambda x: len(set(str(x).split())))
train_texts = train_texts.reset_index()
valid_texts = valid_texts.reset_index()
y_train = y_train.reset_index()
y_valid = y_valid.reset_index()
train_texts.shape, y_train.shape
train_texts['comment'].tail()
# Get the tfidf vectors #
tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
tfidf_vec.fit_transform(train_texts['comment'].values.tolist() + valid_texts['comment'].values.tolist())
train_tfidf = tfidf_vec.transform(train_texts['comment'].values.tolist())
test_tfidf = tfidf_vec.transform(valid_texts['comment'].values.tolist())
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
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
    
plot_wordcloud(train_df["comment"], title="Word Cloud of Comments")
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

## target count ##
cnt_srs = train_df['label'].value_counts()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Label Count',
    font=dict(size=18)
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="LabelCount")

## target distribution ##
labels = (np.array(cnt_srs.index))
sizes = (np.array((cnt_srs / cnt_srs.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Label distribution',
    font=dict(size=18),
    width=600,
    height=600,
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype")
from sklearn.model_selection import KFold
from sklearn import metrics


def runModel(train_X, train_y, test_X, test_y, test_X2):
    model = LogisticRegression(C=5., solver='sag')
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)[:,1]
    pred_test_y2 = model.predict_proba(test_X2)[:,1]
    return pred_test_y, pred_test_y2, model

print("Building model.")
cv_scores = []
pred_full_test = 0
pred_train = np.zeros([train_texts.shape[0]])
kf = KFold(n_splits=5, shuffle=True, random_state=17)
for dev_index, val_index in kf.split(train_texts):
    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
    dev_y, val_y = y_train.iloc[dev_index].label, y_train.iloc[val_index].label
    pred_val_y, pred_test_y, model = runModel(dev_X, dev_y, val_X, val_y, test_tfidf)
    pred_full_test = pred_full_test + pred_test_y
    pred_train[val_index] = pred_val_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
for thresh in np.arange(0.3, 0.36, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
np.mean(cv_scores)
pred_val_y.shape, val_y.shape
model.predict_proba(test_tfidf[3])[:,1]
train_texts['comment'].iloc[1]
test_tfidf[1].shape

all_model = LogisticRegression(C=5., solver='sag')
all_model.fit(train_tfidf, y_train.label)
pred_test_y = all_model.predict_proba(test_tfidf)[:,1]
metrics.log_loss(y_valid.label, pred_test_y)
thresh = 0.36
metrics.f1_score(y_valid.label, (pred_test_y>thresh).astype(int))
for thresh in np.arange(0.34, 0.4, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_valid.label, (pred_test_y>thresh).astype(int))))
import eli5
eli5.show_weights(all_model, vec=tfidf_vec, top=50, feature_filter=lambda x: x != '<BIAS>')
from sklearn.feature_extraction.text import CountVectorizer
%%time
from sklearn.pipeline import make_pipeline

text_pipe_logit = make_pipeline(CountVectorizer(),
                                # for some reason n_jobs > 1 won't work 
                                # with GridSearchCV's n_jobs > 1
                                LogisticRegression(C=5., solver='sag',
                                                   random_state=17))

text_pipe_logit.fit(train_texts.comment, y_train.label)
print(text_pipe_logit.score(valid_texts.comment, y_valid.label))
from sklearn.model_selection import GridSearchCV

param_grid_logit = {'logisticregression__C': np.logspace(-5, 0, 6)}
grid_logit = GridSearchCV(text_pipe_logit, 
                          param_grid_logit, 
                          return_train_score=True, 
                          cv=3, n_jobs=-1)

grid_logit.fit(train_texts.comment, y_train.label)
grid_logit.best_params_, grid_logit.best_score_
def plot_grid_scores(grid, param_name):
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_train_score'],
        color='green', label='train')
    plt.plot(grid.param_grid[param_name], grid.cv_results_['mean_test_score'],
        color='red', label='test')
    plt.legend();
plot_grid_scores(grid_logit, 'logisticregression__C')
grid_logit.score(valid_texts.comment, y_valid.label)
