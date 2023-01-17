import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=1.8)

import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import missingno as msno
import random

from plotly import tools

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
df_app = pd.read_csv('../input/AppleStore.csv')
df_description = pd.read_csv('../input/appleStore_description.csv')
df_app.isnull().sum()
df_description.head()
df_description.isnull().sum()
df_app['app_desc'] = df_description['app_desc']
df_app = df_app.iloc[:, 1:]
df_app.head()
df_app['size_bytes_in_MB'] = df_app['size_bytes'] / (1024 * 1024.0)
df_app['isNotFree'] = df_app['price'].apply(lambda x: 1 if x > 0 else 0)
df_app['isNotFree'].value_counts().plot.bar()
plt.xlabel('IsNotFree(Free == 0, NotFree == 1)')
plt.ylabel('Count')
plt.show()
df_app_notfree = df_app[df_app['isNotFree'] == 1]
df_app_free = df_app[df_app['isNotFree'] == 0]

print('There are {} Not-Free Apps in this dataset'.format(df_app_notfree.shape[0]))
print('There are {} Free Apps in this dataset'.format(df_app_free.shape[0]))
def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

cnt_srs = df_app['prime_genre'].value_counts()
text = ['{:.2f}%'.format(100 * (value / cnt_srs.sum())) for value in cnt_srs.values]

trace = go.Bar(
    x = cnt_srs.index,
    y = cnt_srs.values,
    text = text,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)
data = [trace]

layout = go.Layout(
    title = 'Prime genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Count'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs1 = df_app_free['prime_genre'].value_counts()
text1 = ['{:.2f}%'.format(100 * (value / cnt_srs1.sum())) for value in cnt_srs1.values]

trace1 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    text = text1,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

cnt_srs2 = df_app_notfree['prime_genre'].value_counts()
text2 = ['{:.2f}%'.format(100 * (value / cnt_srs2.sum())) for value in cnt_srs2.values]

trace2 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    text = text2,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

trace3 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    text = text1,
    opacity = 0.7,
    name='Free'
)


trace4 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    text = text2,
    opacity = 0.7,
    name='Not-Free'
)



fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{'colspan':2}, None]], 
                          subplot_titles=('(1) Countplot for Prime_genre of Free', '(2) Countplot for Prime_genre of Not-Free', 
                                          '(3) Grouped barplot containing Free(green) and Not-Free(red)'), print_grid=False)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)

fig['layout']['yaxis1'].update(title='Count')
fig['layout']['yaxis2'].update(title='Count')
fig['layout']['yaxis3'].update(title='Count')

fig['layout'].update(showlegend=False, width=800, height=800)

py.iplot(fig)
for app in df_app.loc[(df_app['isNotFree'] == 1) & (df_app['prime_genre'] == 'Social Networking'), 'track_name'].values:
    print(app)
cnt_srs = df_app[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)

trace = go.Bar(
    x = cnt_srs.index,
    y = cnt_srs.values,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)
data = [trace]

layout = go.Layout(
    title = 'User rating depending on Prime genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Mean User Rating'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs1 = df_app_free[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)
text1 = ['{:.2f}%'.format(100 * (value / cnt_srs1.sum())) for value in cnt_srs1.values]

trace1 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

cnt_srs2 = df_app_notfree[['prime_genre', 'user_rating']].groupby('prime_genre').mean()['user_rating'].sort_values(ascending=False)
text2 = ['{:.2f}%'.format(100 * (value / cnt_srs2.sum())) for value in cnt_srs2.values]

trace2 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

trace3 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    opacity = 0.7,
    name='Free'
)


trace4 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    opacity = 0.7,
    name='Not-Free'
)



fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{'colspan':2}, None]], 
                          subplot_titles=('(1) Mean user rating of Free', '(2) Mean user rating of Not-Free', 
                                          '(3) Grouped barplot containing Free(green) and Not-Free(red)'), print_grid=False)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)

fig['layout']['yaxis1'].update(title='Mean User Rating')
fig['layout']['yaxis2'].update(title='Mean User Rating')
fig['layout']['yaxis3'].update(title='Mean User Rating')
fig['layout'].update(showlegend=False, width=800, height=800)

py.iplot(fig)
cnt_srs = df_app[['prime_genre', 'size_bytes_in_MB']].groupby('prime_genre').mean()['size_bytes_in_MB'].sort_values(ascending=False)
text = ['{:.2f}'.format(value) for value in cnt_srs.values]

trace = go.Bar(
    x = cnt_srs.index,
    y = cnt_srs.values,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)
data = [trace]

layout = go.Layout(
    title = 'Mean App size(MB) depending on Prime genre',
    margin = dict(
        l = 100
    ),
    xaxis = dict(
        title = 'Genre'
    ),
    yaxis = dict(
        title = 'Mean App size'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cnt_srs1 = df_app_free[['prime_genre', 'size_bytes_in_MB']].groupby('prime_genre').mean()['size_bytes_in_MB'].sort_values(ascending=False)
text1 = ['{:.2f}%'.format(100 * (value / cnt_srs1.sum())) for value in cnt_srs1.values]

trace1 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    text = text1,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

cnt_srs2 = df_app_notfree[['prime_genre', 'size_bytes_in_MB']].groupby('prime_genre').mean()['size_bytes_in_MB'].sort_values(ascending=False)
text2 = ['{:.2f}%'.format(100 * (value / cnt_srs2.sum())) for value in cnt_srs2.values]

trace2 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    text = text2,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)

trace3 = go.Bar(
    x = cnt_srs1.index,
    y = cnt_srs1.values,
    text = text1,
    opacity = 0.7,
    name='Free'
)


trace4 = go.Bar(
    x = cnt_srs2.index,
    y = cnt_srs2.values,
    text = text2,
    opacity = 0.7,
    name='Not-Free'
)



fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{'colspan':2}, None]], subplot_titles=('Free', 'Not-Free', 'third'), print_grid=False)
fig = tools.make_subplots(rows=2, cols=2, specs = [[{}, {}], [{'colspan':2}, None]], 
                          subplot_titles=('(1) Mean App size(MB) of Free', '(2) Mean App size(MB) of Not-Free', 
                                          '(3) Grouped barplot containing Free(green) and Not-Free(red)'), print_grid=False)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.append_trace(trace3, 2, 1)
fig.append_trace(trace4, 2, 1)

fig['layout']['yaxis1'].update(title='Mean App size(MB)')
fig['layout']['yaxis2'].update(title='Mean App size(MB)')
fig['layout']['yaxis3'].update(title='Mean App size(MB)')
fig['layout'].update(showlegend=False, width=800, height=800)

py.iplot(fig)
cnt_srs = df_app_notfree[['prime_genre', 'price']].groupby('prime_genre').mean()['price'].sort_values(ascending=False)
text = ['{:.2f}%'.format(100 * (value / cnt_srs.sum())) for value in cnt_srs.values]

trace = go.Bar(
    x = cnt_srs.index,
    y = cnt_srs.values,
    text = text,
    marker = dict(
        color = random_color_generator(100),
        line = dict(color='rgb(8, 48, 107)',
                    width = 1.5)
    ),
    opacity = 0.7
)
data = [trace]

layout = go.Layout(
    title = 'Mean App price of Not-Free Apps',
    margin = dict(
        l = 100
    ),
    yaxis = dict(
        title = 'Mean App price'
    ),
    width = 800,
    height = 500
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sns.lmplot(x='user_rating', y='user_rating_ver', hue='isNotFree', data=df_app)
df_temp = df_app.drop('id', axis=1)
data = [
    go.Heatmap(
        z = df_temp.corr().values,
        x = df_temp.corr().columns.values,
        y = df_temp.corr().columns.values,
        colorscale='YlGnBu',
        reversescale=False,
    )
]

layout = go.Layout(
    title='Pearson Correlation of float-type features',
    xaxis = dict(ticks=''),
    yaxis = dict(ticks='' ),
    width = 800, height = 800,
    margin = dict(
        l = 100
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')
df_corr = df_app.drop('id', axis=1).corr()
df_corr['user_rating'].sort_values(ascending=False)
df_corr['price'].sort_values(ascending=False)
plt.scatter(df_app['user_rating'], df_app['rating_count_ver'])
foo = df_app['app_desc'][0].split(' ')
import nltk
from nltk.corpus import stopwords
%%time
for i in range(df_app.shape[0]):
    temp_desc = df_app['app_desc'][i]
    temp_word_list = nltk.word_tokenize(temp_desc)
    temp_word_list = [word.lower() for word in temp_word_list if word not in stopwords.words('english')]
    for char in " {}()#&[]^`´-_·@|¿?¡!'+*\"?.!/;:<>’•“”–»%■,":
        for ele in temp_word_list:
            if char in ele:
                temp_word_list.remove(ele)
    fdist = nltk.FreqDist(temp_word_list)
    temp_srs = pd.Series(fdist).sort_values(ascending=False)
    try:
        df_app.loc[i, 'most_freq_word_1'] = temp_srs.index[0]
        df_app.loc[i, 'most_freq_word_2'] = temp_srs.index[1]
        df_app.loc[i, 'most_freq_word_3'] = temp_srs.index[2]
    except:
        df_app.loc[i, 'most_freq_word_1'] = temp_srs.index[0]
df_app.loc[df_app['user_rating'] > 4, 'most_freq_word_3'].value_counts().head(20).plot.bar()
freq_total = nltk.FreqDist(df_app['most_freq_word_1'].tolist() + 
              df_app['most_freq_word_2'].tolist() +
             df_app['most_freq_word_3'].tolist())
freq_total = pd.Series(freq_total).sort_values(ascending=False)
freq_total.head(20).plot.bar()
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
df_app['rating_count_before'] = df_app['rating_count_tot'] - df_app['rating_count_ver']
df_app.head()
df_train = df_app[['size_bytes_in_MB', 'isNotFree', 'price', 'rating_count_before', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic', 'prime_genre']]
target = df_app['user_rating']

df_train = pd.get_dummies(df_train)

def categorize_rating(x):
    if x <= 4:
        return 0
    else:
        return 1

target = target.apply(categorize_rating)

target.astype(str).hist()
X_train, X_test, y_train, y_test = train_test_split(df_train.values, target, test_size=0.2, random_state=1989, stratify=target)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
models = [RandomForestClassifier(), LGBMClassifier(), XGBClassifier()]

kfold = KFold(n_splits=5, random_state=1989)

clf_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score'])

for i, model in enumerate(models):
    clf = model
    cv_result = cross_validate(model, X_train, y_train, cv=kfold, scoring='accuracy')
    clf_comparison.loc[i, 'Classfier_name'] = model.__class__.__name__
    clf_comparison.loc[i, 'train_score'] = cv_result['train_score'].mean()
    clf_comparison.loc[i, 'test_score'] = cv_result['test_score'].mean()

clf_comparison
df_app.loc[:, 'isGame'] = df_app['app_desc'].apply(lambda x: 1 if 'game' in x.lower() else 0)
df_app.loc[:, 'descLen'] = df_app['app_desc'].apply(lambda x: len(x.lower()))
df_train = df_app[['size_bytes_in_MB', 'isNotFree', 'price', 'rating_count_before', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic', 'prime_genre', 'isGame', 'descLen']]
target = df_app['user_rating']

df_train = pd.get_dummies(df_train)

def categorize_rating(x):
    if x <= 4:
        return 0
    else:
        return 1

target = target.apply(categorize_rating)

target.astype(str).hist()
X_train, X_test, y_train, y_test = train_test_split(df_train.values, target, test_size=0.2, random_state=1989, stratify=target)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
models = [RandomForestClassifier(), LGBMClassifier(), XGBClassifier()]

kfold = KFold(n_splits=5, random_state=1989)

clf_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score'])

for i, model in enumerate(models):
    clf = model
    cv_result = cross_validate(model, X_train, y_train, cv=kfold, scoring='accuracy')
    clf_comparison.loc[i, 'Classfier_name'] = model.__class__.__name__
    clf_comparison.loc[i, 'train_score'] = cv_result['train_score'].mean()
    clf_comparison.loc[i, 'test_score'] = cv_result['test_score'].mean()

clf_comparison


