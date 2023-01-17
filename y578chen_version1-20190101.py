# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
from datetime import datetime, date



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.metrics.scorer import make_scorer

from sklearn.metrics import recall_score



from sklearn.preprocessing import LabelEncoder



from wordcloud import WordCloud,STOPWORDS

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



%matplotlib inline 
(market_train_df, news_train_df) = env.get_training_data()
data = []

#market_train_df['close'] = market_train_df['close'] / 20

for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:

    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()



    data.append(go.Scatter(

        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,

        y = price_df['close'].values,

        name = f'{i} quantile',

    ))

layout = go.Layout(dict(title = "Trends of closing prices by quantiles",

                        xaxis = dict(),

                        yaxis = dict(title = 'Price (USD)')),

                   legend=dict(orientation="h"))

py.iplot(dict(data=data, layout=layout), filename='basic-line')
print(market_train_df['open'].describe().apply(lambda x: format(x, 'f')))

print(market_train_df['close'].describe().apply(lambda x: format(x, 'f')))
market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']

grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]

g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)

trace = go.Scatter(

    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,

    y = g['price_diff']['std'].values,

    mode='markers',

    marker=dict(

        size = g['price_diff']['std'].values,

        color = g['price_diff']['std'].values,

        colorscale='Viridis',

        showscale=True

    ),

    text = g['min_text'].values

    #text = f"Maximum price drop: {g['price_diff']['min'].values}"

    #g['time'].dt.strftime(date_format='%Y-%m-%d').values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Top 10 biggest volatility of daily price',

    hovermode= 'closest',

    yaxis=dict(

        title= 'price_diff_std',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
market_train_df.sort_values('price_diff')[:10]
market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])





print(f"In {(market_train_df['close_to_open'] >= 2).sum()} lines price increased by 100% or more.")

print(f"In {(market_train_df['close_to_open'] <= 0.5).sum()} lines price decreased by 50% or more.")





market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')

market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')



# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.

for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():

    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):

        market_train_df.iloc[i,5] = row['assetName_mean_open']

    else:

        market_train_df.iloc[i,4] = row['assetName_mean_close']

        

for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.1].iterrows():

    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):

        market_train_df.iloc[i,5] = row['assetName_mean_open']

    else:

        market_train_df.iloc[i,4] = row['assetName_mean_close']
# after replacing the wired open and close price, we see them again

print(market_train_df['open'].describe().apply(lambda x: format(x, 'f')))

print(market_train_df['close'].describe().apply(lambda x: format(x, 'f')))
# maybe do not need this part





market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']

grouped = market_train_df.groupby(['time']).agg({'price_diff': ['std', 'min']}).reset_index()

g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]

g['min_text'] = 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)

trace = go.Scatter(

    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,

    y = g['price_diff']['std'].values,

    mode='markers',

    marker=dict(

        size = g['price_diff']['std'].values * 5,

        color = g['price_diff']['std'].values,

        colorscale='Viridis',

        showscale=True

    ),

    text = g['min_text'].values

    #text = f"Maximum price drop: {g['price_diff']['min'].values}"

    #g['time'].dt.strftime(date_format='%Y-%m-%d').values

)

data = [trace]



layout= go.Layout(

    autosize= True,

    title= 'Top 10 biggest volatility of daily price',

    hovermode= 'closest',

    yaxis=dict(

        title= 'price_diff',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig,filename='scatter2010')
market_train_df = market_train_df.loc[market_train_df['time'].dt.date >=date(2010, 1, 1)]



news_train_df = news_train_df.loc[news_train_df['time'].dt.date >=date(2010, 1, 1)]
print(market_train_df.shape)

print(news_train_df.shape)
print(market_train_df['assetCode'].nunique())

print(market_train_df['assetName'].nunique())
a = market_train_df[market_train_df.assetName == 'Unknown'].size 

print(a)
print(market_train_df['volume'].describe().apply(lambda x: format(x, 'f')))
# plot the volumn to see any pattern 

x = (market_train_df['volume'] - np.mean(market_train_df['volume']))/np.std(market_train_df['volume'])

y = (market_train_df['close'] - np.mean(market_train_df['close']))/np.std(market_train_df['close'])

with sns.axes_style("white"):

    sns.jointplot(x=x, y=y, kind="hex", color="k");
# plot the missing percentage

percent = (100 * market_train_df.isnull().mean()).sort_values(ascending=False)

percent.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 10)

plt.ylabel("Value Percent(%)", fontsize = 10)

plt.title("Total Missing Value by market_train_df", fontsize = 10)
def mis_impute(data):

    for i in data.columns:

        #if data[i].dtype == "object":

            #data[i] = data[i].fillna("other")

        if (data[i].dtype == "int64" or data[i].dtype == "float64"):

            data[i] = data[i].fillna(data[i].mean())

        else:

            pass

    return data



market_train_df = mis_impute(market_train_df)
# checking missing

market_train_df.isnull().any()
# plot the target variable 



data = []

market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']



price_df = market_train_df.groupby('time')['returnsOpenNextMktres10'].mean().reset_index()



data.append(go.Scatter(

    x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,

    y = price_df['returnsOpenNextMktres10'].values,

    name = f'{i} quantile'

))

layout = go.Layout(dict(title = "Trend of returnsOpenNextMktres10 ",

                  xaxis = dict(title = 'Month'),

                  yaxis = dict(title = 'Value'),

                  ),legend=dict(

                orientation="h"),)

py.iplot(dict(data=data, layout=layout), filename='basic-line')
corr = market_train_df[['returnsClosePrevRaw1','returnsClosePrevMktres1','returnsClosePrevRaw10','returnsClosePrevMktres10',

                        'returnsOpenPrevRaw1','returnsOpenPrevMktres1','returnsOpenPrevRaw10','returnsOpenPrevMktres10',

                        'returnsOpenNextMktres10']].corr()



corr.style.background_gradient().set_precision(2)

f,ax = plt.subplots(figsize=(10,8))

sns.heatmap(corr, annot=True, linewidths=.2, fmt= '.3f',ax=ax)
# '' convert to NA

news_train_df['headlineTag'] = news_train_df['headlineTag'].replace('', np.nan)  

news_train_df['headline'] = news_train_df['headline'].replace('', np.nan)  
text=news_train_df.headline.values[:100000]

wc= WordCloud(background_color="white",max_words=2000,stopwords=STOPWORDS)

wc.generate(" ".join(text))

plt.figure(figsize=(20,20))

plt.axis("off")

plt.title("Words could of Headlines", fontsize=30)

plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

plt.show()



percent = (100 * news_train_df.isnull().mean()).sort_values(ascending=False)

percent.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 10)

plt.ylabel("Value Percent(%)", fontsize = 10)

plt.title("Total Missing Value by news_train_df", fontsize = 10)
print(news_train_df['headlineTag'].value_counts(dropna=False))



# most news do not have tags


news_train_df.loc[news_train_df['time'] == news_train_df['sourceTimestamp']].shape[0]/len(news_train_df['time'])


news_train_df.loc[news_train_df['time'] == news_train_df['firstCreated']].shape[0]/len(news_train_df['time'])
news_train_df['sourceId'].value_counts()
news_train_df['urgency'].value_counts(dropna=False)
news_train_df['takeSequence'].value_counts(dropna = False)

sns.distplot(news_train_df['takeSequence'])
news_train_df['provider'].value_counts(dropna = False)

news_train_df['subjects'].value_counts(dropna = False) 
news_train_df['audiences'].value_counts()
# I noticed that there three columns have similar meaning

corr = news_train_df[['bodySize','wordCount','sentenceCount']].corr()

corr.style.background_gradient().set_precision(2)



f,ax = plt.subplots(figsize=(4,2))

sns.heatmap(corr, annot=True, linewidths=.1, fmt= '.3f',ax=ax)
sns.distplot(news_train_df.companyCount)
news_train_df['marketCommentary'].value_counts()

news_train_df['marketCommentary'] = news_train_df['marketCommentary'] *1

news_train_df['marketCommentary'].value_counts()
print(news_train_df['assetName'].nunique())

print(news_train_df['assetName'].nunique())
b = news_train_df[news_train_df.assetName == 'Unknown'].size 

print(b)
news_train_df['relevance'].describe().apply(lambda x: format(x, 'f'))
news_train_df['firstMentionSentence'].describe().apply(lambda x: format(x, 'f'))


news_train_df['sentimentClass'].value_counts()

news_train_df['sentimentWordCount'].describe().apply(lambda x: format(x, 'f'))
news_train_df['sentimentrelated'] = news_train_df['sentimentWordCount']/news_train_df['wordCount']

news_train_df['sentimentrelated'].describe().apply(lambda x: format(x, 'f'))

news_train_df['sentimentrelated'].corr(news_train_df['relevance'])
corr = news_train_df[['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D','noveltyCount7D']].corr()

corr.style.background_gradient().set_precision(2)



f,ax = plt.subplots(figsize=(4,3))

sns.heatmap(corr, annot=True, linewidths=.1, fmt= '.3f',ax=ax)
corr = news_train_df[['volumeCounts12H','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']].corr()

corr.style.background_gradient().set_precision(2)



f,ax = plt.subplots(figsize=(4,3))

sns.heatmap(corr, annot=True, linewidths=.1, fmt= '.3f',ax=ax)
def prepare_market_data(market_df):

    market_df['time'] = market_df['time'].dt.date

    market_df['average'] = (market_df['close'] + market_df['open'])/2

    market_df['beta'] = market_df['close_to_open']

    droplist = ['assetName','assetName_mean_open','assetName_mean_close','price_diff',

                'returnsClosePrevRaw1','returnsOpenPrevRaw1',

                'returnsClosePrevMktres1','returnsOpenPrevMktres1',

                'returnsClosePrevRaw10','returnsOpenPrevRaw10','close_to_open','close','open']

    market_df.drop(droplist, axis=1, inplace=True)



    



    return market_df
def prepare_news_data(news_df):

    news_df['position'] = news_df['firstMentionSentence'] / news_df['sentenceCount']



    droplist = ['sourceTimestamp','firstCreated','sourceId','headline',

                'takeSequence','provider','firstMentionSentence',

                'sentenceCount','bodySize','headlineTag','marketCommentary',

                'subjects','audiences','sentimentClass',

                'assetName', 'urgency','wordCount','sentimentWordCount']

    news_df.drop(droplist, axis=1, inplace=True)

    news_df['time'] = news_df['time'].dt.date



    # create a mapping between 'assetCode' to 'news_index'

    assets = []

    indices = []

    for i, values in news_df['assetCodes'].iteritems():

        assetCodes = eval(values)

        assets.extend(assetCodes)

        indices.extend([i]*len(assetCodes))

    mapping_df = pd.DataFrame({'news_index': indices, 'assetCode': assets})

    del assets, indices

    

    # join 'news_train_df' and 'mapping_df' (effectivly duplicating news entries)

    news_df['news_index'] = news_df.index.copy()

    expanded_news_df = mapping_df.merge(news_df, how='left', on='news_index')

    del mapping_df, news_df

    

    expanded_news_df.drop(['news_index', 'assetCodes'], axis=1, inplace=True)

    return expanded_news_df.groupby(['time', 'assetCode']).mean().reset_index()
market = market_train_df.copy()

news = news_train_df.copy()

market_df = prepare_market_data(market)

news_df = prepare_news_data(news)

merged_df = market_df.merge(news_df, how='left', on=['assetCode', 'time']).fillna(0)

merged_df.shape
# join news_df to market_df using ['assetCode', 'time']

merged_df = market_df.merge(news_df, how='left', on=['assetCode', 'time']).fillna(0)

merged_df.shape
merged_df.columns
col = [x for x in merged_df.columns if x not in ['assetCode', 'time', 'returnsOpenNextMktres10','universe']]



X = merged_df[col].values



y = (merged_df.returnsOpenNextMktres10 >= 0).astype(int).values
print(len(col))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {'learning_rate': 0.01, 

          'max_depth': 12, 

          'boosting': 'gbdt', 

          'objective': 'binary', 

          'metric': 'binary_logloss', 

          'is_training_metric': True, 

          'seed': 42}

model = lgb.train(params, d_train, 

                  num_boost_round = 2000,

                  #valid_sets = [d_train, lgb.Dataset(X_test, label=y_test)],

                  verbose_eval = 100, 

                  #early_stopping_rounds = 100

                 )



y_pred=model.predict(X_test)

y_pred[y_pred >= 0.5] = 1

y_pred[y_pred < 0.5] = 0



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

accuracy=accuracy_score(y_pred,y_test)

f1 = f1_score(y_pred,y_test)

recall = recall_score(y_pred,y_test)

print(accuracy,f1,recall)
importances = pd.DataFrame({'feature': list(merged_df[col].columns), 'importance': list(model.feature_importance())})

importances = importances.sort_values('importance',ascending = False)

print(importances)



# importances.plot.bar()



plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

feature = importances.feature

y_pos = np.arange(len(feature))

importance = importances.importance





ax.barh(y_pos, importance, align='center',

        color='red', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(feature)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Importances')





plt.show()
from xgboost import XGBClassifier

xgb = XGBClassifier(n_jobs = 4, n_estimators = 200, max_depth = 8, eta = 0.1)

xgb.fit(X_train,y_train)



accuracy = accuracy_score(xgb.predict(X_test),y_test)

f1 = f1_score(xgb.predict(X_test),y_test)

recall = recall_score(xgb.predict(X_test),y_test)



print(accuracy,f1,recall)

  
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, xgb.predict(X_test)))
importances = pd.DataFrame({'feature': list(merged_df[col].columns), 'importance': xgb.feature_importances_})

importances = importances.sort_values('importance',ascending = False)

print(importances)



# importances.plot.bar()



plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

feature = importances.feature

y_pos = np.arange(len(feature))

importance = importances.importance





ax.barh(y_pos, importance, align='center',

        color='red', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(feature)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Importances')





plt.show()
from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import roc_curve, auc
# Set the number of folds to 10

num_folds = 10

scoring = 'accuarcy'

# Append the models to the models list

models = []

models.append(('LR' , LogisticRegression()))

models.append(('LDA' , LinearDiscriminantAnalysis()))

models.append(('KNN' , KNeighborsClassifier()))

models.append(('CART' , DecisionTreeClassifier()))

models.append(('NB' , GaussianNB()))

models.append(('SVM' , SVC()))

models.append(('RF' , RandomForestClassifier(n_estimators=50)))



# Evaluate each algorithm for accuracy

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=42)

    cv_results = cross_val_score(model, X_train[:10000], y_train[:10000], cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Compare Algorithms

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# prepare the model LDA

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model_lda = LinearDiscriminantAnalysis()

model_lda.fit(rescaledX, y_train)

# estimate accuracy on validation dataset

rescaledValidationX = scaler.transform(X_test)

predictions = model_lda.predict(rescaledValidationX)



accuracy = accuracy_score(predictions,y_test)

f1 = f1_score(predictions,y_test)

precision = precision_score(predictions,y_test)



print(accuracy,f1,precision)


from catboost import CatBoostClassifier

import time



print('Training XGBoost')

t = time.time()

catb = CatBoostClassifier(thread_count=4, 

                          n_estimators=200, 

                          max_depth=8, eta=0.1, 

                          loss_function='Logloss', 

                          verbose=10)

catb.fit(X_train, y_train)



accuracy = accuracy_score(catb.predict(X_test),y_test)

f1 = f1_score(catb.predict(X_test),y_test)

recall = recall_score(catb.predict(X_test),y_test)



print(accuracy,f1,recall)
importances = pd.DataFrame({'feature': list(merged_df[col].columns), 'importance': catb.feature_importances_})

importances = importances.sort_values('importance',ascending = False)

print(importances)



# importances.plot.bar()



plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

feature = importances.feature

y_pos = np.arange(len(feature))

importance = importances.importance





ax.barh(y_pos, importance, align='center',

        color='red', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(feature)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Importances')





plt.show()
import dask 

from dask_ml.xgboost import XGBRegressor