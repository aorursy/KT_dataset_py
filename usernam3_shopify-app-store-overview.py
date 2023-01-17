import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

# source
apps = pd.read_csv('../input/apps.csv')
categories = pd.read_csv('../input/categories.csv')
reviews = pd.read_csv('../input/reviews.csv')
apps.head()
categories.head()
reviews.head()
categories['category_id'] = categories['category'].factorize()[0]
apps_and_reviews = pd.merge(apps, reviews, how='right', left_on='url', right_on='app_url')
apps_with_categories = pd.merge(apps, categories, how='right', left_on='url', right_on='app_url')
apps_and_reviews_with_categories = pd.merge(apps_and_reviews, categories, how='right', left_on='app_url', right_on='app_url')
reviews_count_check = pd.merge(
    apps[['url', 'reviews_count']], 
    reviews.groupby(['app_url']).size().reset_index(name='reviews_available_count'), 
    how='left', left_on='url', right_on='app_url')

reviews_count_check[['reviews_available_count']] = reviews_count_check[['reviews_available_count']].fillna(value=0)
reviews_count_check['diff'] = reviews_count_check['reviews_available_count'] - reviews_count_check['reviews_count']
reviews_count_check.loc[reviews_count_check['diff'] != 0].drop_duplicates(subset=['url'])
import holoviews as hv

hv.extension('bokeh')
%output size=250

category_id_df = apps_with_categories[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)

results = pd.DataFrame()

for cat in categories['category'].unique():
    app_urls = categories[categories['category'] == cat]['app_url'].values
    category_connections = categories[(categories['app_url'].isin(app_urls)) & (categories['category'] != cat)]\
        .groupby(['category']) \
        .size() \
        .reset_index(name='connections')
    category_connections['from'] = cat
    category_connections['to'] = category_connections['category']
    
    category_connections['source']=category_connections['from'].map(category_to_id)
    category_connections['target']=category_connections['to'].map(category_to_id)
    category_connections['value']=category_connections['connections']
    
    results = pd.concat([results, category_connections[['source', 'target', 'value']]])

source = results[['source', 'target']].min(axis=1)
target = results[['source', 'target']].max(axis=1)
results['source'] = source
results['target'] = target
results = results.drop_duplicates(subset=['source', 'target'])

nodes_data = categories[['category']].drop_duplicates()
nodes_data['index'] = categories['category'].map(category_to_id)

nodes = hv.Dataset(nodes_data, 'index')

%opts Chord [label_index='category' color_index='index' edge_color_index='source'] 

hv.Chord((results, nodes))
apps_per_category = apps_with_categories\
    .groupby(['category']) \
    .size() \
    .reset_index(name='apps_count') \
    .sort_values('apps_count', ascending=False)

plotly.offline.iplot({
    'data': [go.Pie(labels=apps_per_category['category'], values=apps_per_category['apps_count'])],
    'layout': go.Layout(title='Apps per category')
})
apps_and_reviews_grouped_by_category = apps_and_reviews_with_categories\
    .groupby(['category']) \
    .size() \
    .reset_index(name='reviews') \
    .sort_values('reviews', ascending=False)

plotly.offline.iplot({
    'data': [go.Pie(labels=apps_and_reviews_grouped_by_category['category'], values=apps_and_reviews_grouped_by_category['reviews'])],
    'layout': go.Layout(title='Reviews per category')
})
apps_and_reviews_grouped_by_app_title = apps_and_reviews \
    .groupby(['title']) \
    .size() \
    .reset_index(name='reviews') \
    .sort_values('reviews', ascending=False)

limit = 15
plotly.offline.iplot({
    'data': [go.Bar(
        name='Reviews',
        x=apps_and_reviews_grouped_by_app_title.head(limit)['title'],
        y=apps_and_reviews_grouped_by_app_title.head(limit)['reviews']
    )],
    'layout': go.Layout(title='Apps ordered by number of reviews', margin=go.layout.Margin(b=200))
})
reviews_in_category = (apps_and_reviews_with_categories.groupby(['category', 'title'], as_index=False)['rating_x'].mean())\
    .dropna(subset=['rating_x'])\
    .groupby(['category'], as_index=False)\
    .mean()\
    .sort_values('rating_x', ascending=True)
reviews_in_category[['mean_rating']] = reviews_in_category[['rating_x']]
reviews_in_category = reviews_in_category.drop(columns=['rating_x'])

plotly.offline.iplot({
    'data': [go.Bar(
        name='Rating',
        x=reviews_in_category['mean_rating'],
        y=reviews_in_category['category'],
        orientation = 'h'
    )],
    'layout': go.Layout(title='Average rating per category', margin=go.layout.Margin(l=250))
})
each_rating_count_in_category = pd.DataFrame({
    'category': apps_and_reviews_with_categories['category'].unique(),
    'r_count': [
        apps_and_reviews_with_categories.loc[apps_and_reviews_with_categories['category'] == cat].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_1_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 1.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_2_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 2.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_3_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 3.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_4_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 4.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
    'r_5_count': [
        apps_and_reviews_with_categories.loc[(apps_and_reviews_with_categories['rating_y'] == 5.0) & (apps_and_reviews_with_categories['category'] == cat)].shape[0]
        for cat in apps_and_reviews_with_categories['category'].unique()],
})
each_rating_count_in_category = each_rating_count_in_category.loc[each_rating_count_in_category.category.notna()]

groups = ['r_1_count', 'r_2_count', 'r_3_count', 'r_4_count', 'r_5_count']
traces = []
x_axis_lables = ['Share of 1 star ratings', 'Share of 2 star ratings', 'Share of 3 star ratings', 'Share of 4 star ratings', 'Share of 5 star ratings']

for idx, row in each_rating_count_in_category.iterrows():
    traces.append(go.Bar(
        x=x_axis_lables,
        y=list((row[groups] / row['r_count'])),
        name=row['category']
    ))

layout = go.Layout(
    title='Share of reviews per rating per category',
    yaxis = dict(
        tickformat='.2%'
    )
)

plotly.offline.iplot(go.Figure(data=traces, layout=layout))

each_rating_count_in_category
from plotly import tools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

apps_with_categories['normalized_description'] = (apps_with_categories['description'].map(str) + apps_with_categories['pricing'])

category_id_df = apps_with_categories[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', ngram_range=(2, 5), stop_words='english')
features = tfidf.fit_transform(apps_with_categories['normalized_description']).toarray()
labels = apps_with_categories['category_id']

N = 15
fig = tools.make_subplots(rows=6, cols=2, 
                          shared_yaxes=False, shared_xaxes=False,
                          horizontal_spacing=0.5, print_grid=False, 
                          subplot_titles=["'{0}' term scores".format(entry[0]) for entry in category_to_id.items()])

for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])    
    feature_names = np.array(tfidf.get_feature_names())[indices]
    scores = features_chi2[0][indices]

    category_significant_terms = pd.DataFrame({'term': feature_names, 'score': scores})\
        .sort_values('score', ascending=True)\
        .tail(N)

    row = int(category_id / 2) + 1
    col = 1 if (category_id % 2 == 0) else 2
    bar_chart = go.Bar(
        name=category, 
        x=category_significant_terms['score'], 
        y=category_significant_terms['term'], 
        orientation='h'
    )
    fig.append_trace(bar_chart, row, col)

    
fig['layout'].update(title='Category terms', 
                     height=1024, width=1024, 
                     margin=go.layout.Margin(l=225, r=225), showlegend=False)
plotly.offline.iplot(fig)
reviews_with_low_ratings = pd.DataFrame()
for category in apps_and_reviews_with_categories['category'].dropna().unique():
    reviews_in_category = apps_and_reviews_with_categories[apps_and_reviews_with_categories['category'] == category]
    reviews_with_rating_lower_than_app_rating = reviews_in_category[(reviews_in_category['rating_y'] < reviews_in_category['rating_x']) & \
                                                        (reviews_in_category['reviews_count'] > 0)]
    reviews_with_low_ratings = pd.concat([reviews_with_low_ratings, reviews_with_rating_lower_than_app_rating])
reviews_with_low_ratings.head()
from plotly import tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

reviews_with_low_ratings = reviews_with_low_ratings.dropna(subset=['body'])
reviews_with_low_ratings['negative_prob'] = reviews_with_low_ratings['body'].apply(lambda body: analyser.polarity_scores(body)["neg"])
negative_reviews_with_low_rating = reviews_with_low_ratings[reviews_with_low_ratings['negative_prob'] >= 0.5]

category_id_df = negative_reviews_with_low_rating[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, 
                        min_df=3, 
                        norm='l2',
                        ngram_range=(2, 5), 
                        stop_words='english')

features = tfidf.fit_transform(negative_reviews_with_low_rating['body']).toarray()
labels = negative_reviews_with_low_rating['category_id']

N = 15
fig = tools.make_subplots(rows=6, cols=2, 
                          shared_yaxes=False, shared_xaxes=False,
                          horizontal_spacing=0.5, print_grid=False, 
                          subplot_titles=["'{0}' term scores".format(entry[0]) for entry in category_to_id.items()])

charts = []

for category, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])    
    feature_names = np.array(tfidf.get_feature_names())[indices]
    scores = features_chi2[0][indices]

    category_significant_terms = pd.DataFrame({'term': feature_names, 'score': scores})\
        .sort_values('score', ascending=True)\
        .tail(N)

    if category_significant_terms.shape[0] == 0:
        next

    bar_chart = go.Bar(
        name=category, 
        x=category_significant_terms['score'], 
        y=category_significant_terms['term'], 
        orientation='h'
    )
    charts.append(bar_chart)

for number, chart in enumerate(charts):
    row = int(number / 2) + 1
    col = 1 if (number % 2 == 0) else 2
    fig.append_trace(chart, row, col)
    
fig['layout'].update(title='Terms of low rating reviews', 
                     height=1024, width=1024, 
                     margin=go.layout.Margin(l=225, r=225), showlegend=False)
plotly.offline.iplot(fig)
limit = 15

apps_per_developer = apps\
    .groupby(['developer']) \
    .size() \
    .reset_index(name='apps_count') \
    .sort_values('apps_count', ascending=False)

plotly.offline.iplot({
    'data': [go.Bar(
            name='Summary rating',
            x=apps_per_developer.head(limit)['developer'],
            y=apps_per_developer.head(limit)['apps_count']
    )],
    'layout': go.Layout(title='Developers ordered by the number of apps', margin=go.layout.Margin(b=100))
})
limit = 15

apps_and_reviews_grouped_by_developer = apps_and_reviews \
    .groupby(['developer']) \
    .agg({'rating_y': ['size', 'mean', 'sum']}) \
    .reset_index() \
    .sort_values(('rating_y', 'sum'), ascending=False)


plotly.offline.iplot({
    'data': [go.Bar(
            name='Summary rating',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'sum')]
    ),go.Bar(
            name='Reviews',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'size')]
    ),go.Scatter(
            name='Rating',
            x=apps_and_reviews_grouped_by_developer.head(limit)['developer'],
            y=apps_and_reviews_grouped_by_developer.head(limit)[('rating_y', 'mean')],
            yaxis='y2'
    )],
    'layout': go.Layout(
        title='Developers with the highest summary rating',
        legend=dict(x=1.25, y=1),
        barmode='group',
        yaxis2=dict(
            overlaying='y',
            anchor='x',
            side='right'
        )
    )
})

apps_with_rating_scores = apps_with_categories[(apps_with_categories['reviews_count'] > 0)].copy()
apps_with_rating_scores['rating_mult_by_reviews'] = apps_with_rating_scores['rating'] * apps_with_rating_scores['reviews_count']

apps_with_highest_score = apps_with_rating_scores\
    .loc[apps_with_rating_scores.groupby(['category'])['rating_mult_by_reviews'].idxmax()]

apps_with_highest_score[['category', 'app_url', 'title', 'rating', 'reviews_count']]