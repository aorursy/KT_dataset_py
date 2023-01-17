# !pip install pandas-profiling[notebook]

import json, requests, re, time, datetime
import pandas as pd, numpy as np
import plotly.express as px, plotly.io as pio, plotly.graph_objects as go
from scipy            import stats
from plotly.subplots  import make_subplots
from IPython.display  import Image
from pandas_profiling import ProfileReport
from tqdm.notebook    import tqdm
from sklearn.cluster  import KMeans
from sklearn          import preprocessing
from sklearn.tree     import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics  import mean_squared_error

pio.templates.default = "plotly_white"
Image(filename='../input/competitivedatascienceschema/schema.png') 
data = pd.read_csv('../input/zr-pfs-data/pfs_data.csv')
test = pd.read_csv('../input/pfs-test/pfs_test.csv')

data = data.drop(columns=['Unnamed: 0'])
test = test.drop(columns=['Unnamed: 0']) \
           .set_index('ID')
data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
# profile = ProfileReport(data, title='Predict Future Sales')
# profile.to_file("pfs_data_profile.html")
# profile.to_notebook_iframe()
n_missing_shops_from_data = len(test.loc[~test['shop_id'].isin(data['shop_id']), 'shop_id'].unique())
n_missing_items_from_data = len(test.loc[~test['item_id'].isin(data['item_id']), 'item_id'].unique())

n_missing_shops_from_test = len(data.loc[~data['shop_id'].isin(test['shop_id']), 'shop_id'].unique())
n_missing_items_from_test = len(data.loc[~data['item_id'].isin(test['item_id']), 'item_id'].unique())

print(f'There are {n_missing_shops_from_test} shops and {n_missing_items_from_test} items in the data set that are not found in test set')
print(f'There are {n_missing_shops_from_data} shops and {n_missing_items_from_data} items in the test set that are not found in data set')
f"{len(data[data['item_cnt_day']==1.])*100 / len(data):.2f}% of items had a qty sold of 1"
print('How many unique shops sold a specific item on a given month')
pivot = pd.pivot_table(data, index=['item_name_en'], columns=['date_block_num'], values=['shop_id'], aggfunc=pd.Series.nunique)
pivot.reindex(pivot['shop_id'].sort_values(by=0, ascending=False).index)
cod_moscow_sales = data[(data['item_name_en']=='Call of Duty: Black Ops II [PC, Jewel, Russian]')&(data['shop_name_en']=='Moscow shopping center Semenovsky')] \
                   .groupby('date_block_num').agg({
                       'item_price'  : ['min', 'max', 'mean'],
                       'item_cnt_day': ['min', 'max', 'sum'] 
                   })

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=cod_moscow_sales.index, y=cod_moscow_sales['item_price']['max'],
    fill=None,
    mode='lines',
    name='Max Price'
), secondary_y=False)

fig.add_trace(go.Scatter(x=cod_moscow_sales.index, y=cod_moscow_sales['item_price']['min'],
    fill='tonexty',
    mode='lines',
    name='Min Price'
), secondary_y=False)
fig.add_trace(go.Scatter(x=cod_moscow_sales.index, y=cod_moscow_sales['item_cnt_day']['sum'],
    mode='markers',
    fill='tozeroy',
    name='Qty Sold'
), secondary_y=True)

fig.update_layout(title_text="Prices and Quantities of COD Black Ops sold by the Moscow Shopping Center By Month")
fig.update_xaxes(title_text="Month")
fig.update_yaxes(title_text="Price Min/Max", secondary_y=False)
fig.update_yaxes(title_text="Qty Sold", secondary_y=True)

fig
cod_moscow_sales.iloc[np.r_[0:5, -5:0]]
fig = px.bar(data['item_category_name_en'].value_counts(), title='Transactions by Item Categories')
fig.update_layout(xaxis_tickangle=-45)
transactions_by_date = data[data['date_block_num']<=12].groupby('date').agg({
    'item_id': 'count',
    'date_block_num': 'mean'
})
px.bar(x=transactions_by_date.index, y=transactions_by_date['item_id'], color=transactions_by_date['date_block_num'].astype(str), title='Transactions by Day')
fig = px.bar(pd.DataFrame(data['shop_name_en'].value_counts()).reset_index(), x='index', y='shop_name_en', text='shop_name_en', title='Transactions by Shop')
fig.update_layout(xaxis_tickangle=-45)
fig = make_subplots(
    rows=32, 
    cols=2, 
    specs=np.reshape([{'colspan': 2, 'rowspan': 2}, {}] + [{} for _ in range(62)], (32, 2)).tolist(),
    subplot_titles=['Total', '', '', ''] + list(data['shop_name_en'].unique())
)
# total
fig.add_trace(
    go.Bar(x=list(data['date_block_num'].value_counts().index), y=data['date_block_num'].value_counts()),
    row=1, col=1
)
# Individual Shop plots
for i, shop_name in enumerate(data['shop_name_en'].unique()):
    row = (i+6) // 2
    col = ((i+6) % 2) + 1
    values = data.loc[data['shop_name_en']==shop_name, 'date_block_num'].value_counts()
    
    fig.add_trace(
        go.Bar(x=list(values.index), y=values.values),
        row=row, col=col
    )

fig.update_layout(
    autosize=False,
    width=1200,
    height=5000,
    showlegend=False
)
moscow_sales_by_cat_date = data.loc[data['shop_name_en']=='Moscow shopping center Semenovsky'].groupby(['date_block_num', 'item_category_name_en']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['date_block_num', 'item_id'], ascending=[True,False])

px.bar(
    x     = moscow_sales_by_cat_date.index.get_level_values(0), 
    y     = moscow_sales_by_cat_date.values.ravel(), 
    color = moscow_sales_by_cat_date.index.get_level_values(1),
    title = 'Transactions by Category and Month for the Moscow Shopping Center'
)
kolomna_sales_by_cat_date = data.loc[data['shop_name_en']=='Kolomna TC Rio'].groupby(['date_block_num', 'item_category_name_en']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['date_block_num', 'item_id'], ascending=[True,False])

px.bar(
    x     = kolomna_sales_by_cat_date.index.get_level_values(0), 
    y     = kolomna_sales_by_cat_date.values.ravel(), 
    color = kolomna_sales_by_cat_date.index.get_level_values(1),
    title = 'Transactions by Category and Month for the Kolomnia Shopping Center'
)
digital_warehouse_sales_by_cat_date = data.loc[data['shop_name_en']=='Digital warehouse 1C-Online'].groupby(['date_block_num', 'item_category_name_en']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['date_block_num', 'item_id'], ascending=[True,False])

px.bar(
    x     = digital_warehouse_sales_by_cat_date.index.get_level_values(0), 
    y     = digital_warehouse_sales_by_cat_date.values.ravel(), 
    color = digital_warehouse_sales_by_cat_date.index.get_level_values(1),
    title = 'Transactions by Category and Month for the Digital Warehouse'
)
offsite_trade_sales_by_cat_date = data.loc[data['shop_name_en']=='Offsite Trade'].groupby(['date_block_num', 'item_category_name_en']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['date_block_num', 'item_id'], ascending=[True,False])

px.bar(
    x     = offsite_trade_sales_by_cat_date.index.get_level_values(0), 
    y     = offsite_trade_sales_by_cat_date.values.ravel(), 
    color = offsite_trade_sales_by_cat_date.index.get_level_values(1),
    title = 'Transactions by Category and Month for the Offsite Trade'
)
# Create a second GroupBy that just sums total sales per month to divide by
moscow_sales_by_date = data.loc[data['shop_name_en']=='Moscow shopping center Semenovsky'].groupby(['date_block_num']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['item_id'], ascending=[False])

moscow_pct_of_sales_by_cat_date = moscow_sales_by_cat_date.div(moscow_sales_by_date, level='date_block_num') * 100

fig = px.line(
    pd.DataFrame(moscow_pct_of_sales_by_cat_date).reset_index(), 
    x='date_block_num',
    y='item_id',
    color='item_category_name_en',
    title='Pct of Total Monthly Sales per Item Category for Moscow Shopping Center'
)

fig.update_layout({'height': 800})
# Create a second GroupBy that just sums total sales per month to divide by
digital_warehouse_sales_by_date = data.loc[data['shop_name_en']=='Digital warehouse 1C-Online'].groupby(['date_block_num']) \
    .agg({'item_id': 'count'}) \
    .sort_values(['item_id'], ascending=[False])

digital_warehouse_pct_of_sales_by_cat_date = digital_warehouse_sales_by_cat_date.div(digital_warehouse_sales_by_date, level='date_block_num') * 100

fig = px.line(
    pd.DataFrame(digital_warehouse_pct_of_sales_by_cat_date).reset_index(), 
    x='date_block_num',
    y='item_id',
    color='item_category_name_en',
    title='Pct of Total Monthly Sales per Item Category for Digital Warehouse'
)

fig.update_layout({'height': 800})
def seach_movie_database(movie_name, language='en-US', is_tv_show=False):

    movie_seach_url = f'https://api.themoviedb.org/3/search/' + ('tv' if is_tv_show else 'movie')
    
    params = {
        'api_key'      : '2ef06af48197ad9382d7bbc2a84e242c',
        'language'     : 'en-US',
        'query'        : movie_name,
        'include_adult': 'false'
    }
    
    r = requests.get(movie_seach_url, params=params)
    
    if r.status_code == 200:
        return r.json()
    return {'total_results': 0}

# create a sample of n movies
def get_movie_sample(n_samples=200):
    sample_movies = data.loc[data['item_category_name_en']=='Movie - DVD', ['item_name', 'item_name_en']].sample(n_samples)
    sample_movies_en = pd.DataFrame(sample_movies['item_name_en'])
    sample_movies_ru = pd.DataFrame(sample_movies['item_name'])

    sample_movies_en['result_count'] = None
    sample_movies_ru['result_count'] = None

    # Remove all items in parentheses, trim and lowercase
    sample_movies_en['item_name_en'] = sample_movies_en['item_name_en'] \
        .str.replace("\(.+\)", '', regex=True) \
        .str.lower() \
        .str.strip()

    sample_movies_ru['item_name'] = sample_movies_ru['item_name'] \
        .str.replace("\(.+\)", '', regex=True) \
        .str.lower() \
        .str.strip()

    sample_movies_en = sample_movies_en.set_index('item_name_en')
    sample_movies_ru = sample_movies_ru.set_index('item_name')
    
    return sample_movies_en, sample_movies_ru

def get_result_counts(movie_samples, language='en-US'):
    for movie in movie_samples.index:
        result_count = seach_movie_database(movie, language=language)['total_results']
        movie_samples.loc[movie] = result_count
    
    return movie_samples

def score_results_found(movie_samples):
    n_results = len(movie_samples[(movie_samples['result_count']>1) & (movie_samples['result_count']<50)])
    return n_results / len(movie_samples)

english_results = []
russian_results = []
for _ in tqdm(range(10)):
    sample_movies_en, sample_movies_ru = get_movie_sample()
    get_result_counts(sample_movies_en)
    get_result_counts(sample_movies_ru, language='ru')
    
    english_results.append(score_results_found(sample_movies_en))
    russian_results.append(score_results_found(sample_movies_ru))
    
    time.sleep(2)

f'English results: {np.mean(english_results)} - Russian results: {np.mean(russian_results)}'
shop_data = data[['shop_id', 'date', 'date_block_num', 'shop_name_en', 'item_price', 'item_cnt_day', 'item_category_name_en', 'item_name_en']].copy()
shop_data = shop_data.set_index('shop_id')

shop_data.sample(5).T
# Avg sales per month and total sales for each store
shop_data = shop_data.join(shop_data['item_cnt_day'].groupby('shop_id').sum().rename('total_sales'))
shop_data = shop_data.join(
    shop_data[['date_block_num','item_cnt_day']] \
    .groupby(['shop_id', 'date_block_num']) \
    .sum() \
    .reset_index(level=1, drop=True) \
    .groupby('shop_id') \
    .mean() \
    .rename({'item_cnt_day':'avg_monthly_sales'}, axis=1)
)

# Calculate slope of monthly sales to look at MoM sales performance
import warnings
warnings.simplefilter('ignore', np.RankWarning)

shop_data = shop_data.join(
    shop_data.pivot_table(index=['shop_id'], columns=['date_block_num'], values=['item_name_en'], aggfunc='count').apply(
        lambda row: np.polyfit(row.dropna().index.get_level_values(1), row.dropna().values, 1)[0],
        axis=1
    )
    .rename('sales_slope')
)
# How many months shops were open in past 12 months
shop_data['months_opened_in_last_12'] = 0
shop_data.update(
    shop_data.loc[shop_data['date_block_num'] >= shop_data['date_block_num'].max() - 12, 'date_block_num']
    .groupby('shop_id')
    .nunique()
    .rename('months_opened_in_last_12')
)

# Was shop open last month?
shop_data['open_last_month'] = 0
shop_data.loc[shop_data.index.isin(shop_data.loc[shop_data['date_block_num'] == 33, 'open_last_month'].index), 'open_last_month'] = 1

# Create col with day number, starting from the first date in training set
shop_data['day'] = (shop_data['date'] - shop_data['date'].min()).dt.days
# How mnay different item categories did each shop sell (looking for general vs specific stores)
shop_data = shop_data.join(shop_data.groupby('shop_id')['item_category_name_en'].nunique().rename('n_categories_sold'))

# Convert item categories to one-hot
shop_data = pd.concat([shop_data, pd.get_dummies(shop_data['item_category_name_en'])], axis=1)

# Variability in price?

# Avg item price?

# avg qty for bulk reseller?
shop_cols = ['shop_name_en' ,'total_sales', 'avg_monthly_sales', 'months_opened_in_last_12', 'open_last_month', 'n_categories_sold', 'sales_slope']
shops = shop_data.loc[~shop_data['shop_name_en'].duplicated(), shop_cols]
shops = shops.reset_index(drop=False).set_index(['shop_id', 'shop_name_en'])
shops[['total_sales', 'avg_monthly_sales']].hist()
# log transform sales rcords
shops['total_sales']       = np.log(shops['total_sales'])
shops['avg_monthly_sales'] = np.log(shops['avg_monthly_sales'])

# Scale shop continuous data
shops['total_sales']       = preprocessing.scale(shops['total_sales'])
shops['avg_monthly_sales'] = preprocessing.scale(shops['avg_monthly_sales'])
shops['n_categories_sold'] = preprocessing.scale(shops['n_categories_sold'])
shops['sales_slope']       = preprocessing.scale(shops['sales_slope'])
shops[['total_sales', 'avg_monthly_sales']].hist()
# Update shop_data with scaled features
shop_data = shop_data.set_index(['shop_name_en'], append=True)
shop_data.update(shops)
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(shops)
fig, ax = plt.subplots(figsize=(20, 30))
plot_dendrogram(model, truncate_mode='level', orientation="right", labels=shops.index.get_level_values(1), leaf_rotation=0, leaf_font_size=14)
fig.show()
# Scale item_price
shop_data['item_price'] = np.log(shop_data['item_price'])
shop_data['item_price'] = preprocessing.scale(shop_data['item_price'])
shop_data = shop_data.dropna()

# Split into dependent/independent features
x = shop_data[shop_data.columns[~shop_data.columns.isin(['item_cnt_day', 'date', 'item_category_name_en', 'item_name_en'])]].values
y = shop_data['item_cnt_day'].values

# Fit Decision Tree and score
regressor = DecisionTreeRegressor(random_state=711).fit(x,y)
y_pred = regressor.predict(x)
mean_squared_error(y, y_pred, squared=False)
plt_kwargs = {
    'max_depth'    : 3,
    'feature_names': shop_data.columns[~shop_data.columns.isin(['item_cnt_day', 'date', 'item_category_name_en', 'item_name_en'])],  
    'class_names'  : 'item_cnt',
    'filled'       : True,
    'rounded'      : True
}

# Export tree graph to dot file and then run bash command to convert to png
export_graphviz(regressor, out_file='tree.dot', **plt_kwargs)
!dot -Tpng tree.dot -o tree.png
Image('./tree.png')
feature_importances = pd.Series(regressor.feature_importances_, index=shop_data.columns[~shop_data.columns.isin(['item_cnt_day', 'date', 'item_category_name_en', 'item_name_en'])])
px.bar(feature_importances.sort_values(ascending=False)[:15], title='Top 15 Feature Importance')
random_sample = np.arange(len(y))
np.random.shuffle(random_sample)
random_sample = random_sample[:3000]

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(random_sample)), y=y[random_sample], mode='markers', name='y'))
fig.add_trace(go.Scatter(x=np.arange(len(random_sample)), y=y_pred[random_sample], mode='lines', name='y_pred'))
fig.update_layout(title='Predictions vs Actuals on 3k samples')
# Shops in test set
rows = data.loc[data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].nunique()
fig = make_subplots(
    rows=rows,
    cols=1, 
    subplot_titles=list(data.loc[data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].unique())
)
# Individual Shop plots
for i, shop_name in enumerate(np.sort(data.loc[data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].unique())):
    values = data.loc[data['shop_name_en']==shop_name, 'date_block_num'].value_counts()
    
    fig.add_trace(
        go.Bar(x=list(values.index), y=values.values),
        row=i+1, col=1
    )

fig.update_layout(
    autosize=False,
    width=600,
    height=5000,
    showlegend=False,
    title=f'{rows} shops that are IN test set'
)
# Shops NOT in test set
rows = data.loc[~data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].nunique()
fig = make_subplots(
    rows=rows, 
    cols=1,
    subplot_titles=list(data.loc[~data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].unique())
)
# Individual Shop plots
for i, shop_name in enumerate(np.sort(data.loc[~data['shop_id'].isin(test['shop_id'].unique()), 'shop_name_en'].unique())):
    values = data.loc[data['shop_name_en']==shop_name, 'date_block_num'].value_counts()
    
    fig.add_trace(
        go.Bar(x=list(values.index), y=values.values),
        row=i+1, col=1
    )

fig.update_layout(
    autosize=False,
    width=600,
    height=2000,
    showlegend=False,
    title=f'{rows} shops that are NOT in test set'
)
len(test.loc[~test['shop_id'].isin(data['shop_id'].unique())])
# Potentially discontinued products
item_count_list = []
for n in range(1, 13):
    # Aggregate item sales over last n months
    conditions = (
          (data['date_block_num']> (33 - n)) # Aggregate sales from the last n months
        & (data['item_id'].isin(test['item_id'].unique())) # only include items that are in test set
    )
    item_agg_sales = data.loc[conditions].groupby('item_id').agg({
        'item_cnt_day': 'sum'
    })

    # Filter test items that are in training set but NOT sold in last n months
    conditions = (
          (~test['item_id'].isin(item_agg_sales.index)) # Item has not been sold in last n months
        & (test['item_id'].isin(data['item_id'].unique())) # Item HAS been sold in training set
        & (~test['item_id'].duplicated()) # just log one rccord per item
    )
    
    item_counts      = pd.DataFrame(test.loc[conditions, 'item_category_name_en'].value_counts()).reset_index()
    item_counts['x'] = n
    
    item_count_list.append(item_counts)

item_counts = pd.concat(item_count_list)    

# Graphing
fig = px.bar(
    x=item_counts['x'], 
    y=item_counts['item_category_name_en'], 
    color=item_counts['index'], 
#     text=item_counts.groupby('x').agg({'item_category_name_en':'sum'}), 
    title='Count of items in testing set that have not been sold in x months'
)
total_labels = [
    {"x": x, "y": total, "text": str(total), "showarrow": False} 
    for x, total in zip(range(1,13), np.ravel(item_counts.groupby('x').agg({'item_category_name_en':'sum'}).values))
]
fig.update_layout(annotations=total_labels)
# New Products
conditions = (
    (~test['item_id'].isin(data['item_id'].unique())) # test items that are not in the data set
    & (~test['item_id'].duplicated()) # just one record per item id
)

item_categories = test.loc[conditions, 'item_category_name_en'].value_counts()

px.bar(item_categories, text=item_categories.values, title=f'{item_categories.sum()} items not found in training set')
# Downtrending products
# Aggregate item sales over last n months
conditions = (
      (data['date_block_num']> (33 - 6)) # Aggregate sales from the last 6 months
    & (data['item_id'].isin(test['item_id'].unique())) # only include items that are in test set
)

# Calculate a 6 month slope
trailing_6_item_sales = data.loc[conditions, ['item_id', 'date_block_num', 'item_cnt_day']] \
                            .pivot_table(index='item_id', columns='date_block_num', aggfunc='sum')

def get_slope(row):
    slope, _, _, _, _ = stats.linregress(row.index.get_level_values(1).tolist(), row.values)
    return slope

trailing_6_item_sales['trend'] = trailing_6_item_sales.fillna(0).apply(get_slope, axis=1)
potentially_failing_items = trailing_6_item_sales.loc[trailing_6_item_sales[('item_cnt_day', 33)] < -trailing_6_item_sales['trend']]

print(f"{len(potentially_failing_items)} items in test set that could be a zero by month 34.")
potentially_failing_items
test['item_cnt_month'] = None
n_months = test['item_cnt_month'] = None
n_months = 3
# Lookback period to determine if a shop sold an item category 
shop_categories = data.loc[
                            (
                                (data['date_block_num']> 33 - n_months) & # Limit to sales N months ago
                                (data['shop_id'].isin(test['shop_id'].unique())) # Limit to shops in test set
                            )
                        ] \
                      .groupby('shop_id')['item_category_id'] \
                      .unique()

def categories_to_exclude(test_shop_id, test_shop_categories):
    return set(test_shop_categories) - set(shop_categories.loc[test_shop_id])

# categories_to_exclude('Adygea shopping center Mega', shop_categories.iloc[2])

# Loop through test shops and set itm_cnt_month to zero if shop hasn't sold that category in N months
for shop_id in test['shop_id'].unique():
    
    excluded_categories = categories_to_exclude(shop_id, test['item_category_id'].unique())
    
    test.loc[(
          (test['shop_id'] == shop_id)
        & (test['item_category_id'].isin(excluded_categories))
    ), 'item_cnt_month'] = 0

n_zeros = len(test[test['item_cnt_month']==0])
f'{n_zeros*100/len(test):.2f}% of records are zero'
#turns out, the only category that gets completely removed is "PC - Headsets / Headphones". Only 3 were sold back in early 2013

categories_in_last_n_months = data.loc[(
                                    (data['date_block_num']> 33 - n_months) # Limit to sales N months ago
                                ), 'item_category_id'].unique()

categories_to_exclude = test.loc[~test['item_category_id'].isin(categories_in_last_n_months), 'item_category_id'].unique()

test.loc[test['item_category_id'].isin(categories_to_exclude), 'item_cnt_month'] = 0

n_zeros = len(test[test['item_cnt_month']==0])
f'{n_zeros*100/len(test):.2f}% of records are zero'
items_unsold_in_last_n_months = data.loc[(
                                    (data['date_block_num']> 33 - n_months) # Limit to sales N months ago
                                ), 'item_id'].unique()

test.loc[(
    (~test['item_id'].isin(items_unsold_in_last_n_months)) # Item is NOT in last 3 months
    & (test['item_id'].isin(data['item_id'].unique())) # BUT Item IS in data set, ruling out entirely new products
), 'item_cnt_month'] = 0

n_zeros = len(test[test['item_cnt_month']==0])
f'{n_zeros*100/len(test):.2f}% of records are zero'
# items that are unique to the test set
new_items = test.loc[~test['item_id'].isin(data['item_id'].unique()), 'item_id'].unique()

shop_items = data.loc[
                        (
                            (data['date_block_num']> 33 - n_months) & # Limit to sales N months ago
                            (data['shop_id'].isin(test['shop_id'].unique())) # Limit to shops in test set
                        )
                    ] \
                  .groupby('shop_id')['item_id'] \
                  .unique()

def items_to_exclude(test_shop_id, test_shop_items):
    return set(test_shop_items) - set(shop_items.loc[test_shop_id])

# Loop through test shops and set itm_cnt_month to zero if shop hasn't sold that category in N months AND item is NOT new
for shop_id in test['shop_id'].unique():
    
    excluded_items = items_to_exclude(shop_id, test.loc[(~test['item_id'].isin(new_items)), 'item_id'].unique())
    
    test.loc[(
          (test['shop_id'] == shop_id)
        & (test['item_id'].isin(excluded_items))
    ), 'item_cnt_month'] = 0

n_zeros = len(test[test['item_cnt_month']==0])
f'{n_zeros*100/len(test):.2f}% of records are zero'
n_zeros = len(test[test['item_cnt_month']==0])
f'{n_zeros*100/len(test):.2f}% of records are zero'