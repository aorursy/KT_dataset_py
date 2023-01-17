import random
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.model_selection import train_test_split
!pip install similaritymeasures &>/dev/null
from similaritymeasures import dtw

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv(
    '/kaggle/input/pga-tour-20102018-data/PGA_Data_Historical.csv', 
)
df.date = pd.to_datetime(df.date)
df = df[df.date > '2015-01-01']

StatVar = df['statistic'] + ' - (' + df['variable'] + ')'
del df['variable']
df['variable'] = StatVar

wide_df = df.set_index(['player_name', 'variable', 'date', 'tournament'])['value'].unstack('variable').reset_index()
for col in  wide_df.columns[2:]:
    if 'Money' in col:
        wide_df[col] = wide_df[col].str.replace('$','')
        wide_df[col] = wide_df[col].str.replace(',','')
    try:
        wide_df[col] = wide_df[col].astype(float)
    except ValueError:
        pass
plt.rcParams["figure.figsize"] = (8,8)
sns.heatmap(wide_df.corr(), square=True)
wide_df.describe()
player_to_acc_dynamics = {
    name: group['Accuracy Rating - (RATING)'].dropna().astype(float).array
    for name, group in wide_df.groupby('player_name')
    if group.count()['date'] > 15
}
from scipy import ndimage


def normalize_series(dynamics, target_length=100):
    "Turn `dynamics` of length n to length `num_chunks` using linear interpolation"
    length = len(dynamics)
    return ndimage.gaussian_filter1d(np.interp(np.linspace(0, length-1, target_length), range(length), dynamics), sigma=2.5)

def get_matrix(player_to_acc_dynamics, target_length=50):
    matrix = np.ndarray((len(player_to_acc_dynamics), target_length))
    for i, (_, acc_dynamics) in enumerate(player_to_acc_dynamics.items()):
        acc_dynamics = normalize_series(acc_dynamics, target_length=target_length)
        matrix[i, :] = acc_dynamics
    return matrix

matrix = get_matrix(player_to_acc_dynamics)
for row in matrix[:10]:
    plt.plot(row)
plt.legend(player_to_acc_dynamics.keys(), bbox_to_anchor=(0.4, 1, 1., .102))
plt.title('Arcs on a common (quantile-based) x axis')
num_players = matrix.shape[0]
distances = np.ndarray((num_players, num_players))
for i in range(num_players):
    for j in range(num_players):
        distances[i, j], _ = dtw(np.expand_dims(matrix[i], axis=0), 
                                 np.expand_dims(matrix[j], axis=0))
player_names = list(player_to_acc_dynamics.keys())
sns.heatmap(distances[:25, :25], square=True, xticklabels=player_names[:25], yticklabels=player_names[:25], cbar=False)
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
model = model.fit(distances)
distance = np.arange(model.children_.shape[0])
no_of_observations = np.arange(2, model.children_.shape[0]+2)
linkage_matrix = np.column_stack([model.children_, model.distances_, no_of_observations]).astype(float)

g = sns.clustermap(col_linkage=linkage_matrix, row_cluster=True, col_cluster=False, data=matrix, cbar=False, 
                   yticklabels=[], 
                   figsize=(15, 10), cmap="YlGnBu")
small_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')
small_model = small_model.fit(distances[:25, :25])
distance = np.arange(model.children_.shape[0])
no_of_observations = np.arange(2, small_model.children_.shape[0]+2)
linkage_matrix_small = np.column_stack([small_model.children_, small_model.distances_, no_of_observations]).astype(float)

g = sns.clustermap(col_linkage=linkage_matrix_small, row_cluster=True, col_cluster=False, data=matrix[:25], cbar=False, 
                   yticklabels=player_names[:25], 
                   figsize=(10, 5), cmap="YlGnBu")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (20,8)
dendrogram(linkage_matrix, color_threshold=4, orientation='left')
clusters = fcluster(linkage_matrix, 3.3 ,'distance')
print(f'Num clusters: {len(set(clusters))}')
for cluster, player in zip(clusters[:25], player_names[:25]):
    # printing just 25 first players
    print(f'{player} belongs to cluster {cluster}')
dataset = wide_df.groupby('player_name').agg('mean')
dataset = dataset[dataset['Accuracy Rating - (RATING)'].notna()]
X, y = dataset.drop(columns=['Accuracy Rating - (RATING)']), dataset['Accuracy Rating - (RATING)']
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2)
print(X_dev.shape, X_test.shape)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

baseline = DummyRegressor(strategy='mean')

preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')), # Filling in nans with mean value of each column
    ('scaler', MinMaxScaler()),
])

def make_ridge(alpha):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('regression', Ridge(alpha=alpha))
    ])

def make_lasso(alpha):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('regression', Lasso(alpha=alpha))
    ])

def make_forrest(num_estimators):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('regression', RandomForestRegressor(n_estimators=num_estimators))
    ])

def make_gradient_boosting(n_estimators, max_depth):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('regression', GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth))
    ])

def make_adaboost(n_estimators):
    return Pipeline([
        ('preprocessing', preprocessing),
        ('regression', AdaBoostRegressor(n_estimators=n_estimators))
    ])

results = pd.DataFrame(columns=['model name', '$R^2$', 'seed', 'mode'])
MODELS = {
    'always predict mean': baseline,
    'ridge regression $\\alpha = 0.0001$': make_ridge(0.0001),
    'ridge regression $\\alpha = 0.00001$': make_ridge(0.00001),
    'ridge regression $\\alpha = 0.0000001$': make_ridge(0.000005),
    'lasso regression $\\alpha = 0.01$': make_lasso(0.01),
    'lasso regression $\\alpha = 0.001$': make_lasso(0.001),
    'lasso regression $\\alpha = 0.0005$': make_lasso(0.0005),
    'random forrest of 15 trees': make_forrest(15),
    'random forrest of 25 trees': make_forrest(25),
    'random forrest of 50 trees': make_forrest(50),
    'gradient_boosting 50 trees depth 5': make_gradient_boosting(n_estimators=50, max_depth=5),
    'gradient_boosting 200 trees depth 5': make_gradient_boosting(n_estimators=200, max_depth=5),
    'gradient_boosting 50 trees depth 2': make_gradient_boosting(n_estimators=50, max_depth=5),
    'gradient_boosting 200 trees depth 2': make_gradient_boosting(n_estimators=200, max_depth=2),
    'gradient_boosting 200 trees depth 3': make_gradient_boosting(n_estimators=250, max_depth=3),
    'adaboost 100 trees': make_adaboost(100),
    'adaboost 200 trees': make_adaboost(200),
}

for name, model in MODELS.items():
    result_dict = cross_validate(model, X_dev, y_dev, return_train_score=True)
    test_scores = result_dict['train_score'].mean()
    for seed, score in enumerate(result_dict['train_score']):
        results.loc[len(results)] = (name, score, seed, 'train')
        
    test_scores = result_dict['test_score'].mean()
    for seed, score in enumerate(result_dict['test_score']):
        results.loc[len(results)] = (name, score, seed, 'test') 
    # print(f'{name}, R^2 = {test_score:.2f}')
results
plt.rcParams["figure.figsize"] = (30, 6)
plt.xticks(rotation=90)
plt.ylim(0.6, 1)
_ = sns.barplot(data=results, x='model name', y='$R^2$', hue='mode')
best_model = make_lasso(0.01)
best_model.fit(X_dev, y_dev)
best_model.score(X_test, y_test)