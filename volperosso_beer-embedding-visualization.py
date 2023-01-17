import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from umap import UMAP
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from pprint import pprint
style = pd.read_csv('/kaggle/input/beer-recipes/styleData.csv', delimiter=',', encoding='latin')
recipe = pd.read_csv('/kaggle/input/beer-recipes/recipeData.csv', delimiter=',', encoding='latin')
# these columns will be used later after we embed the data
relevant_columns = ['Name', 'Style', 'StyleID', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilTime', 'Efficiency', 'SugarScale']

# these columns will be used for the embedding
data_columns = ['OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilTime', 'Efficiency']

# take the relevant columns
data = recipe[relevant_columns].copy()
data = data[data['SugarScale'].isin(['Specific Gravity'])].copy()
data = data.dropna(how='any').copy()
# drop rows with missing data
X = data[data_columns].copy()

# scale the data by mean and standard deviation
scaler = StandardScaler()
X_normed = scaler.fit_transform(X)

# subsample the data for embedding
np.random.seed(0)
num_to_subsample = 4000
random_idxs = np.random.permutation(X.shape[0])[:num_to_subsample]
X_normed_subsampled = X_normed[random_idxs]
data_subsampled = data.iloc[random_idxs, :].copy()
np.random.seed(1)
X_trans = UMAP(metric='l1').fit_transform(X_normed_subsampled)
fig, ax = plt.subplots()

# color = data_subsampled['StyleID']
color = data_subsampled['Style'].str.contains('IPA|Indian Pale Ale', case=True)

ax.scatter(X_trans[:,0], X_trans[:,1], s=5, c=color)
ax.set_title('Embedding of Beers')
# ax.set_xticks([])
# ax.set_yticks([])


ax.set_aspect('equal')
fig.set_size_inches(10, 10)
top_cluster = data_subsampled['Style'][X_trans[:,1]>5].copy()
styles_in_top_cluster = top_cluster.value_counts()
# styles_in_top_cluster.iloc[:15]
ll_cluster = data_subsampled['Style'][np.logical_and(X_trans[:,0]<-2, X_trans[:,1]<0)]
styles_in_ll_cluster = ll_cluster.value_counts()
# styles_in_ll_cluster.iloc[:15]
r_cluster = data_subsampled['Style'][np.logical_and(X_trans[:,0]>-3, X_trans[:,1]<5)].copy()
styles_in_r_cluster = r_cluster.value_counts()
# styles_in_r_cluster.iloc[:15]
r_diff = styles_in_r_cluster.add(-styles_in_ll_cluster.add(styles_in_top_cluster, fill_value=0), fill_value=0)
r_diff[r_diff>0].sort_values(ascending=False).iloc[:15]
ll_diff = styles_in_ll_cluster.add(-styles_in_r_cluster.add(styles_in_top_cluster, fill_value=0), fill_value=0)
ll_diff[ll_diff>0].sort_values(ascending=False).iloc[:15]
top_diff = styles_in_top_cluster.add(-styles_in_r_cluster.add(styles_in_ll_cluster, fill_value=0), fill_value=0)
top_diff[top_diff>0].sort_values(ascending=False).iloc[:15]
pprint(set(list(ll_cluster.values)).difference(set(list(r_cluster.values)+list(top_cluster.values))))
pprint(set(list(top_cluster.values)).difference(set(list(ll_cluster.values)+list(r_cluster.values))))
data['Style'].value_counts().iloc[:25].plot.bar()
