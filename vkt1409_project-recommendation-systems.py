import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# %matplotlib inline

plt.style.use("ggplot")



import sklearn

from sklearn.decomposition import TruncatedSVD





import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
amazon_ratings = pd.read_csv('../input/ratings_Electronics.csv')

amazon_ratings.columns = ['UserId','ProductId','Rating','Timestamp']

amazon_ratings = amazon_ratings.dropna()

amazon_ratings.head()
print("Shape: %s" % str(amazon_ratings.shape))

print("Column names: %s" % str(amazon_ratings.columns))
amazon_ratings.shape
# Unique Users and Products



print("Unique UserID count: %s" % str(amazon_ratings.UserId.nunique()))

print("Unique ProductID count: %s" % str(amazon_ratings.ProductId.nunique()))
# Rating frequency



sns.set(rc={'figure.figsize': (11.7, 8.27)})

sns.set_style('whitegrid')

ax = sns.countplot(x='Rating', data=amazon_ratings, palette=sns.color_palette('Greys'))

ax.set(xlabel='Rating', ylabel='Count')

plt.show()
# Mean rating for each Product



product_rating = amazon_ratings.groupby('ProductId')['Rating'].mean()

product_rating.head()
# Mean rating KDE distribution



ax = sns.kdeplot(product_rating, shade=True, color='grey')

plt.show()
# Count of the number of ratings per Product



product_rating_count = amazon_ratings.groupby('ProductId')['Rating'].count()

product_rating_count.head()
# Number of ratings per product KDE distribution



ax = sns.kdeplot(product_rating_count, shade=True, color='grey')

plt.show()
# Un-Reliability factor



unreliability = amazon_ratings.groupby('ProductId')['Rating'].std(ddof = -1)

unreliability.head()
# Un-Reliability factor KDE distribution



ax = sns.kdeplot(unreliability, shade=True, color='grey')

plt.show()


unique_products_list = amazon_ratings.ProductId.unique()

data_model = pd.DataFrame({'Rating': product_rating[unique_products_list],\

                           'Count': product_rating_count[unique_products_list], \

                          'Unreliability': unreliability[unique_products_list]})

data_model.head()
print("Data model shape (number of data points): %s" % str(data_model.shape))
# Rating versus count



sns.set_style('ticks')

plt.figure(num=None, figsize=(11.7, 8.27), dpi=100, facecolor='w', edgecolor='k')



ax = data_model.plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.1)

plt.show()
# Less than 100 ratings



ax = data_model[data_model.Count < 101].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.1)

plt.show()
# 100 to 200 ratings



ax = data_model[data_model.Count > 100]\

[data_model.Count<201].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.4)

plt.show()
# 200 to 500 ratings



ax = data_model[data_model.Count > 200]\

[data_model.Count<501].plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.4)

plt.show()
# Adding unreliability factor to the above plots 100 to 200 ratings



ax = data_model[data_model.Count > 100]\

[data_model.Count<201].plot(kind='scatter', x='Unreliability', y='Count', c='Rating', cmap='jet', alpha=0.6)

plt.show()
# Addding unreliability factor to the above plots 200 to 500 ratings



ax = data_model[data_model.Count > 200]\

[data_model.Count<501].plot(kind='scatter', x='Unreliability', y='Count', c='Rating', cmap='jet', alpha=0.6)

plt.show()
# Coefficient of corelation between Unreliability and Rating



coeff_corelation = np.corrcoef(x=data_model.Unreliability, y=data_model.Rating)

print("Coefficient of corelation: ")

print(coeff_corelation)
# Summarise Count



print(data_model.Count.describe())
# Summarise Rating



print(data_model.Rating.describe())
# Summarise Unreliability



print(data_model.Unreliability.describe())
# Removing outliers and improbable data points



data_model = data_model[data_model.Count > 50][data_model.Count < 1001].copy()

print(data_model.shape)
# Normalization function to range 0 - 10



def normalize(values):

    mn = values.min()

    mx = values.max()

    return(10.0/(mx - mn) * (values - mx)+10)

    
data_model_norm = normalize(data_model)

data_model_norm.head()
# Setting up the model



# Recommend 20 similar items

engine = KNeighborsClassifier(n_neighbors=20)



# Training data points

data_points = data_model_norm[['Count', 'Rating', 'Unreliability']].values



#Training labels

labels = data_model_norm.index.values



print("Data points: ")

print(data_points)

print("Labels: ")

print(labels)



engine.fit(data_points, labels)
# Enter product ID to get a list of 20 recommended items



# User entered value

product_id = 'B00L3YHF6O'



product_data = [data_model_norm.loc[product_id][['Count', 'Rating', 'Unreliability']].values]



recommended_products = engine.kneighbors(X=product_data, n_neighbors=20, return_distance=False)



# List of product IDs form the indexes



products_list = []



for each in recommended_products:

    products_list.append(data_model_norm.iloc[each].index)



print("Recommended products: ")

print(products_list)



# Showing recommended products



ax = data_model_norm.plot(kind='scatter', x='Rating', y='Count', color='grey', alpha=0.20)

data_model_norm.iloc[recommended_products[0]].plot(kind='scatter', x='Rating', y='Count',\

                                                   color='orange', alpha=0.5, ax=ax)



ax2 = data_model_norm.plot(kind='scatter', x='Rating', y='Unreliability', color='grey')

data_model_norm.iloc[recommended_products[0]].plot(kind='scatter', x='Rating', y='Unreliability',\

                                                   color='orange', alpha=0.5, ax=ax2)

plt.show()
popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())

most_popular = popular_products.sort_values('Rating', ascending=False)

most_popular.head(10)
most_popular.head(30).plot(kind = "bar")
# Subset of Amazon Ratings



amazon_ratings1 = amazon_ratings.head(10000)
ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)

ratings_utility_matrix.head()
ratings_utility_matrix.shape
X = ratings_utility_matrix.T

X.head()
X.shape
X1 = X
SVD = TruncatedSVD(n_components=10)

decomposed_matrix = SVD.fit_transform(X)

decomposed_matrix.shape
correlation_matrix = np.corrcoef(decomposed_matrix)

correlation_matrix.shape
X.index[99]
i = "1616833742"



product_names = list(X.index)

product_ID = product_names.index(i)

product_ID
correlation_product_ID = correlation_matrix[product_ID]

correlation_product_ID.shape
Recommend = list(X.index[correlation_product_ID > 0.90])



# Removes the item already bought by the customer

Recommend.remove(i) 



Recommend[0:9]