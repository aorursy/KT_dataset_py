# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 



# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import the data (chunksize returns jsonReader for iteration)

businesses = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_business.json", lines=True, orient='columns', chunksize=1000000)

reviews = pd.read_json("/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json", lines=True, orient='columns', chunksize=1000000)
# read the data 

for business in businesses:

    subset_business = business

    break

    

for review in reviews:

    subset_review = review

    break
# peak the tables

display(subset_business.head(2))

display(subset_review.head(2))
# Businesses in Toronto and currently open business

city = subset_business[(subset_business['city'] == 'Toronto') & (subset_business['is_open'] == 1)]

toronto = city[['business_id','name','address', 'categories', 'attributes','stars']]

toronto
# getting just restaurants from Toronto business

rest = toronto[toronto['categories'].str.contains('Restaurant.*')==True].reset_index()

rest   
# Function that extract keys from the nested dictionary

def extract_keys(attr, key):

    if attr == None:

        return "{}"

    if key in attr:

        return attr.pop(key)



# convert string to dictionary

import ast

def str_to_dict(attr):

    if attr != None:

        return ast.literal_eval(attr)

    else:

        return ast.literal_eval("{}")    
# get dummies from nested attributes

rest['BusinessParking'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'BusinessParking')), axis=1)

rest['Ambience'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Ambience')), axis=1)

rest['GoodForMeal'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'GoodForMeal')), axis=1)

rest['Dietary'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Dietary')), axis=1)

rest['Music'] = rest.apply(lambda x: str_to_dict(extract_keys(x['attributes'], 'Music')), axis=1)
rest
# create table with attribute dummies

df_attr = pd.concat([ rest['attributes'].apply(pd.Series), rest['BusinessParking'].apply(pd.Series),

                    rest['Ambience'].apply(pd.Series), rest['GoodForMeal'].apply(pd.Series), 

                    rest['Dietary'].apply(pd.Series) ], axis=1)

df_attr_dummies = pd.get_dummies(df_attr)

df_attr_dummies
# get dummies from categories

df_categories_dummies = pd.Series(rest['categories']).str.get_dummies(',')

df_categories_dummies
# pull out names and stars from rest table 

result = rest[['name','stars']]

result
# Concat all tables and drop Restaurant column

df_final = pd.concat([df_attr_dummies, df_categories_dummies, result], axis=1)

df_final.drop('Restaurants',inplace=True,axis=1)
# map floating point stars to an integer

mapper = {1.0:1,1.5:2, 2.0:2, 2.5:3, 3.0:3, 3.5:4, 4.0:4, 4.5:5, 5.0:5}

df_final['stars'] = df_final['stars'].map(mapper)
# Final table for the models 

df_final
# Create X (all the features) and y (target)

X = df_final.iloc[:,:-2]

y = df_final['stars']
# Split the data into train and test sets

from sklearn.model_selection import train_test_split

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier(n_neighbors=20)

knn.fit(X_train_knn, y_train_knn)



#y_pred = knn.predict(X_test)



accuracy_train = knn.score(X_train_knn, y_train_knn)

accuracy_test = knn.score(X_test_knn, y_test_knn)



print(f"Score on training set: {accuracy_train}")

print(f"Score on test set: {accuracy_test}")
# look at the last row for the test

display(df_final.iloc[-1:])



# look at the restaurant name from the last row.

print("Validation set (Restaurant name): ", df_final['name'].values[-1])
# test set from the df_final table (only last row): Restaurant name: "Steak & Cheese & Quick Pita Restaurant"

test_set = df_final.iloc[-1:,:-2]



# validation set from the df_final table (exclude the last row)

X_val =  df_final.iloc[:-1,:-2]

y_val = df_final['stars'].iloc[:-1]
# fit model with validation set

n_knn = knn.fit(X_val, y_val)
# distances and indeces from validation set (Steak & Cheese & Quick Pita Restaurant)

distances, indeces =  n_knn.kneighbors(test_set)

#n_knn.kneighbors(test_set)[1][0]



# create table distances and indeces from "Steak & Cheese & Quick Pita Restaurant"

final_table = pd.DataFrame(n_knn.kneighbors(test_set)[0][0], columns = ['distance'])

final_table['index'] = n_knn.kneighbors(test_set)[1][0]

final_table.set_index('index')
# get names of the restaurant that similar to the "Steak & Cheese & Quick Pita Restaurant"

result = final_table.join(df_final,on='index')

result[['distance','index','name','stars']].head(5)
# looking at the columns of subset_review table

subset_review.columns
# pull out needed columns from subset_review table

df_review = subset_review[['user_id','business_id','stars', 'date']]

df_review
# pull out names and addresses of the restaurants from rest table

restaurant = rest[['business_id', 'name', 'address']]

restaurant
# combine df_review and restaurant table

combined_business_data = pd.merge(df_review, restaurant, on='business_id')

combined_business_data
# the most POPULAR restaurants by stars.

combined_business_data.groupby('business_id')['stars'].count().sort_values(ascending=False).head()
# see the NAME of the most popular restaurant

Filter = combined_business_data['business_id'] == 'h_4dPV9M9aYaBliH1Eoeeg'

print("Name: ", combined_business_data[Filter]['name'].unique())

print("Address:", combined_business_data[Filter]['address'].unique())
# create a user-item matrix

rating_crosstab = combined_business_data.pivot_table(values='stars', index='user_id', columns='name', fill_value=0)

rating_crosstab.head()
# shape of the Utility matrix (original matrix) 

rating_crosstab.shape
# Transpose the Utility matrix

X = rating_crosstab.values.T

X.shape
import sklearn

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import accuracy_score





SVD = TruncatedSVD(n_components=12, random_state=17)

result_matrix = SVD.fit_transform(X)

result_matrix.shape
# PearsonR coef 

corr_matrix = np.corrcoef(result_matrix)

corr_matrix.shape
# get the index of the popular restaurant

restaurant_names = rating_crosstab.columns

restaurants_list = list(restaurant_names)



popular_rest = restaurants_list.index('Wvrst')

print("index of the popular restaurant: ", popular_rest) 
# restaurant of interest 

corr_popular_rest = corr_matrix[popular_rest]

corr_popular_rest.shape  
list(restaurant_names[(corr_popular_rest < 1.0) & (corr_popular_rest > 0.9)])
display(rest[rest['name'] == 'Wvrst'])
# create the copy of combined_business_data table

combined_business_data_keras = combined_business_data.copy()

combined_business_data_keras.head(1)
from sklearn.preprocessing import LabelEncoder



user_encode = LabelEncoder()



combined_business_data_keras['user'] = user_encode.fit_transform(combined_business_data_keras['user_id'].values)

n_users = combined_business_data_keras['user'].nunique()



item_encode = LabelEncoder()



combined_business_data_keras['business'] = item_encode.fit_transform(combined_business_data_keras['business_id'].values)

n_rests = combined_business_data_keras['business'].nunique()



combined_business_data_keras['stars'] = combined_business_data_keras['stars'].values#.astype(np.float32)



min_rating = min(combined_business_data_keras['stars'])

max_rating = max(combined_business_data_keras['stars'])



print(n_users, n_rests, min_rating, max_rating)



combined_business_data_keras
from sklearn.model_selection import train_test_split



X = combined_business_data_keras[['user', 'business']].values

y = combined_business_data_keras['stars'].values



X_train_keras, X_test_keras, y_train_keras, y_test_keras = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_keras.shape, X_test_keras.shape, y_train_keras.shape, y_test_keras.shape
X_train_keras[:, 0]
n_factors = 50



X_train_array = [X_train_keras[:, 0], X_train_keras[:, 1]]

X_test_array = [X_test_keras[:, 0], X_test_keras[:, 1]]
X_train_array, X_test_array
from keras.layers import Add, Activation, Lambda

from keras.models import Model

from keras.layers import Input, Reshape, Dot

from keras.layers.embeddings import Embedding

from keras.optimizers import Adam

from keras.regularizers import l2



class EmbeddingLayer:

    def __init__(self, n_items, n_factors):

        self.n_items = n_items

        self.n_factors = n_factors

    

    def __call__(self, x):

        x = Embedding(self.n_items, self.n_factors, embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))(x)

        x = Reshape((self.n_factors,))(x)

        

        return x

    

def Recommender(n_users, n_rests, n_factors, min_rating, max_rating):

    user = Input(shape=(1,))

    u = EmbeddingLayer(n_users, n_factors)(user)

    ub = EmbeddingLayer(n_users, 1)(user)

    

    restaurant = Input(shape=(1,))

    m = EmbeddingLayer(n_rests, n_factors)(restaurant)

    mb = EmbeddingLayer(n_rests, 1)(restaurant)   

    

    x = Dot(axes=1)([u, m])

    x = Add()([x, ub, mb])

    x = Activation('sigmoid')(x)

    x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)  

    

    model = Model(inputs=[user, restaurant], outputs=x)

    opt = Adam(lr=0.001)

    model.compile(loss='mean_squared_error', optimizer=opt)  

    

    return model
keras_model = Recommender(n_users, n_rests, n_factors, min_rating, max_rating)

keras_model.summary()
keras_model.fit(x=X_train_array, y=y_train_keras, batch_size=64,\

                          epochs=5, verbose=1, validation_data=(X_test_array, y_test_keras))
# prediction

predictions = keras_model.predict(X_test_array)
# create the df_test table with prediction results

df_test = pd.DataFrame(X_test_keras[:,0])

df_test.rename(columns={0: "user"}, inplace=True)

df_test['business'] = X_test_keras[:,1]

df_test['stars'] = y_test_keras

df_test["predictions"] = predictions

df_test.head()
# Plotting the distribution of actual and predicted stars

import matplotlib.pyplot as plt

import seaborn as sns

values, counts = np.unique(df_test['stars'], return_counts=True)



plt.figure(figsize=(8,6))

plt.bar(values, counts, tick_label=['1','2','3','4','5'], label='true value')

plt.hist(predictions, color='orange', label='predicted value')

plt.xlabel("Ratings")

plt.ylabel("Frequency")

plt.title("Ratings Histogram")

plt.legend()

plt.show()
# # plot 

# import matplotlib.pyplot as plt

# import seaborn as sns



# plt.figure(figsize=(15,6))



# ax1 = sns.distplot(df_test['stars'], hist=False, color="r", label="Actual Value")

# sns.distplot(predictions, hist=False, color="g", label="model2 Fitted Values" , ax=ax1)



# plt.title('Actual vs Fitted Values for Restaurant Ratings')

# plt.xlabel('Stars')

# plt.ylabel('Proportion of Ratings')



# plt.show()

# plt.close()
# Extract embeddings

emb = keras_model.get_layer('embedding_3')

emb_weights = emb.get_weights()[0]



print("The shape of embedded weights: ", emb_weights.shape)

print("The length of embedded weights: ", len(emb_weights))
# normalize and reshape embedded weights

emb_weights = emb_weights / np.linalg.norm(emb_weights, axis = 1).reshape((-1, 1))

len(emb_weights)
# get all unique business_ids (restaurants)

rest_id_emb = combined_business_data_keras["business_id"].unique()

len(rest_id_emb)
rest_pd = pd.DataFrame(emb_weights)

rest_pd["business_id"] = rest_id_emb

rest_pd = rest_pd.set_index("business_id")

rest_pd
# merging rest_pd and temp tables to get the name of the restaurants.

temp = combined_business_data_keras[['business_id', 'name']].drop_duplicates()

df_recommend = pd.merge(rest_pd, temp, on='business_id')

df_recommend
# exrtract the target restaurant from the df_recommend table

target = df_recommend[df_recommend['name'] == 'Wvrst']

target.iloc[:,1:51]
def find_similarity_total(rest_name):

    """Recommends restaurant based on the cosine similarity between restaurants"""

    cosine_list_total = []

    result = []



    for i in range(0, df_recommend.shape[0]):

        sample_name = df_recommend[df_recommend["name"] == rest_name].iloc[:,1:51]

        row = df_recommend.iloc[i,1:51]

        cosine_total = np.dot(sample_name, row)

        

        recommended_name = df_recommend.iloc[i,51]

        cosine_list_total.append(cosine_total)

        result.append(recommended_name)

        

    cosine_df_total = pd.DataFrame({"similar_rest" : result, "cosine" : cosine_list_total})



    return cosine_df_total
# call the function with input of "Wvrst" and store it in result variable.

result = find_similarity_total('Wvrst')
# head of result table

result.head()
'''

- function that replace '[]' to empty str 

- convert string to float

'''

def convert(input):

    return float(str(input).replace('[','').replace(']',''))
# create new column called "cos" in result table

result['cos'] = result.apply(lambda x: convert(x['cosine']), axis=1)



# drop original 'cosine' column (which had values with np.array)

result.drop('cosine', axis=1, inplace=True)



# sort values with cos

result.sort_values('cos', ascending=False).head(10)