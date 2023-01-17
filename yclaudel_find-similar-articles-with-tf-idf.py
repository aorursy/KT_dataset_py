import numpy as np

import pandas as pd

import pickle



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from sklearn.cluster import MiniBatchKMeans
# Read the articles

X = pd.read_csv("../input/all-the-news/articles1.csv")

X = X[pd.isna(X['title'])==False]

X = X[pd.isna(X['content'])==False]
# Preview the data

X.head(3)
# data information

X.info()
# tfidf calculation

text_content = X['content']

vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents

                             #min_df=8,      # only use words that appear at least X times

                             stop_words='english', # remove stop words

                             lowercase=True, # Convert everything to lower case 

                             use_idf=True,   # Use idf

                             norm=u'l2',     # Normalization

                             smooth_idf=True # Prevents divide-by-zero errors

                            )

tfidf = vector.fit_transform(text_content)
# Save objects on filesystem

#pickle.dump(X, open('output/X', 'wb')) 

#pickle.dump(vector, open('output/vector', 'wb')) 

#pickle.dump(tfidf, open('output/tfidf', 'wb')) 
# load objects on filesystem

#X = pickle.load(open('X', 'rb'))

#vector = pickle.load(open('vector', 'rb'))

#tfidf = pickle.load(open('tfidf', 'rb'))
# Request function : search the top_n articles from a request ( request = string)

def search(tfidf_matrix,model,request, top_n = 5):

    request_transform = model.transform([request])

    similarity = np.dot(request_transform,np.transpose(tfidf_matrix))

    x = np.array(similarity.toarray()[0])

    indices=np.argsort(x)[-5:][::-1]

    return indices



# Find similar : get the top_n articles similar to an article 

def find_similar(tfidf_matrix, index, top_n = 5):

    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()

    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

    return [index for index in related_docs_indices][0:top_n]    



# Print the result

def print_result(request_content,indices,X):

    print('\nsearch : ' + request_content)

    print('\nBest Results :')

    for i in indices:

        print('id = {0:5d} - title = {1}'.format(i,X['title'].loc[i]))
request = 'peillon macron fillon marche'

#request = text_content[0]



result = search(tfidf,vector, request, top_n = 5)

print_result(request,result,X)
index = 13084

result = find_similar(tfidf, index, top_n = 5)

print_result('13084 - title = Accusations of Anti-Semitism Taint French Presidential Race',result,X)
# Clustering  Kmeans

k = 20

kmeans = MiniBatchKMeans(n_clusters = k)

kmeans.fit(tfidf)

centers = kmeans.cluster_centers_.argsort()[:,::-1]

terms = vector.get_feature_names()

# centers of the clusters

for i in range(0,k):

    word_list=[]

    print("cluster%d:"% i)

    for j in centers[i,:10]:

        word_list.append(terms[j])

    print(word_list) 
# predict

request_transform = vector.transform(["Accurate predictions of Earth’s warming require computers that are too expensive for one country or institution"])

kmeans.predict(request_transform)