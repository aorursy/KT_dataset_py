import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import MinMaxScaler 

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score 

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.cluster import AgglomerativeClustering as AC

import itertools

from sklearn.cluster import KMeans

from collections import Counter
read_df = pd.read_csv("../input/dmassign1/data.csv", na_values = "?")

df_id = read_df['ID']

Y = read_df['Class']

X = read_df.drop(['Class', 'ID'],  axis = 1)

X = X.fillna(method = 'ffill')

X = pd.get_dummies(X)
# x = read_df.corr()['Class'].sort_values()

# read_df = read_df.drop(x[np.abs(x) < 0.025].index.values,  axis = 1)

X = pd.DataFrame(StandardScaler().fit_transform(X))
pca = PCA(n_components = 60)

pca.fit(X)

T1 = pca.transform(X)

pca.explained_variance_ratio_.sum()
mx, mr, mn = 0, 0, 0



for r in range(30,40):

    for num in range(5, 15):

        kmeans = KMeans(n_clusters = num, random_state=r)

        pred = kmeans.fit_predict(T1)



        a = {}

        for item in range(num):

            a[item] = []

        

        for index, p in enumerate(pred[:1300]):

            a[p].append(index)



        subs = {}

        for item in range(num):

            subs[item] = int(Counter(read_df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



        test = [subs.get(n, n) for n in pred[:1300]]

        pred1 = [subs.get(n, n) for n in pred[1300:]]



        correct, total = 0, 0

        for i,j in zip(test, Y[:1300]):

            if i == int(j):

                correct += 1

            total += 1



        if correct/total > mx:

            mx = correct/total

            mn = num

            mr = r

    print('Iteration :', r)

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)
kmeans = KMeans(n_clusters = mn, random_state = mr)

pred = kmeans.fit_predict(T1)
a = {}

for item in range(mn):

    a[item] = []



for index, p in enumerate(pred[:1300]):

    a[p].append(index)

    

subs = {}

for item in range(mn):

    subs[item] = int(Counter(read_df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



test = [subs.get(n, n) for n in pred[:1300]]

pred1 = [subs.get(n, n) for n in pred[1300:]]
correct, total = 0, 0

for i,j in zip(test, Y[:1300]):

    if i == int(j):

        correct += 1

    total += 1



print(correct/total)
read_df = pd.read_csv("../input/dmassign1/data.csv", na_values = "?")

final_df = pd.DataFrame({'ID': read_df['ID'].iloc[1300:], 'Class': pred1})

final_df.to_csv("attempt1.csv", index = False)
final_df
from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
create_download_link(final_df)