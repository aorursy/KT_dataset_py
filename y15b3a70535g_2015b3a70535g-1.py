import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_o = pd.read_csv("../input/datasetX.csv",sep=",")

data = data_o
#High Dimension data, 59 columns with one hot encoding



data1 = pd.get_dummies(data, columns=["Account1", "History", "GenderType", "Motive", "Account2", "EmploymentPeriod", "InstallmentRate", "Sponsors", "Plotsize", "Plan", "Post", "Housing", "Phone", "Expatriate","Credits"])

data1 = data1.drop(["id"],axis=1)

data1.head()

#High Dimension Data, but lesser. one hot encoding. Encoded only Plan, Housing and Expatriate,History and Motive. 



data2 = pd.get_dummies(data, columns=["Account1", "Account2", "Plan", "Housing", "Expatriate","Motive"])

data2 = data2.drop(["id", "GenderType", "History", "EmploymentPeriod", "InstallmentRate", "Sponsors", "Plotsize", "Post", "Phone", "Credits"],axis=1)

data2.head()
from sklearn.decomposition import PCA

pca2 = PCA(n_components=2)

pca2.fit(data2)

T2 = pca2.transform(data2)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 5,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(data2)

plt.scatter(T2[:, 0], T2[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(data2, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 4).T

y_ac
res1 = pd.DataFrame(y_ac.T)

final = pd.concat([data_o["id"], res1], axis=1).reindex()

final = final.rename(columns={0: "id"})

final.head()



final.to_csv('submission2.csv', index = False)
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



create_download_link(final)