#STEP 1: Get right arrows in quiver



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import StandardScaler,normalize

from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture as GMM

from sklearn.manifold import TSNE



import warnings #To hide warnings



#1.1: Set the stage

from IPython.core.interactiveshell import InteractiveShell

from IPython.core.display import display, HTML



InteractiveShell.ast_node_interactivity = "all"

warnings.filterwarnings("ignore")
cc = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

cc.info()

cc[cc.columns[cc.isna().any()]].isna().sum().to_frame().T

cc.sample(5)

cc.describe()
cc.quantile([0.75,0.8,.85,.9,.95,1])
display(HTML('<h4>There are '+str(np.sum(cc.BALANCE>cc.CREDIT_LIMIT))

             +' customers in the list who have more balance than the credit limit assigned. '

             +'It may be due to more payament than usage and/or continuous pre-payment.</h4>'))
cc.rename(columns = {col:col.lower() for col in cc.columns.values},inplace=True)

sns.jointplot(cc.credit_limit,cc.minimum_payments,kind = 'kde', dropna=True)
cc.fillna(cc.median(),inplace=True) #More outliers thus median in both cases

cust = cc.cust_id

cc.drop(columns = ['cust_id'],inplace=True)
ss = StandardScaler()

X= normalize(ss.fit_transform(cc.copy()))

X = pd.DataFrame(X,columns=cc.columns.values)
fig, axs = plt.subplots(6,3, figsize=(20, 20))

for i in range(17):

        p = sns.distplot(cc[cc.columns[i]], ax=axs[i//3,i%3],kde_kws = {'bw':2})

        p = sns.despine()

plt.show()
X.boxplot(figsize = (30,25),grid=True,fontsize=25,rot=90)
plt.figure(figsize=(16,12))

p = sns.heatmap(cc.corr(),annot=True,cmap='jet').set_title("Correlation of credit card data\'s features",fontsize=20)

plt.show()
#Selecting correct number of components for GMM

models = [GMM(n,random_state=0).fit(X) for n in range(1,12)]

d = pd.DataFrame({'BIC Score':[m.bic(X) for m in models],

                  'AIC Score': [m.aic(X) for m in models]},index=np.arange(1,12))

d.plot(use_index=True,title='AIC and BIC Scores for GMM wrt n_Compnents',figsize = (10,5),fontsize=12)
from sklearn.base import ClusterMixin

from yellowbrick.cluster import KElbow



class GMClusters(GMM, ClusterMixin):



    def __init__(self, n_clusters=1, **kwargs):

        kwargs["n_components"] = n_clusters

        kwargs['covariance_type'] = 'full'

        super(GMClusters, self).__init__(**kwargs)



    def fit(self, X):

        super(GMClusters, self).fit(X)

        self.labels_ = self.predict(X)

        return self 



oz = KElbow(GMClusters(), k=(2,12), force_model=True)

oz.fit(X)

oz.show()
model= models[6]

model.n_init = 10

model
clusters = model.fit_predict(X)

display(HTML('<b>The model has converged :</b>'+str(model.converged_)))

display(HTML('<b>The model has taken iterations :</b>'+str(model.n_iter_)))
sns.countplot(clusters).set_title('Cluster sizes',fontsize=20)
cc1 = cc.copy()

cc1['cluster']=clusters

for c in cc1:

    if c != 'cluster':

        grid= sns.FacetGrid(cc1, col='cluster',sharex=False,sharey=False)

        p = grid.map(sns.distplot, c,kde_kws = {'bw':2})

plt.show()
#cc1.groupby('cluster').agg({np.min,np.max,np.mean}).T

for i in range(7):

    display(HTML('<h2>Cluster'+str(i)+'</h2>'))

    cc1[cc1.cluster == i].describe()
tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(X.copy())



plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker=10,

            s=10,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=clusters   # Colour as per gmm

            )
density = model.score_samples(X)

density_threshold = np.percentile(density,4)

cc1['cluster']=clusters

cc1['Anamoly'] = density<density_threshold

cc1
df = cc1.melt(['Anamoly'], var_name='cols',  value_name='vals')



g = sns.FacetGrid(df, row='cols', hue="Anamoly", palette="Set1",sharey=False,sharex=False,aspect=3)

g = (g.map(sns.distplot, "vals", hist=True, rug=True,kde_kws = {'bw':2}).add_legend())
unanomaly = X[density>=density_threshold]

c = clusters[density>=density_threshold]

tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(unanomaly)

plt.figure(figsize=(15,10))

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],marker='x',s=10, linewidths=5, c=c)