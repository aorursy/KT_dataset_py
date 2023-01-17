import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from pandas import DataFrame 

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture 

from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split

from sklearn import metrics
raw_df = pd.read_csv('../input/ccdata/CC GENERAL.csv')

raw_df = raw_df.drop('CUST_ID', axis = 1) 

raw_df.fillna(method ='ffill', inplace = True) 

raw_df.head(2)
# Standardize data

scaler = StandardScaler() 

scaled_df = scaler.fit_transform(raw_df) 

  

# Normalizing the Data 

normalized_df = normalize(scaled_df) 

  

# Converting the numpy array into a pandas DataFrame 

normalized_df = pd.DataFrame(normalized_df) 

  

# Reducing the dimensions of the data 

pca = PCA(n_components = 2) 

X_principal = pca.fit_transform(normalized_df) 

X_principal = pd.DataFrame(X_principal) 

X_principal.columns = ['P1', 'P2'] 

  

X_principal.head(2)
gmm = GaussianMixture(n_components = 3) 

gmm.fit(X_principal)
# Visualizing the clustering 

plt.scatter(X_principal['P1'], X_principal['P2'],  

           c = GaussianMixture(n_components = 3).fit_predict(X_principal), cmap =plt.cm.winter, alpha = 0.6) 

plt.show() 
def SelBest(arr:list, X:int)->list:

    '''

    returns the set of X configurations with shorter distance

    '''

    dx=np.argsort(arr)[:X]

    return arr[dx]
n_clusters=np.arange(2, 8)

sils=[]

sils_err=[]

iterations=20

for n in n_clusters:

    tmp_sil=[]

    for _ in range(iterations):

        gmm=GaussianMixture(n, n_init=2).fit(X_principal) 

        labels=gmm.predict(X_principal)

        sil=metrics.silhouette_score(X_principal, labels, metric='euclidean')

        tmp_sil.append(sil)

    val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))

    err=np.std(tmp_sil)

    sils.append(val)

    sils_err.append(err)
plt.errorbar(n_clusters, sils, yerr=sils_err)

plt.title("Silhouette Scores", fontsize=20)

plt.xticks(n_clusters)

plt.xlabel("N. of clusters")

plt.ylabel("Score")

#Courtesy of https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms. Here the difference is that we take the squared root, so it's a proper metric



def gmm_js(gmm_p, gmm_q, n_samples=10**5):

    X = gmm_p.sample(n_samples)[0]

    log_p_X = gmm_p.score_samples(X)

    log_q_X = gmm_q.score_samples(X)

    log_mix_X = np.logaddexp(log_p_X, log_q_X)



    Y = gmm_q.sample(n_samples)[0]

    log_p_Y = gmm_p.score_samples(Y)

    log_q_Y = gmm_q.score_samples(Y)

    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)



    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))

            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)
n_clusters=np.arange(2, 8)

iterations=20

results=[]

res_sigs=[]

for n in n_clusters:

    dist=[]

    

    for iteration in range(iterations):

        train, test=train_test_split(X_principal, test_size=0.5)

        

        gmm_train=GaussianMixture(n, n_init=2).fit(train) 

        gmm_test=GaussianMixture(n, n_init=2).fit(test) 

        dist.append(gmm_js(gmm_train, gmm_test))

    selec=SelBest(np.array(dist), int(iterations/5))

    result=np.mean(selec)

    res_sig=np.std(selec)

    results.append(result)

    res_sigs.append(res_sig)
plt.errorbar(n_clusters, results, yerr=res_sigs)

plt.title("Distance between Train and Test GMMs", fontsize=20)

plt.xticks(n_clusters)

plt.xlabel("N. of clusters")

plt.ylabel("Distance")

plt.show()
n_clusters=np.arange(2, 8)

bics=[]

bics_err=[]

iterations=20

for n in n_clusters:

    tmp_bic=[]

    for _ in range(iterations):

        gmm=GaussianMixture(n, n_init=2).fit(X_principal) 

        

        tmp_bic.append(gmm.bic(X_principal))

    val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))

    err=np.std(tmp_bic)

    bics.append(val)

    bics_err.append(err)
plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')

plt.title("BIC Scores", fontsize=20)

plt.xticks(n_clusters)

plt.xlabel("N. of clusters")

plt.ylabel("Score")

plt.legend()
plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')

plt.title("Gradient of BIC Scores", fontsize=20)

plt.xticks(n_clusters)

plt.xlabel("N. of clusters")

plt.ylabel("grad(BIC)")

plt.legend()