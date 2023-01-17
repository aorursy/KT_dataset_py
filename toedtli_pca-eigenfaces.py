# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")
N=10000
cov = [[3,3],[3,4]] # Kovarianzmatrix

mu = [1,2] #Mittelpunkt der Punktwolke

var1,var2 = np.linalg.eig(cov)[0]  # Sog. Hauptkomponenten: Eigenwerte der Kovarianzmatrix

sig1,sig2 = np.sqrt([var1,var2])   # Standardabweichungen: Wurzeln der Eigenwerte der Kovarianzmatrix
X = np.random.multivariate_normal(mu,cov,N)

plt.scatter(X[:,0],X[:,1],s=1);
eigval,eigvec = np.linalg.eig(np.dot(X.T,X))

np.sqrt(eigval),var1,var2
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X)
#Den Mittelwert können wir aus den Daten schätzen

mu = np.mean(X,axis=0)

plt.scatter(X[:,0],X[:,1],s=1)

ax = plt.gca()



ax.arrow(mu[0], mu[1], sig2*pca.components_[0,0], sig2*pca.components_[0,1],color='r', head_width=0.5, head_length=0.4)

ax.arrow(mu[0], mu[1], sig1*pca.components_[1,0], sig1*pca.components_[1,1],color='r', head_width=0.5, head_length=0.4)

plt.axis('equal');
#Die Hauptkomponenten werden auf die Einheitsvektoren transformiert:

transformed_principal_compontents = pca.transform(pca.components_+mu)

transformed_principal_compontents
#Der selbe Plot nach der Transformation:

Xhat = pca.transform(X)



plt.scatter(Xhat[:,0],Xhat[:,1],s=1)

ax = plt.gca()



#plt.scatter(transformed_principal_compontents[:,0],transformed_principal_compontents[:,1],color='green',marker='x',s=250)

ax.arrow(0, 0, 0, sig1,color='r', head_width=0.5, head_length=0.4)

ax.arrow(0, 0, sig2, 0,color='r', head_width=0.5, head_length=0.4)

plt.axis('equal');

plt.xticks(np.arange(-7,7));

plt.xlim(-7,7);
proj0=np.var(np.dot(pca.components_[0,:].reshape(-1,2),(X-mu).T))

proj1=np.var(np.dot(pca.components_[1,:].reshape(-1,2),(X-mu).T))

print(f'Varianz entlang der 1.Hauptkomponente: {proj0:1.2f}')

print(f'Varianz entlang der 2.Hauptkomponente: {proj1:1.2f}')

#Varianzen der den Daten zu Grunde liegenden Verteilung:

print(f'{var1:1.2f},{var2:1.2f}')
pca.singular_values_/np.sqrt(N),sig1,sig2
!ls ../input/spiraldatensatz/
import pandas as pd

df = pd.read_csv('../input/spiraldatensatz/spiraldatensatz.csv')

df.head()
df.plot(kind='scatter',x='x1',y='x2');
df.plot(kind='scatter',x='x2',y='x3');
df.plot(kind='scatter',x='x1',y='x3');
from sklearn.decomposition import PCA

trf = PCA(n_components=3)
trf.fit(df.values)
trf.singular_values_
import matplotlib.pyplot as plt

plt.figure(1,figsize=(15,5))

plt.subplot(1,2,1)

plt.plot(trf.explained_variance_ratio_);

plt.subplot(1,2,2)

plt.bar([1,2,3],trf.explained_variance_);
trf = PCA(n_components=2)

Xhat = trf.fit_transform(df.values)
plt.scatter(Xhat[:,0],Xhat[:,1],s=0.1);
#trf = PCA(n_components=??)

#Xhat = trf.fit_transform(df.values)

#plt.scatter(Xhat[??],Xhat[??],s=0.1)
trf = PCA(n_components=3)

Xhat = trf.fit_transform(df.values)

df_pca = pd.DataFrame({'pc1':Xhat[:,0], 'pc2':Xhat[:,1], 'pc3':Xhat[:,2]})
df['s']=df.x1**2

df['s'].iloc[0]=1000 #nötig, ansonsten zeichnet plotly alle Scatterpunkte zu gross?

df.head()
import plotly.express as px

px.scatter_3d(df, x='x1', y='x2', z='x3',size='s')
df = pd.read_csv('/kaggle/input/csvversion-der-kreuz-kreis-und-plusdaten/KreuzKreisPlus_train.csv')

df.head()
y=df.target

X=df.iloc[:,:-2].values

X.shape
import matplotlib.pyplot as plt

import random

X2x2 = X.reshape(-1,15,15)

i=random.randint(0,X2x2.shape[0]-1)

plt.imshow(X2x2[i].reshape(15,15))

targetlabel={0:'Kreuz', 1: 'Kreis', 2:'Plus'}

plt.title(f'Label: {targetlabel[y[i]]}');
from sklearn.decomposition import PCA

n_components=50 #Wieviele Hauptkomponenten wollen wir?

trf = PCA(n_components=n_components)

trf.fit_transform(X,None)
plt.figure(1,figsize=(15,20)),plt.subplot(1,3,1),plt.imshow(trf.components_[0].reshape(-1,15));

plt.subplot(1,3,2),plt.imshow(trf.components_[1].reshape(-1,15));

plt.subplot(1,3,3),plt.imshow(trf.components_[2].reshape(-1,15));
plt.bar(np.arange(trf.singular_values_.size),trf.singular_values_);

plt.title('Singulärwerte: Eigenwerte der Kovarianzmatrix');
trf.explained_variance_ratio_
import matplotlib.pyplot as plt

plt.figure(1,figsize=(15,5))

plt.plot(100*np.cumsum(trf.explained_variance_ratio_));

plt.title('Anteil der durch die ersten $m$ Kompontenten erkläreten Varianz [in Prozent]')

plt.ylim(0,100)

plt.xlabel('$m$');
Xhat = trf.fit_transform(X,None)

X.shape,Xhat.shape
X_reconstructed = trf.inverse_transform(Xhat)
import matplotlib.pyplot as plt

import random

X2x2 = X.reshape(-1,15,15)

i=random.randint(0,X2x2.shape[0]-1)

plt.subplot(1,2,1)

im = X2x2[i].reshape(15,15)

im_mean = np.mean(X2x2,axis=0)  

plt.imshow(im-im_mean) #Mittelwert muss abgezogen werden

plt.title('Original')

plt.subplot(1,2,2)

plt.imshow(X_reconstructed[i].reshape(15,15))

plt.title(f'PCA-Rekonstruktion mit n={n_components}')

targetlabel={0:'Kreuz', 1: 'Kreis', 2:'Plus'}

plt.suptitle(f'Label: {targetlabel[y[i]]}');
from sklearn.tree import DecisionTreeClassifier

clf=DecisionTreeClassifier(max_depth=5)
from sklearn.pipeline import Pipeline

pip1 = Pipeline([('dt',clf)])

pip2 = Pipeline([('pca',trf),('dt',clf)])
pip1.fit(X,y)

pip1.score(X,y)
pip2.fit(X,y)

pip2.score(X,y)