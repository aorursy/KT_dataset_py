import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.decomposition  import PCA,KernelPCA

from sklearn.metrics import *

from sklearn.model_selection import train_test_split as tts

import sklearn



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

X= df.copy()

y = X.pop('label')
print('The dataset has ',X.shape[0],' tuples and ',X.shape[1],' features\n Now we see a sample of the dataset')
X.sample(5)
fig,axe = plt.subplots(2,5)

axe = axe.flatten()

for i in range(10):

    axe[i].axis('off')

    abc = axe[i].imshow(X.iloc[i,:].to_numpy().reshape(28,28))

_= fig.suptitle('First 10 images',fontsize=20)

plt.tight_layout(pad= 0,h_pad= 0,w_pad= 0.5)
print('Now we see the balance of dataset')

y.value_counts()
print('It shows dataset is having equal values for each class thus is PERFECTLY balanced.\n Now we see the variances in features')
X.var().sort_values(ascending=False)
pca = PCA(random_state=0)

pca.fit(X)
pca.explained_variance_ratio_[:20]
c = np.cumsum(pca.explained_variance_ratio_)

n = np.argmax(c>=0.95)+1

_ = plt.figure(figsize=(16,8))

_ = plt.axis([0, 400, 0, 1])

_ = plt.plot(c,lw=1)

_ = plt.plot([n,n],[0,1],"k:")

_ = plt.xlabel('Number of Componets')

_ = plt.ylabel('Explained Variance Ratio')

_ = plt.suptitle('Explained Variance vs n_Components',fontsize=15)

_ = plt.xticks(np.arange(0,400,10))

_ = plt.yticks(np.arange(0,1.0,0.05))

_ = plt.grid(True)

_ = plt.annotate('n_Componets = '+str(n),xy=(n,0.95),xytext=(n+10,0.9),arrowprops=dict(arrowstyle="->"))
fig, ax = plt.subplots(1, figsize=(14, 10))

plt.scatter(x = X.iloc[:,0], y= X.iloc[:,1],s=10, c=y.to_numpy(), cmap='Spectral', alpha=1.0)
from sklearn.ensemble import RandomForestClassifier

X=df.copy()

X.pop('label')

pca = PCA(n_components=187,random_state=0)

X=pca.fit_transform(X)

rf = RandomForestClassifier(max_depth=50)

rf.fit(X,y)
print(classification_report(y,rf.predict(X)))
X_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

y_test=X_test.pop('label')

X_= pca.transform(X_test)

y_hat=rf.predict(X_)

print(classification_report(y_test,y_hat))
classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

i = [(y_hat[i] != y_test[i]) for i in range(10000)]

X_1 = X_[i]

y_ = y_hat[i]



fig,axe = plt.subplots(5,10)

axe = axe.flatten()

for x in range(50):

    axe[x].axis('off')

    axe[x].set_title(classes[y_[x]])

    abc = axe[x].imshow(pca.inverse_transform(X_1[x,:]).reshape(28,28))

_= fig.suptitle('Some incorrectly classified images',fontsize=20)
import umap

X= df.copy()

y = X.pop('label')

mapper = umap.UMAP(random_state=0,n_neighbors=5).fit(X,y)

X = mapper.transform(X)

rf.fit(X,y)

X_ = mapper.transform(X_test)

y_hat= rf.predict(X_)



print(classification_report(y_test,y_hat))
i = [(y_hat[i] != y_test[i]) for i in range(10000)]

X_1 = X_[i]

y_ = y_hat[i]

fig,axe = plt.subplots(5,10)

axe = axe.flatten()

for x in range(50):

    axe[x].axis('off')

    axe[x].set_title(classes[y_[x]])

    abc = axe[x].imshow(mapper.inverse_transform(X_1[x,:]).reshape(28,28))

_= fig.suptitle('Some incorrectly classified images',fontsize=20)