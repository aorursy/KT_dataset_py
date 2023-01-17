import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image

from scipy.stats import norm
import scipy.stats as stats

import math
%matplotlib inline
from ipywidgets import IntProgress
from IPython.display import display

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# %matplotlib notebook

plt.style.use('ggplot')

import warnings            
warnings.filterwarnings("ignore") 

color = sns.color_palette()

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

from pandas.tools.plotting import parallel_coordinates
veri=pd.read_csv("../input/iris.data.csv", header=None)
veri.columns=['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width', 'Species']
veri.head()
parallel_coordinates(veri, 'Species', colormap=plt.get_cmap("Set1"))
melt=pd.melt(veri,id_vars="Species")
sns.swarmplot(x="variable", y="value", hue="Species", data=melt)
sns.violinplot(x="variable", y="value", hue="Species", data=melt, inner="quart")
sns.boxplot(x="variable", y="value", hue="Species", data=melt)
X=veri.iloc[:,2:3].values
y=veri.iloc[:,4].values

from sklearn.model_selection import train_test_split 
X_egitim,X_dene,y_egitim,y_dene=train_test_split(X, y, test_size=0.25,random_state=0, stratify=y)

from sklearn.ensemble import RandomForestClassifier
sinif=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=1000)
sinif.fit(X_egitim,y_egitim)
y_tahmin=sinif.predict(X_dene)

from sklearn.metrics import accuracy_score
accuracy_score(y_tahmin, y_dene)
X=veri.iloc[:,:-1].values
y=veri.iloc[:,4].values

from sklearn.model_selection import train_test_split 
X_egitim,X_dene,y_egitim,y_dene=train_test_split(X, y, test_size=0.25,random_state=0, stratify=y)
clf_rf = RandomForestClassifier(random_state=0)      
clr_rf = clf_rf.fit(X_egitim,y_egitim)
y_tahmin=clr_rf.predict(X_dene)
accuracy_score(y_tahmin, y_dene)
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(X_egitim,y_egitim)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_egitim.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.title("Feature importances")
plt.bar(range(X_egitim.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X_egitim.shape[1]), veri.columns[indices],rotation=90)
plt.xlim([-1, X_egitim.shape[1]])
plt.show()
X=veri.iloc[:,2:4].values
y=veri.iloc[:,4].values

from sklearn.model_selection import train_test_split 
X_egitim,X_dene,y_egitim,y_dene=train_test_split(X, y, test_size=0.25,random_state=0, stratify=y)

from sklearn.ensemble import RandomForestClassifier
sinif=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=1000)
sinif.fit(X_egitim,y_egitim)
y_tahmin=sinif.predict(X_dene)

from sklearn.metrics import accuracy_score
accuracy_score(y_tahmin, y_dene)
X=veri.iloc[:,:-1].values
y=veri.iloc[:,4].values

from sklearn.model_selection import train_test_split 
X_egitim,X_dene,y_egitim,y_dene=train_test_split(X, y, test_size=0.25,random_state=0, stratify=y)

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier() 
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_egitim, y_egitim)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', veri.columns[:-1:][rfecv.support_])

print("\nSıralama:")
sayaç=1
for i in np.argsort(rfecv.ranking_):
    print("{0}. {1}".format(sayaç,veri.columns[i]))
    sayaç+=1    
