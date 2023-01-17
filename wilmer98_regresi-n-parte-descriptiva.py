import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import f1_score,make_scorer, mean_squared_error, f1_score, confusion_matrix, classification_report, roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from random import seed
from random import gauss
import numpy as np
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import stats, polyval
import itertools
import os
import math
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor 
import seaborn as sb
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math 
from scipy import stats
import warnings
import seaborn as sb
from sklearn.decomposition import PCA
import pylab as pl
from scipy.stats import shapiro
from scipy.stats import wilcoxon
# Example of the Pearson's Correlation test
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso,RidgeCV,LassoCV,RidgeClassifier,RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
filename = "/kaggle/input/dm-regresion/data_regression.csv"
Base=pd.read_csv(filename)
Base.head(150)
#eliminar variables
Base=Base.drop(['id'],axis=1)
Base
Base.isnull().sum()
#histogramas de las variables 
Base.hist(bins=10,figsize=(10,10))
plt.show()
summary = Base.describe()
summary = summary.transpose()
summary.head()
ax = sb.boxplot(x=Base["pm2_5"],orient="h")


subBase=Base.iloc[:,1:8];subBase
ax = sb.boxplot(data=subBase, orient="h", palette="Set2",width=0.5)
sb.boxplot(data=Base[["ws","tem","wd"]],orient="v")
sb.boxplot(data=Base[["prec","press"]],orient="v")
# Asimetría y curtosis:

print("Skewness: %f" % Base["pm2_5"].skew())
print("Kurtosis: %f" % Base["pm2_5"].kurt())

# Matriz de correlación:
corrmat = Base.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# Matriz de correlación
k = 7 # Número de variables.
cols = corrmat.nlargest(k, 'pm2_5')['pm2_5'].index
cm = np.corrcoef(Base[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()
#Correlaciones
corr = Base.corr()
corr[['pm2_5']].sort_values(by = 'pm2_5',ascending = False).style.background_gradient()


# Scatter plot:

sns.set()
cols = ['pm2_5', 'hum', 'press', 'prec', 'tem', 'wd', 'ws']
sns.pairplot(Base[cols], size = 2.5)
plt.show()
#funcion de estandarización 
escala=preprocessing.StandardScaler().fit(Base)
# Base de datos
new_z=pd.DataFrame(escala.transform(Base));new_z
new_z.columns=Base.columns
pca = PCA().fit(new_z)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
# Plot a variable factor map for the first two dimensions.
(fig, ax) = plt.subplots(figsize=(10, 10))
for i in range(0, len(pca.components_)):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  #0 for PC1
             pca.components_[1, i],  #1 for PC2
             head_width=0.1,
             head_length=0.1)
    plt.text(pca.components_[0, i] + 0.02,
         pca.components_[1, i] + 0.02,
         new_z.columns.values[i])

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plt.axis('equal')
ax.set_title('Círculo de Correlaciones')
plt.show()
def Funcion_PCA(data):
    # Estimar los ejes principales
    pca1=PCA().fit(data)
    # Ejes retenidos
    LAMBDA=pca1.explained_variance_ratio_[pca1.explained_variance_ratio_>pca1.explained_variance_ratio_.mean()]
    # Coordenadas factoriales
    coord=np.zeros(data.shape[0]*data.shape[1]); coord=coord.reshape(data.shape[0],data.shape[1])
    for i in range(0,data.shape[1]):
        coord[:,i]=pca1.fit_transform(data)[:,i]
    # Correlaciones
    cor=np.zeros(data.shape[1]**2); cor=cor.reshape(data.shape[1],data.shape[1])
    coordenadas=pd.DataFrame(coord)
    for i in range(0,data.shape[1]):
        for j in range(0,data.shape[1]):
            cor[j,i]=data[data.columns[j]].corr(coordenadas[coordenadas.columns[i]])
    # Cosenos cuadrados
    cos2=cor**2
    # Contribuciones
    contrib=np.zeros(data.shape[1]**2); contrib=contrib.reshape(data.shape[1],data.shape[1])
    for k in range(0,data.shape[1]):
        contrib[:,k]=100*cos2[:,k]/cos2[:,k].sum()
    # Gráfico del plano 1-2
    acum=np.cumsum(pca1.explained_variance_ratio_)
    plt.plot(coord[:,0],coord[:,1],'*')
    plt.xlabel('First Component '+str(np.round(100*acum[0],2))+'%')
    plt.ylabel('Second Component '+str(np.round(100*(acum[1]-acum[0]),2))+'%')
    plt.title('Plane 1-2'); plt.show()
    # Tupla de salida
    return (pca1.explained_variance_ratio_,acum,LAMBDA,coord,cor,cos2,contrib)
# Función de Componentes Principales
pca1=Funcion_PCA(new_z)
# Histograma y gráfico de probabilidad normal variable respuesta:

sns.distplot(Base["pm2_5"], fit = norm);
fig = plt.figure()
res = stats.probplot(Base["pm2_5"], plot = plt)
# Histograma y gráfico de probabilidad normal variable velocidad promedio:

sns.distplot(Base["ws"], fit = norm);
fig = plt.figure()
res = stats.probplot(Base["ws"], plot = plt)
# Histograma y gráfico de probabilidad normal variable dirección del viento:

sns.distplot(Base["wd"], fit = norm);
fig = plt.figure()
res = stats.probplot(Base["wd"], plot = plt)
# Gráfico de dispersión variable respuesta:

plt.scatter(Base["pm2_5"], Base["ws"], alpha = 0.5)
# Gráfico de dispersión:

plt.scatter(Base["wd"], Base["pm2_5"], alpha = 0.5)
stat, p = shapiro(Base["pm2_5"])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably Gaussian')
else:
	print('Probably not Gaussian')
stat, p = wilcoxon(Base["pm2_5"], Base["ws"])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
stat, p = wilcoxon(Base["pm2_5"], Base["wd"])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
stat, p = pearsonr(Base["pm2_5"], Base["wd"])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
stat, p = pearsonr(Base["pm2_5"], Base["ws"])
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
	print('Probably independent')
else:
	print('Probably dependent')
