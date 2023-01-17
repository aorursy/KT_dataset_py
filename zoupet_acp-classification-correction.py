# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings



warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## PCA

## pandas

## numpy

## matplotlib.pyplot

## seaborn



%matplotlib inline
import pandas as pd



df = pd.read_excel("/kaggle/input/ml-training-vlib/autos_acp.xls", sheet_name=0,header=0,index_col=0)

finition = df['FINITION']

df = df.drop('FINITION', axis = 1)

df.head(), df.info(), df.describe(), df.tail(), df.columns, df.shape, df.shape[0], df.shape[1]
#classe pour standardisation

from sklearn.preprocessing import StandardScaler

#instanciation

sc = StandardScaler()

#transformation – centrage-réduction

Z = sc.fit_transform(df)



#vérification - librairie numpy

import numpy

#moyenne

print(numpy.mean(Z,axis=0))



#écart-type

print(numpy.std(Z,axis=0))



print(Z)
#classe pour l'ACP

from sklearn.decomposition import PCA

#instanciation

acp = PCA()



#calculs

coord = acp.fit_transform(Z)

#nombre de composantes calculées

print(acp.n_components_) 
#variance expliquée

print(acp.explained_variance_)
list_acp = ["CP1","CP2","CP3","CP4","CP5","CP6","CP7","CP8"]

df_acp = pd.DataFrame(list_acp, columns = ["ACP"])

df_acp['explained_variance'] = acp.explained_variance_
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(6, 8))



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x="explained_variance", y="ACP", data=df_acp,

            label="Total", color="b")
#cumul de variance expliquée

plt.plot(list_acp,numpy.cumsum(acp.explained_variance_ratio_))

plt.title("Explained variance vs. # of factors")

plt.ylabel("Cumsum explained variance ratio")

plt.xlabel("Factor number")

plt.show()
#positionnement des individus dans le premier plan

fig, axes = plt.subplots(figsize=(12,12))

axes.set_xlim(-6,6) #même limites en abscisse

axes.set_ylim(-6,6) #et en ordonnée

#placement des étiquettes des observations

for i in range(18):

    plt.annotate(df.index[i],(coord[i,0],coord[i,1]))

#ajouter les axes

plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)

plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)

#affichage

plt.show()

di = np.sum(Z**2,axis=1)

df_ctr_ind = pd.DataFrame({'ID':df.index,'d_i':di})

df_ctr_ind
df_ctr_ind[df_ctr_ind['d_i'] > df_ctr_ind.d_i.mean()]
#contributions aux axes

eigval = (18-1)/18*acp.explained_variance_



ctr = coord**2

for j in range(8):

    ctr[:,j] = ctr[:,j]/(18*eigval[j])



df_ctr_cp1cp2 = pd.DataFrame({'id':df.index,'CTR_1':ctr[:,0],'CTR_2':ctr[:,1]})
df_ctr_cp1cp2[df_ctr_cp1cp2['CTR_1'] > df_ctr_cp1cp2.CTR_1.mean()]
df_ctr_cp1cp2[df_ctr_cp1cp2['CTR_2'] > df_ctr_cp1cp2.CTR_2.mean()]
#qualité de représentation des individus - COS2

cos2 = coord**2

for j in range(8):

    cos2[:,j] = cos2[:,j]/di

    df_ctr_12 = pd.DataFrame({'id':df.index,'COS2_1':cos2[:,0],'COS2_2':cos2[:,1]})

df_ctr_12
df_ctr_12[df_ctr_12['COS2_1'] > df_ctr_12.COS2_1.mean()]
df_ctr_12[df_ctr_12['COS2_2'] > df_ctr_12.COS2_2.mean()]
df_ctr_12[df_ctr_12['COS2_1'] > df_ctr_12.COS2_1.mean()].id.append(df_ctr_12[df_ctr_12['COS2_2'] > df_ctr_12.COS2_2.mean()].id).drop_duplicates().reset_index(drop = True)
#racine carrée des valeurs propres

sqrt_eigval = np.sqrt(eigval)



#corrélation des variables avec les axes

corvar = np.zeros((8,8))

for k in range(8):

    corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]



#afficher la matrice des corrélations variables x facteurs

print(pd.DataFrame(corvar))
#on affiche pour les deux premiers axes

print(pd.DataFrame({'id':df.columns,'COR_1':corvar[:,0],'COR_2':corvar[:,1]}))
#cercle des corrélations

fig, axes = plt.subplots(figsize=(8,8))

axes.set_xlim(-1,1)

axes.set_ylim(-1,1)

#affichage des étiquettes (noms des variables)

for j in range(8):

    plt.annotate(df.columns[j],(corvar[j,0],corvar[j,1]))



#ajouter les axes

plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)

plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)



#ajouter un cercle

cercle = plt.Circle((0,0),1,color='blue',fill=False)

axes.add_artist(cercle)

#affichage

plt.show()
#cosinus carré des variables

cos2var = corvar**2

df_ctr_variables = pd.DataFrame({'id':df.columns,'COS2_1':cos2var[:,0],'COS2_2':cos2var[:,1]})
df_ctr_variables[df_ctr_variables['COS2_1'] > df_ctr_variables['COS2_1'].mean()]
df_ctr_variables[df_ctr_variables['COS2_2'] > df_ctr_variables['COS2_2'].mean()]
print(finition)
#modalités de la variable qualitative

modalites = numpy.unique(finition)

print(modalites)

#liste des couleurs

couleurs = ['red','green','blue']



#faire un graphique en coloriant les points

fig, axes = plt.subplots(figsize=(12,12))

axes.set_xlim(-6,6)

axes.set_ylim(-6,6)



#pour chaque modalité de la var. illustrative

for c in range(len(modalites)):

 #numéro des individus concernés

 numero = numpy.where(finition == modalites[c])

 #les passer en revue pour affichage

 for i in numero[0]:

    plt.annotate(df.index[i],(coord[i,0],coord[i,1]),color=couleurs[c])



#ajouter les axes

plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)

plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)



#affichage

plt.show()
from sklearn import datasets



digits = datasets.load_digits()

digits
X_digits = digits.data

y_digits = digits.target
X_digits.shape
y_digits.shape
pca = PCA()

pca.fit(X_digits)
pca.explained_variance_ratio_
plt.figure(1, figsize=(8, 5))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca.explained_variance_ratio_.cumsum(), linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_')
from sklearn.pipeline import Pipeline

from sklearn import linear_model



# Création du pipeline et détermination des meilleurs paramètres

logistic = linear_model.LogisticRegression()

pca = PCA()

pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
from sklearn.model_selection import GridSearchCV



n_components = [20, 40, 64]

Cs = np.logspace(-4, 4, 3)

penalties = ["l1", "l2"]



#Parameters of pipelines can be set using ‘__’ separated parameter names:

estimator = GridSearchCV(pipe,

                         dict(pca__n_components=n_components,

                              logistic__C=Cs,

                              logistic__penalty=penalties), scoring = 'accuracy')



estimator.fit(X_digits, y_digits)
print(estimator.best_estimator_)

print(X_digits.shape, y_digits.shape)
pd.DataFrame(estimator.cv_results_)
print("Best parameters set found on development set:")

print()

print(estimator.best_params_)

print()

print("Grid scores on development set:")

print()

means = estimator.cv_results_['mean_test_score']

stds = estimator.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, estimator.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))