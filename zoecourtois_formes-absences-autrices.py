import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import interpolate
df = pd.read_csv("../input/datasetauteur.csv")
df_maj = pd.read_csv("../input/MJ.csv", delimiter=';', encoding="utf-8")

df = df.append(df_maj,ignore_index=True,verify_integrity=True, sort=False)
df = df.drop(columns = ["Source","Thématique","Questionnement","Problématique"])



df.loc[997, "Nombre de titres"] = 3

nan_index = df[df["Nombre de titres"].isnull()].index

df.loc[nan_index, "Nombre de titres"] = 1



niveaux_ordonnes = [

    'sixième', 

    'cinquième', 

    'quatrième', 

    'troisième', 

    'terminale L', 

    "classes préparatoires aux grandes écoles scientifiques", 

    "concours A/L de l'ENS", 

    "agrégation externe de lettres modernes"

]

niveaux_ordonnes_court = [

    '6e', 

    '5e', 

    '4e', 

    '3e', 

    'Tle', 

    "Prépa Sc", 

    "Prépa A/L", 

    "Agreg"

]
df["Genre"].value_counts(normalize=True)*100
df_andro = df[(df["Genre"] == "-") | (df["Genre"].isnull())]

df_andro.sample(5)

#Pour tout voir, enlevez le .sample(5) au dessus et pressez Shift + Enter
df = df[df["Genre"] != "-"]
df_top = df["Autrice - auteur"].value_counts().sort_values(ascending=False)

df_top.head(15)

#Pour voir plus de quinze noms, changez le 15 entre parenthèses par le nombre de votre choix

#puis relancez la cellule en tapant shift/enter
df_F = df[df["Genre"] == "F"]

df_F = df_F["Autrice - auteur"].value_counts().sort_values(ascending=False)

df_F.head(15)

#Pour voir plus de dix noms, changez le 10 entre parenthèses par le nombre de votre choix

#puis relancez la cellule en tapant shift/enter
df_corpus = df[df["Nombre de titres"] >= 3]

df_corpus = df_corpus.sort_values(by="Nombre de titres", ascending = False)

df_corpus["Autrice - auteur"].value_counts()

#Il est possible de voir les corpus de deux oeuvres seulement,

#remplacez le 3 pas un 2, puis pressez Shift + Enter pour relancer la cellule
df_niveau = df.groupby(["Programme"])["Genre"].value_counts(normalize=True) * 100

df_niveau = 100 - df_niveau.xs('M', level=1)

df_niveau = df_niveau.reindex(niveaux_ordonnes)

df_niveau
plt.figure(figsize=(14,6))

graph_niveau = sns.barplot(x=niveaux_ordonnes_court, y=df_niveau.values , palette="GnBu_d")

plt.ylabel("Pourcentage des oeuvres d'autrices")

plt.xlabel("Niveau")
df_agreg = df[df["Programme"] == "agrégation externe de lettres modernes"].groupby(["Année"])["Genre"].value_counts(normalize=True) * 100

df_agreg = 100 - df_agreg.xs('M', level=1)

#Pour lire le pourcentage précis du nombre d'oeuvres d'autrices par année, 

#tapez df_agreg puis pressez la touche "shift" et la touche "enter"
plt.figure(figsize=(14,6))

sns.set(style="darkgrid")

sns.barplot(x=df_agreg.index, y=df_agreg.values, palette="GnBu_d")

plt.ylabel("Pourcentage des oeuvres d'autrices")

plt.xlabel("Sessions de l'Agrégation externe de Lettres Modernes")
# Pour voir toutes les écrivaines qui ont un jour été au programme de l'agrégation, 

# enlevez le # devant ce qui suit, puis pressez shift + enter.



# df[ (df["Genre"] == "F") & (df["Programme"] == "agrégation externe de lettres modernes")]
df[ (df["Genre"] == "F") 

   & (df["Année"].isin([1996,2000,2005,2006])) 

   & (df["Programme"] == "agrégation externe de lettres modernes")]
df_obligatoire = df[df["Niveau d'enseignement"] != "collège"].groupby(["Année"])["Genre"].value_counts(normalize=True) * 100

df_obligatoire = 100 - df_obligatoire.xs('M', level=1)

plt.figure(figsize=(14,6))

sns.set(style="darkgrid")

sns.lineplot(x=df_obligatoire.index, y=df_obligatoire.values, palette="husl")

plt.ylabel("Pourcentage des oeuvres d'autrices dans les programmes prescriptifs")

plt.xlabel("Années")
#Vous souhaitez jeter un coup d'oeil au graphique 

#qui contient aussi les oeuvres non-obligatoires du collège ?

#Enlevez les # ci-dessous, puis pressez Shift + Enter



#df_Fannee = df.groupby(["Année"])["Genre"].value_counts(normalize=True) * 100

#df_Fannee = 100 - df_Fannee.xs('M', level=1)

#plt.figure(figsize=(14,6))

#sns.set(style="darkgrid")

#sns.barplot(x=df_Fannee.index, y=df_Fannee.values, palette="GnBu_d")

#plt.ylabel("Pourcentage des oeuvres d'autrices dans les différents programmes")

#plt.xlabel("Années")