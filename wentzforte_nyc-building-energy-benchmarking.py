import xgboost as xgb

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt



import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab

%matplotlib inline



from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')



import os

#print(os.listdir("../input"))
def showCorr(df):

    fig = plt.subplots(figsize = (10,10))

    sb.set(font_scale=1.5)

    sb.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

    plt.show()
treino_base = pd.read_csv('../input/dataset_treino.csv')

treino = treino_base.copy()

treino.describe().T
teste_base = pd.read_csv('../input/dataset_teste.csv')

teste = teste_base.copy()

teste.info()
treino.head()
sb.countplot(x='ENERGY STAR Score',data=treino)
featuresTeste = [

    "Postal Code",

    "Latitude",

    "Longitude",

    "DOF Gross Floor Area",

    "Year Built",

    "Number of Buildings - Self-reported",

    "Occupancy",

    "Site EUI (kBtu/ft²)",

    "Property GFA - Self-Reported (ft²)",

    "Source EUI (kBtu/ft²)",

    "Community Board",

    "Council District",

    "Census Tract",

    "Weather Normalized Site EUI (kBtu/ft²)",

    "Weather Normalized Site Electricity Intensity (kWh/ft²)",

    "Weather Normalized Source EUI (kBtu/ft²)",

    "Weather Normalized Site Natural Gas Use (therms)",

    "Weather Normalized Site Electricity (kWh)",

    "Water Use (All Water Sources) (kgal)",

    "Water Intensity (All Water Sources) (gal/ft²)",

    "Total GHG Emissions (Metric Tons CO2e)",    

    "Direct GHG Emissions (Metric Tons CO2e)",

    "Indirect GHG Emissions (Metric Tons CO2e)",

    "Electricity Use - Grid Purchase (kBtu)",

    "Natural Gas Use (kBtu)",

    "Manhattan", "Queens", "Brooklyn", "Staten Island"]



featuresTreino = featuresTeste + ["ENERGY STAR Score"]
def setCity(df):

    lista = df["Borough"].value_counts()

    for item in lista.index:

        df[item] = df["Borough"] == item

        df[item] = df[item].astype(int)

    return df
def setPostalCode(df):

    df["Postal Code"] = df["Postal Code"].str.replace("-", "")

    df["Postal Code"] = df["Postal Code"].astype(int)  

    return df
def setMean(df, features):

    df = df.replace('Not Available',np.nan, regex=True)

    for item in features:

        if df[item].dtype == "object":

            df[item] = df[item].astype(float)



    for item in features:

        df[item] = df[item].fillna(df[item].mean())        

    

    return df
def setGeneralData(df, features):    

    df["Number of Buildings - Self-reported"][df["Number of Buildings - Self-reported"] > 30 ] = 30

    df["Number of Buildings - Self-reported"][df["Number of Buildings - Self-reported"] <= 0] = 1

    df["Occupancy"][df["Occupancy"] <= 0] = 1

    df["Site EUI (kBtu/ft²)"][df["Site EUI (kBtu/ft²)"] <= 0] = 1

    df["Property GFA - Self-Reported (ft²)"][df["Property GFA - Self-Reported (ft²)"] >= 2500000] = 2500000

    df["Source EUI (kBtu/ft²)"][df["Source EUI (kBtu/ft²)"] < 1] = 1

    df["Year Built"][df["Year Built"] < 1800] = 1800

    df["Year Built"][df["Year Built"] > 2015] = 2015    

    df = df.round(2)

    

    return df
showCorr(treino)
treino.head()
treino = setPostalCode(treino)

teste = setPostalCode(teste)



treino = setCity(treino)

teste = setCity(teste)



treino = setMean(treino, featuresTreino)

teste = setMean(teste, featuresTeste)



treino = setGeneralData(treino, featuresTreino)

teste = setGeneralData(teste, featuresTeste)



print(treino.shape)

print(teste.shape)

treino = treino.filter(items=featuresTreino)

teste = teste.filter(items=featuresTeste)

print(treino.shape)

print(teste.shape)



showCorr(treino)
showCorr(treino)
treino.dtypes
X_train, X_test, y_train, y_test = train_test_split(treino.drop(columns=['ENERGY STAR Score']), pd.DataFrame(treino["ENERGY STAR Score"]))
finalModel = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=1000, n_jobs=50)

finalModel
finalModel.fit(X_train, y_train, eval_metric='mae')
y_pred = finalModel.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))
y_test['ENERGY STAR Score'] = finalModel.predict(X_test).round()

y_test.classe = y_test["ENERGY STAR Score"].astype(int)

sb.countplot(x='ENERGY STAR Score',data= y_test)
teste.head()
envio_final = pd.DataFrame(teste_base["Property Id"])

envio_final['score'] = finalModel.predict(teste).round()

envio_final['score'] = envio_final["score"].astype(int)

sb.countplot(x='score',data=envio_final)

envio_final.describe().T
envio_final.score[envio_final.score < 1] = 1

envio_final.score[envio_final.score > 100] = 100

sb.countplot(x='score',data=envio_final)
envio_final.to_csv('final.csv', index=False)