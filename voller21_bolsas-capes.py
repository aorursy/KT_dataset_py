# Bibliotecas importadas

import pandas as pd
# Base de dados importada

import pandas as pd

capes_bolsas = pd.read_csv("../input/capes-bolsas/capes_bolsas.csv")

capes_bolsas.dropna(axis=0)

print(capes_bolsas.head(n=10))
# Selecionando somente bolsas de mestrado e doutorado em física na UFMG

DS_UFMG = capes_bolsas[(capes_bolsas["IES"]=="UFMG") & (capes_bolsas["Ano"]>2010) & (capes_bolsas["Área Conhecimento"]=="FÍSICA")]

print(DS_UFMG)
# Por fim , juntamos os números de bolsas com as seguintes colunas abaixo com groupby

UFMG_groupby = DS_UFMG.groupby(["Ano","Área Conhecimento","MESTRADO","DOUTORADO PLENO"])["Total Linha"].sum()

print(UFMG_groupby)