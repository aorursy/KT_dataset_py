import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import mpld3
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import xlsxwriter
import os

from matplotlib import style
from datetime import datetime

%matplotlib inline
%load_ext autoreload
%autoreload 2

style.use('classic')

print(os.listdir("../input"))
dates =['DATPRE']
df = pd.read_csv('../input/COTAHIST_A2009_to_A2018P.csv')
df['DATPRE'] = pd.to_datetime(df['DATPRE'], errors='coerce')
Datas = df[df.CODNEG == 'PETR4']
Datas = df['DATPRE']
df.set_index('DATPRE', inplace=True)
Petrobras4 = df[df.CODNEG == 'PETR4']
ItauUnibanco = df[df.CODNEG == 'ITUB4']
Vale = df[df.CODNEG == 'VALE3']
Bradesco = df[df.CODNEG == 'BBDC4']
# Ambev = df[df.CODNEG == 'ABEV3']
BancodoBrasil = df[df.CODNEG == 'BBAS3']
BMFBovespa = df[df.CODNEG == 'BVMF3']
Cielo = df[df.CODNEG == 'CIEL3']
Itausa = df[df.CODNEG == 'ITSA4']
# Kroton = df[df.CODNEG == 'KROT3']
BRF = df[df.CODNEG == 'BRFS3']
Petrobras3 = df[df.CODNEG == 'PETR3']
Gerdau = df[df.CODNEG == 'GGBR4']
# BBSeguridade = df[df.CODNEG == 'BBSE3']
Usiminas = df[df.CODNEG == 'USIM5']
JBS = df[df.CODNEG == 'JBSS3']
CCR = df[df.CODNEG == 'CCRO3']
LojasRenner = df[df.CODNEG == 'LREN3']
LojasAmericanas = df[df.CODNEG == 'LAME4']
# Magaluiza = df[df.CODNEG == 'MGLU3']
Gol = df[df.CODNEG == 'GOLL4']
# CVC = df[df.CODNEG == 'CVCB3']
Localiza = df[df.CODNEG == 'RENT3']
Estacio = df[df.CODNEG == 'ESTC3']
# RAIADROGASILON = df[df.CODNEG == 'RADL3']
# QUALICORP = df[df.CODNEG == 'QUAL3']
MULTIPLAN = df[df.CODNEG == 'MULT3']
EMBRAER = df[df.CODNEG == 'EMBR3']
SABESP = df[df.CODNEG == 'SBSP3']
BRASKEM = df[df.CODNEG == 'BRKM5']
MRV = df[df.CODNEG == 'MRVE3']
MARFRIG = df[df.CODNEG == 'MRFG3']
NATURA = df[df.CODNEG == 'NATU3']

Petrobras4 = Petrobras4['PREULT']
ItauUnibanco = ItauUnibanco['PREULT']
Vale = Vale['PREULT']
Bradesco = Bradesco['PREULT']
# Ambev = Ambev['PREULT']
BancodoBrasil = BancodoBrasil['PREULT']
BMFBovespa = BMFBovespa['PREULT']
Cielo = Cielo['PREULT']
Itausa = Itausa['PREULT']
# Kroton = Kroton['PREULT']
BRF = BRF['PREULT']
Petrobras3 = Petrobras3['PREULT']
Gerdau = Gerdau['PREULT']
# BBSeguridade = BBSeguridade['PREULT']
Usiminas = Usiminas['PREULT']
JBS = JBS['PREULT']
CCR = CCR['PREULT'] 
LojasRenner = LojasRenner['PREULT']
LojasAmericanas = LojasAmericanas['PREULT']
# Magaluiza = Magaluiza['PREULT']
Gol = Gol['PREULT']
# CVC = CVC['PREULT']
Localiza = Localiza['PREULT']
Estacio = Estacio['PREULT']
# RAIADROGASILON = RAIADROGASILON['PREULT']
# QUALICORP = QUALICORP['PREULT']
MULTIPLAN = MULTIPLAN['PREULT']
EMBRAER = EMBRAER['PREULT']
SABESP = SABESP['PREULT']
BRASKEM = BRASKEM['PREULT']
MRV = MRV['PREULT']
MARFRIG = MARFRIG['PREULT']
NATURA = NATURA['PREULT']

df_brasil = pd.DataFrame(Petrobras4, index=Petrobras4.index)
df_brasil.columns = ['PETR4']
df_brasil['ITUB4'] = ItauUnibanco
df_brasil['VALE3'] = Vale
df_brasil['BBDC4'] = Bradesco
# df_brasil['ABEV3'] = Ambev
df_brasil['BBAS3'] = BancodoBrasil
df_brasil['BVMF3'] = BMFBovespa
df_brasil['CIEL3'] = Cielo
df_brasil['ITSA4'] = Itausa
# df_brasil['KROT3'] = Kroton
df_brasil['BRFS3'] = BRF
df_brasil['PETR3'] = Petrobras3
df_brasil['GGBR4'] = Gerdau
# df_brasil['BBSE3'] = BBSeguridade
df_brasil['USIM5'] = Usiminas
df_brasil['JBSS3'] = JBS
df_brasil['CCRO3'] = CCR
df_brasil['LREN3'] = LojasRenner
df_brasil['LAME4'] = LojasAmericanas
# df_brasil['MGLU3'] = Magaluiza
df_brasil['GOLL4'] = Gol
# df_brasil['CVCB3'] = CVC
df_brasil['RENT3'] = Localiza
df_brasil['ESTC3'] = Estacio
# df_brasil['RADL3'] = RAIADROGASILON
# df_brasil['QUAL3'] = QUALICORP
df_brasil['MULT3'] = MULTIPLAN
df_brasil['EMBR3'] = EMBRAER
df_brasil['SBSP3'] = SABESP
df_brasil['BRKM5'] = BRASKEM
df_brasil['MRVE3'] = MRV
df_brasil['MRFG3'] = MARFRIG
df_brasil['NATU3'] = NATURA

df_brasil.tail()

def mark_mont(date):
    base = date
    ret_10_1 = np.log(base/base.shift(1))
    indice_sharp = -10
    
    returns = ret_10_1.fillna(0)
    df_retornos1 = pd.DataFrame()
    df_variancias1 = pd.DataFrame()
    df_pesos = pd.DataFrame()
    #global dat2
    dat2 = pd.DataFrame()
    print("1")

    # formula nova
    for i in range(3000):
        pesos = np.random.random(len(returns.columns))
        pesos /= np.sum(pesos)
        df_pesos = df_pesos.append([pesos], ignore_index=True)
        retorno = np.sum(returns.mean()*pesos)*250
        risco = np.sqrt(np.dot(pesos.T,np.dot(returns.cov()*250,pesos)))
        df_retornos1 = df_retornos1.append([retorno], ignore_index=True)
        df_variancias1 = df_variancias1.append([risco], ignore_index=True)
        if risco > 0 and (retorno/risco) > indice_sharp:
            indice_sharp = retorno/risco
            ind_pesos = pesos
            ind_retorno = retorno
            ind_risco = risco

        
    print("2")
    df_pesos.columns = ['PETR4', 'ITUB4','VALE3','BBDC4','BBAS3','BVMF3',
                        'CIEL3','ITSA4','BRFS3','PETR3','GGBR4','USIM5',
                        'JBSS3','CCRO3','LREN3','LAME4','GOLL4','RENT3',
                        'ESTC3','MULT3','EMBR3','SBSP3','BRKM5','MRVE3',
                        'MRFG3','NATU3']

    dat2 = pd.concat([df_variancias1, df_retornos1, df_pesos], axis=1, ignore_index=True)
    dat2.columns = ['Risco','Retorno','PETR4', 'ITUB4','VALE3','BBDC4','BBAS3','BVMF3',
                    'CIEL3','ITSA4','BRFS3','PETR3','GGBR4','USIM5','JBSS3','CCRO3',
                    'LREN3','LAME4','GOLL4','RENT3','ESTC3',
                    'MULT3','EMBR3','SBSP3','BRKM5','MRVE3','MRFG3','NATU3']
    print("3")
    data1 = df_pesos
    print("4")
    return df_retornos1, df_variancias1 , indice_sharp, ind_pesos, ind_retorno, ind_risco
retornos, riscos, ind_sharp, peso, retorno, risco = mark_mont(df_brasil['2016-10-01':'2017-01-01'])
plt.scatter(riscos, retornos, c=retornos/riscos, marker='o')
plt.xlabel("Risco")
plt.ylabel("Retorno")
plt.colorbar(label="Indice Sharpe")
print("O indice Sharpe desse período é: " + str(ind_sharp))
