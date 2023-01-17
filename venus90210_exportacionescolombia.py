import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sys

import os

import seaborn as sns; sns.set_style('darkgrid')

import numpy as np

import scipy.stats as sp

print(os.listdir("../input"))

print(os.listdir("../input/exportaciones2016"))

print(os.listdir("../input/exportaciones2017"))

print(os.listdir("../input/exportaciones2018"))

print(os.listdir("../input/exportaciones2018"))
print(os.listdir("../input/industria"))
datasetEnero2018=pd.read_csv('../input/exportaciones2018/Enero 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetFebrero2018=pd.read_csv('../input/exportaciones2018/Febrero 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetMarzo2018=pd.read_csv('../input/exportaciones2018/Marzo 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetAbril2018=pd.read_csv('../input/exportaciones2018/Abril 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetMayo2018=pd.read_csv('../input/exportaciones2018/Mayo 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetJunio2018=pd.read_csv('../input/exportaciones2018/Junio 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetJulio2018=pd.read_csv('../input/exportaciones2018/Julio 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetAgosto2018= pd.read_csv('../input/exportaciones2018/Agosto 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetSeptiembre2018=pd.read_csv('../input/exportaciones2018/Septiembre 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetOctubre2018=pd.read_csv('../input/exportaciones2018/Octubre 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetNoviembre2018=pd.read_csv('../input/exportaciones2018/Noviembre 2018.csv', sep=';',decimal=',',encoding = "ISO-8859-1")

datasetDiciembre2018=pd.read_csv('../input/exportaciones2018/Diciembre.csv', sep=';',decimal=',',encoding = "ISO-8859-1")



datasetEnero2017=pd.read_csv('../input/exportaciones2017/Enero 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetFebrero2017=pd.read_csv('../input/exportaciones2017/Febrero  2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetMarzo2017=pd.read_csv('../input/exportaciones2017/Marzo 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetAbril2017=pd.read_csv('../input/exportaciones2017/Abril 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetMayo2017=pd.read_csv('../input/exportaciones2017/Mayo 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetJunio2017=pd.read_csv('../input/exportaciones2017/Junio 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetJulio2017=pd.read_csv('../input/exportaciones2017/Julio 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetAgosto2017 = pd.read_csv('../input/exportaciones2017/Agosto 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetSeptiembre2017=pd.read_csv('../input/exportaciones2017/Septiembre 2017.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetOctubre2017=pd.read_csv('../input/exportaciones2017/Octubre 2017.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetNoviembre2017=pd.read_csv('../input/exportaciones2017/Noviembre 2017.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetDiciembre2017=pd.read_csv('../input/exportaciones2017/Diciembre 2017.csv',encoding = "ISO-8859-1", sep=';',decimal=',')



datasetEnero2016=pd.read_csv('../input/exportaciones2016/Enero 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetFebrero2016=pd.read_csv('../input/exportaciones2016/Febrero 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetMarzo2016=pd.read_csv('../input/exportaciones2016/Marzo 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetAbril2016=pd.read_csv('../input/exportaciones2016/Abril 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetMayo2016=pd.read_csv('../input/exportaciones2016/Mayo 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetJunio2016=pd.read_csv('../input/exportaciones2016/Junio 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetJulio2016=pd.read_csv('../input/exportaciones2016/Julio 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetAgosto2016 = pd.read_csv('../input/exportaciones2016/Agosto 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetSeptiembre2016=pd.read_csv('../input/exportaciones2016/Septiembre 2016.csv',encoding = "ISO-8859-1", sep=',',decimal='.')

datasetOctubre2016=pd.read_csv('../input/exportaciones2016/Octubre 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetNoviembre2016=pd.read_csv('../input/exportaciones2016/Noviembre 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

datasetDiciembre2016=pd.read_csv('../input/exportaciones2016/Diciembre 2016.csv',encoding = "ISO-8859-1", sep=';',decimal=',')

#CARGAMOS DATASET CON INFORMACIÓN DE LA RELACIÓN DE LOS CODIGOS CIIU Y LAS SUBPARTIDA

relacionSubpartidaCIIU=pd.read_csv('../input/industria/subpartida_ciiu.csv',encoding = "ISO-8859-1", sep=';',decimal=',')
#CARGAMOS DATASET CON INFORMACIÓN DE LOS CÓDIGOS CIIU

dataCiiuActividad=pd.read_excel('../input/sectoreseconomicos/relacionciiusector.xlsx')
#CARGAMOS DATASET CON INFORMACIÓN DE LOS CÓDIGOS CIIU

codigosCIIU=pd.read_csv('../input/industria/CIUU.csv',encoding = "ISO-8859-1", sep=';',decimal=',')
#CARGAMOS DATASET CON INFORMACIÓN DE LOS CÓDIGOS CIIU

dataTrm=pd.read_excel('../input/tasarepresentativa/trmpromediomes.xlsx')

dataTrm.head()
#Unificación de fecha del Grupo de Datos de Abril de 2018

datasetAbril2018['FECH'] = 1804 
datasettotal2018 = pd.concat([datasetEnero2018,datasetFebrero2018,datasetMarzo2018,datasetAbril2018,datasetMayo2018,datasetJunio2018,datasetJulio2018,

                    datasetAgosto2018,datasetSeptiembre2018,datasetOctubre2018,datasetNoviembre2018,datasetDiciembre2018])



datasettotal2017 = pd.concat([datasetEnero2017,datasetFebrero2017,datasetMarzo2017,datasetAbril2017,datasetMayo2017,datasetJunio2017,datasetJulio2017,

                    datasetAgosto2017,datasetSeptiembre2017,datasetOctubre2017,datasetNoviembre2017,datasetDiciembre2017])



datasettotal2016 = pd.concat([datasetEnero2016,datasetFebrero2016,datasetMarzo2016,datasetAbril2016,datasetMayo2016,datasetJunio2016,datasetJulio2016,

                    datasetAgosto2016,datasetSeptiembre2016,datasetOctubre2016,datasetNoviembre2016,datasetDiciembre2016])



datasettotal = pd.concat([datasettotal2017,datasettotal2016,datasettotal2018])

datasettotal.head()

datasettotal['FOBDOL']=datasettotal['FOBDOL'].astype(str)

datasettotal['FOBDOL'] = [c.replace(",",".") for c in datasettotal['FOBDOL']]

datasettotal['FOBDOL'] = [c.replace(" ","0") for c in datasettotal['FOBDOL']]

datasettotal['FOBDOL'] = datasettotal['FOBDOL'].astype('float64')
datasettotal['CANTI'] = datasettotal['CANTI'].astype(str)

datasettotal['CANTI'] = [c.replace(",",".") for c in datasettotal['CANTI']]

datasettotal['CANTI'] = [c.replace(" ","0") for c in datasettotal['CANTI']]

datasettotal['CANTI'] = datasettotal['CANTI'].astype('float64')
# Visualización previa del tipo de datos de las columnas en el dataset

datasettotal.dtypes
#METODO PARA TOMAR LA PARTIDA DE UNA SUBPARTIDA

def definirPartida(posicion):

    partida = posicion[0:4]

    return partida
#FUNCIÓN PARA TOMAR EL CAPÍTULO DE UNA SUBPARTIDA

def definirCapitulo(posicion):

    capitulo = posicion[0:2]

    return capitulo
#SE AGREGA AL DATASET LA COLUMNA CON LA PARTIDA

PARTIDA = np.array([definirPartida(str(POSAR)) for POSAR in datasettotal['POSAR'].values])

datasettotal = datasettotal.assign(PARTIDA = PARTIDA)



#SE AGREGA AL DATASET LA COLUMNA CON LA PARTIDA

CAPITULO = np.array([definirCapitulo(str(POSAR)) for POSAR in datasettotal['POSAR'].values])

datasettotal = datasettotal.assign(CAPITULO = CAPITULO)
datasettotal.groupby(['FECH'])['FOBDOL'].sum().plot(kind='bar',figsize=(27, 9))
datasettotal.groupby(['FECH'])['CANTI'].sum().plot(kind='bar',figsize=(27, 9))
datasettotal.groupby(['CAPITULO'])['FOBDOL'].sum().plot(kind='bar', grid=True,figsize=(27, 9))
datasettotal.CAPITULO.describe()
#ELIMINAR COLUMNAS CON INFORMACIÓN QUE NO SERÁ USADA

subpartida_ciiu = relacionSubpartidaCIIU.drop(['Año','Cuode','CIIU Rev. 3.0 A.C','CIIU Rev. 4.0 A.C','CPC Ver. 1.0 A.C.','CPC Ver. 2.0 A.C.',

                             'OBSERVACIONES','Vigencia','Decreto','CUCI Rev.2','CUCI Rev.3','CUCI Rev.4','Fecha_Vi','Fecha_Ex'], axis=1)

subpartida_ciiu.head()
#BORRAMOS REGISTROS VACIOS EXISTEN EN EL SET DE DATOS

subpartida_ciiu.dropna()
#AGREGAMOS LA COLUMNA CAPITULO PARA USAR EN EL CRUCE CON EL CIIU

CAPITULO = np.array([definirCapitulo(str(POSAR)) for POSAR in subpartida_ciiu['Subpartida Arancelaria'].values])

subpartida_ciiu = subpartida_ciiu.assign(CAPITULO = CAPITULO)
#AGREGAMOS LA COLUMNA PARTIDA PARA USAR EN EL CRUCE CON EL CIIU

PARTIDA = np.array([definirPartida(str(POSAR)) for POSAR in subpartida_ciiu['Subpartida Arancelaria'].values])

subpartida_ciiu = subpartida_ciiu.assign(PARTIDA = PARTIDA)

subpartida_ciiu.head()
#SE CONSTRUYE SET DE DATOS CON DATOS NO REPETIDOS DE RELACION DE SUBPARTIDAS Y CIIU

supartidaciiu_resument = subpartida_ciiu.drop(['Subpartida Arancelaria','Descripción Arancelaria','Descripción Arancelaria','CGCE Rev.3 ','PARTIDA'],

                                              axis=1)
supartidaciiu_resument = supartidaciiu_resument.drop_duplicates()

#BORRAMOS REGISTROS VACIOS EXISTEN EN EL SET DE DATOS

supartidaciiu_resument = supartidaciiu_resument.dropna()

supartidaciiu_resument.head()
#CONSTRUCCIÓN DE DICCIONARIO DE TRM

diccionarioTRM=dataTrm.set_index('anomes')['Promedio'].to_dict() 

print(diccionarioTRM)
#CONSTRUCCIÓN DE DICCIONARIO DE DATOS CAPITULOS - CIIU

diccionariosubpartida=supartidaciiu_resument.set_index('CAPITULO')['CIIU Rev. 2 DANE'].to_dict() 

print(diccionariosubpartida)
#CONSTRUCCIÓN DE DICCIONARIO DE DATOS CAPITULOS - CLASE

diccionariosector=supartidaciiu_resument.set_index('CAPITULO')['CIIU Rev. 3.1 A.C'].to_dict() 

print(diccionariosector)
supartidaciiu_resument.head()
dataCiiuActividad['Clase'] = dataCiiuActividad['Clase'].astype(int)

dataCiiuActividad.head()
#CONSTRUCCIÓN DE DICCIONARIO DE DATOS

diccionarioActividad=dataCiiuActividad.set_index('Clase')['DescripcionSeccion'].to_dict() 

print(diccionarioActividad)
#CREAMOS UNA COLUMNA 'CIIU' CON LA INFORMACIÓN DE LA PARTIDA

datasettotal['CIIU'] = datasettotal['CAPITULO']

datasettotal = datasettotal.dropna()



datasettotal['CIIU'] = datasettotal['CIIU'].replace(' ','01') 

datasettotal.dtypes
datasettotal['CIIU'] = datasettotal['CIIU'].apply(lambda x:diccionariosubpartida[x])
#CREAMOS UNA COLUMNA 'CIIU' CON LA INFORMACIÓN DE LA PARTIDA

datasettotal['TRM'] = datasettotal['FECH']

datasettotal['TRM'] = datasettotal['TRM'].replace(' ','0') 

datasettotal['TRM']=datasettotal['TRM'].astype(int)
for i in datasettotal['TRM']:

    diccionarioTRM.setdefault(i,'0')
datasettotal['TRM'] = datasettotal['TRM'].apply(lambda x:diccionarioTRM[x])
datasettotal.head()
#CREAMOS UNA COLUMNA 'SECTOR' CON LA INFORMACIÓN DE LA PARTIDA

datasettotal['SECTOR'] = datasettotal['CAPITULO']

datasettotal = datasettotal.dropna()



datasettotal['SECTOR'] = datasettotal['SECTOR'].replace(' ','01') 

#Se pasan los valores del sector al que corresponden a cada registro

datasettotal['SECTOR'] = datasettotal['SECTOR'].apply(lambda x:diccionariosector[x])
datasettotal['SECTOR']=datasettotal['SECTOR'].astype(int)
#Se agragegan los valores que no se encuentren en el diccionario de actividades económicas 

for i in datasettotal['SECTOR']:

    diccionarioActividad.setdefault(i,'')
datasettotal['SECTOR'] = datasettotal['SECTOR'].apply(lambda x:diccionarioActividad[x])
def asignarAno(fecha):

    ano = fecha[0:2]

    return ano



ANO = np.array([asignarAno(str(FECH)) for FECH in datasettotal['FECH'].values])

datasettotal = datasettotal.assign(ANO = ANO)
def asignarMes(fecha):

    mes = fecha[2:]

    return mes



MES = np.array([asignarMes(str(FECH)) for FECH in datasettotal['FECH'].values])

datasettotal = datasettotal.assign(MES = MES)

def definirSubpartida(posicion):

    subpartida = posicion[0:6]

    return subpartida



SUBPARTIDA = np.array([definirSubpartida(str(POSAR)) for POSAR in datasettotal['POSAR'].values])

datasettotal = datasettotal.assign(SUBPARTIDA = SUBPARTIDA)
#Se eliminan columnas que no son consideradas relevantes para el análisis actual



#Codigo de aduanas

datasettotal.drop(['ADUA','FLETES','SEGURO','OTROSG','REGIM','RAZ_SIAL','AGRENA','PBK','DPTO1',

                   'DPTO2','PNK','SISESP','FOBPES','COD_SAL1','COD_SAL','POSAR','NIT','FECH'], axis=1)
datasettotal.head()
#Tabla de Frecuencias de exportaciones por fechas

datasettotal.groupby(['SECTOR'])['FOBDOL'].count().plot(kind='BAR',subplots=True,figsize=(10, 9))

datasettotal.groupby(["MES"]).size().plot(kind='PIE',subplots=True, autopct='%1.1f%%',figsize=(10, 8))
datasettotal.groupby(["ANO"]).size().plot(kind='PIE',subplots=True, autopct='%1.1f%%',figsize=(10, 8))
plot2 = datasettotal['ANO'].value_counts().plot(kind='bar',

                                            figsize=(16, 7),

                                            title='Exportaciones Anuales')

 
# Tabla de contingencia class / survived

pd.crosstab(index=datasettotal['POSAR'],

            columns=datasettotal['CAPITULO'], margins=True)
#HISTOGRAMA DE TASA REPRESENTATIVA DE LA MONEDA

plt.hist(datasettotal.TRM, 10,density=1)

plt.ylabel('frequencia')

plt.xlabel('TRM')

plt.title('Histograma DE TRM')

plt.show()
#HISTOGRAMA DE TASA REPRESENTATIVA DE LA MONEDA

plt.hist(datasettotal.TRM, 10,density=1)

plt.ylabel('frequencia')

plt.xlabel('TRM')

plt.title('Histograma DE TRM')

plt.show()
#HISTOGRAMA DE TASA REPRESENTATIVA DE LA MONEDA





dist= sns.distplot(datasettotal.TRM)

 
def compute_freq_chi2(x,y):

    freqtab = pd.crosstab(x,y)

    print("Frequency table")

    print("============================")

    print(freqtab)

    print("============================")

    chi2,pval,dof,expected = sp.chi2_contingency(freqtab)

    print("ChiSquare test statistic: ",chi2)

    print("p-value: ",pval)

    return
datasettotal.head()
datasettotal.dtypes
tablachi2 = compute_freq_chi2(datasettotal.SECTOR,datasettotal.ANO)
tablachi2 = compute_freq_chi2(datasettotal.SECTOR,datasettotal.TRM)
tablachi2 = compute_freq_chi2(datasettotal.SECTOR,datasettotal.FOBDOL)
tablachi2 = compute_freq_chi2(datasettotal.FOBPES,datasettotal.TRM)
sns.set(style="white")



# Compute the correlation matrix

corr = datasettotal.corr()



# Se genera una mascara para el triangulo superior

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Configuración de la imagen

f, ax = plt.subplots(figsize=(11, 9))



# Generar un mapa de colores personalizado

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Dibuja el mapa de calor con la máscara y la relación de aspecto

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
datasettotal.corr()