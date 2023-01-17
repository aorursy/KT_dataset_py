# MODELO HISTORICO XGB





#!pip3 install --trusted-host artifactory.lima.bcp.com.pe --index-url https://artifactory.lima.bcp.com.pe/artifactory/api/pypi/python-pypi/simple xgboost

#!pip3 install --trusted-host artifactory.lima.bcp.com.pe --index-url https://artifactory.lima.bcp.com.pe/artifactory/api/pypi/python-pypi/simple sklearn

#!pip3 install --trusted-host artifactory.lima.bcp.com.pe --index-url https://artifactory.lima.bcp.com.pe/artifactory/api/pypi/python-pypi/simple featuretools



import pandas as pd

import pickle as pickle

from datetime import datetime

import numpy as np

import pkg_resources

from scipy.stats import uniform, randint

import matplotlib.pyplot as plt

import math

from math import log

import sklearn as sklearn

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.metrics import r2_score

from xgboost import plot_importance



pd.options.display.html.table_schema=True #Para ver con barra los dataframes



campaña = pd.read_csv('/content/drive/My Drive/BELCORP/campana_consultora.csv')

test = pd.read_csv('/content/drive/My Drive/BELCORP/predict_submission.csv',  encoding='latin-1')



del campaña["Unnamed: 0"]





#----- Se obtiene los IDs a utilizar



#Id's para el train

ids = pd.DataFrame()

ids["IdConsultora"] = test.idconsultora





#----- DESFASE



def mesToNumero(codmes):

  anio = codmes//100

  mes = codmes%100

  return anio*18+mes

def numeroToMes(codmes):

  anio = codmes//18

  mes = codmes%18

  if(mes==0):

    anio-=1

    mes=18

  return anio*100+mes

def restarMes(codmes):

  return numeroToMes(mesToNumero(codmes)-1)





d = campaña[["IdConsultora","campana","Flagpasopedido"]]

d['campana_1'] = d['campana'].apply(restarMes)

del d['campana']

d = d.rename(columns={'campana_1': 'campana'}) 

d = d.rename(columns={'Flagpasopedido': 'Target'}) 



base = pd.merge(campaña,d,how='left', on=['IdConsultora','campana'] )





#------ LAS BASES PARA TRAIN, TEST y KTEST, DEFINIR CUAL ES LA CANTIDAD DE CAMPAÑAS A USAR

test = pd.merge(ids, base[["IdConsultora","Target","campana"]] ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



test = test[test.campana == 201905]

test["Target"].fillna(0, inplace=True)



test = pd.merge(ids, test[["IdConsultora","Target"]] ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

test["Target"].fillna(0, inplace=True)



del campaña



###############################################################################################

########################        BASE DE DATOS CAMPAÑA CONSULTORA      #########################

###############################################################################################  



campaña = pd.read_csv('/content/drive/My Drive/BELCORP/campana_consultora.csv')

test = pd.read_csv('/content/drive/My Drive/BELCORP/predict_submission.csv',  encoding='latin-1')



del campaña["Unnamed: 0"]





#----- Se obtiene los IDs a utilizar



#Id's para el train

ids = pd.DataFrame()

ids["IdConsultora"] = test.idconsultora





#----- DESFASE



def mesToNumero(codmes):

  anio = codmes//100

  mes = codmes%100

  return anio*18+mes

def numeroToMes(codmes):

  anio = codmes//18

  mes = codmes%18

  if(mes==0):

    anio-=1

    mes=18

  return anio*100+mes

def restarMes(codmes):

  return numeroToMes(mesToNumero(codmes)-1)





d = campaña[["IdConsultora","campana","Flagpasopedido"]]

d['campana_1'] = d['campana'].apply(restarMes)

del d['campana']

d = d.rename(columns={'campana_1': 'campana'}) 

d = d.rename(columns={'Flagpasopedido': 'Target'}) 



y_test = campaña[["IdConsultora","campana","Flagpasopedido"]][campaña.campana == 201906]

y_test = y_test.rename(columns={'Flagpasopedido': 'Target'}) 

y_test = pd.merge(ids, y_test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])





base = pd.merge(campaña,d,how='left', on=['IdConsultora','campana'] )





#------ LAS BASES PARA TRAIN, TEST y KTEST, DEFINIR CUAL ES LA CANTIDAD DE CAMPAÑAS A USAR

train = pd.merge(ids, base ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

test = pd.merge(ids, base ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

k_test = pd.merge(ids, base ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



train = base[(base.campana <= 201904)] #18 meses



train["Target"].fillna(0, inplace=True)



test = test[test.campana == 201905]

test["Target"].fillna(0, inplace=True)



k_test = k_test[k_test.campana == 201906]

del k_test["Target"]





#----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS



vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['IdConsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.9:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)





#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)    

k_test = k_test.drop(vars_miss, axis=1)    



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)    

k_test = k_test.drop(monotonas, axis=1)    





#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

for i in range(len(discretas)):

  a = discretas[i]

  n = num_discr[i] 

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):

          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1



          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1



          k_test[a + '_' + str(b[j])] = 0

          k_test[a + '_' + str(b[j])][(k_test[a] == b[j]) == True] = 1



      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)

      k_test.drop([a], axis='columns', inplace=True)



      

tr_dat = pd.merge(ids, train ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

te_dat = pd.merge(ids, test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

k_dat = pd.merge(ids, k_test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



tr_dat["Target"].fillna(0, inplace=True)

te_dat["Target"].fillna(0, inplace=True)



#Pequeño arreglo

te_dat["Target"] = y_test["Target"]

te_dat["Target"].fillna(0, inplace=True)



del d, base , campaña



###############################################################################################

#######################        BASE DE DATOS MAESTRO CONSULTORA        ########################

###############################################################################################  



consultora = pd.read_csv('/content/drive/My Drive/BELCORP/maestro_consultora.csv',  encoding='latin-1')

del consultora["Unnamed: 0"]



train = pd.merge(ids[['IdConsultora']], consultora ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

test = pd.merge(ids[['IdConsultora']], consultora ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

k_test = pd.merge(ids[['IdConsultora']], consultora ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])





#----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS



vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['IdConsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.9:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)



#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)    

k_test = k_test.drop(vars_miss, axis=1)    



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)    

k_test = k_test.drop(monotonas, axis=1)    





#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

for i in range(len(discretas)):

  a = discretas[i]

  n = num_discr[i] 

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):

        

          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1



          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1



          k_test[a + '_' + str(b[j])] = 0

          k_test[a + '_' + str(b[j])][(k_test[a] == b[j]) == True] = 1

            

      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)

      k_test.drop([a], axis='columns', inplace=True)



      

tr_dat = pd.merge(tr_dat, train ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

te_dat = pd.merge(te_dat, test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

k_dat = pd.merge(k_dat, k_test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



tr_dat["Dif_ultimopedido"] = tr_dat["campana"] - tr_dat["campanaultimopedido"]

tr_dat["Dif_primerpedido"] = tr_dat["campana"] - tr_dat["campanaprimerpedido"]

tr_dat["Dif_ingreso"] = tr_dat["campana"] - tr_dat["campanaingreso"]



te_dat["Dif_ultimopedido"] = te_dat["campana"] - te_dat["campanaultimopedido"]

te_dat["Dif_primerpedido"] = te_dat["campana"] - te_dat["campanaprimerpedido"]

te_dat["Dif_ingreso"] = te_dat["campana"] - te_dat["campanaingreso"]



k_dat["Dif_ultimopedido"] = k_dat["campana"] - k_dat["campanaultimopedido"]

k_dat["Dif_primerpedido"] = k_dat["campana"] - k_dat["campanaprimerpedido"]

k_dat["Dif_ingreso"] = k_dat["campana"] - k_dat["campanaingreso"]





del tr_dat["campanaultimopedido"]

del tr_dat["campanaprimerpedido"]

del tr_dat["campanaingreso"]



del te_dat["campanaultimopedido"]

del te_dat["campanaprimerpedido"]

del te_dat["campanaingreso"]



del k_dat["campanaultimopedido"]

del k_dat["campanaprimerpedido"]

del k_dat["campanaingreso"]



del train, test, k_test , consultora



###############################################################################################

#######################        BASE DE DATOS CAMPAÑA        ########################

###############################################################################################  



producto = pd.read_csv('/content/drive/My Drive/BELCORP/maestro_producto.csv',  encoding='latin-1')

detallecamp = pd.read_csv('/content/drive/My Drive/BELCORP/dtt_fvta_cl.csv',  encoding='latin-1')





#------ UNION DE BASE PRODUCTO Y MAESTRO PRODUCTO



n = int(tr_dat["campana"].min())

detallecamp = detallecamp[detallecamp.campana >= n]



train = pd.merge(ids[["IdConsultora"]], detallecamp ,how='left', left_on=['IdConsultora'], right_on=['idconsultora'])

test = pd.merge(ids[["IdConsultora"]], detallecamp ,how='left', left_on=['IdConsultora'], right_on=['idconsultora'])

k_test = pd.merge(ids[["IdConsultora"]], detallecamp ,how='left', left_on=['IdConsultora'], right_on=['idconsultora'])





train = pd.merge(train, producto[["idproducto","unidadnegocio","marca","categoria"]] ,

                how='left', left_on=['idproducto'], right_on=['idproducto'])



test = pd.merge(test, producto[["idproducto","unidadnegocio","marca","categoria"]] ,

                how='left', left_on=['idproducto'], right_on=['idproducto'])



k_test = pd.merge(k_test, producto[["idproducto","unidadnegocio","marca","categoria"]] ,

                how='left', left_on=['idproducto'], right_on=['idproducto'])





#------ DEFINIR CUAL ES LA CANTIDAD DE CAMPAÑAS A USAR



train = train[(train.campana >= n) & (train.campana <= 201904)] #6 meses



test = test[test.campana == 201905]



k_test = k_test[k_test.campana == 201906]





##----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS



del train["palancapersonalizacion"]

del test["palancapersonalizacion"]

del k_test["palancapersonalizacion"]

del train["idconsultora"]

del test["idconsultora"]

del k_test["idconsultora"]



vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['Idconsultora','idconsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.8:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)





#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)    

k_test = k_test.drop(vars_miss, axis=1)    



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)    

k_test = k_test.drop(monotonas, axis=1)    





#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

vec = []

for i in range(len(discretas)):

  a = discretas[i]

  n = num_discr[i] 

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):



          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1



          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1

          

          k_test[a + '_' + str(b[j])] = 0

          k_test[a + '_' + str(b[j])][(k_test[a] == b[j]) == True] = 1

        

          vec.append(a + '_' + str(b[j]))  

      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)

      k_test.drop([a], axis='columns', inplace=True)



del train["idproducto"]

del test["idproducto"]

del k_test["idproducto"]



c=['count','nunique']

n=['mean','max','min','sum','std']

n1=["sum"]

nn=['mean','max','min','sum','std','quantile']



agg_c={'codigotipooferta':c,'codigopalancapersonalizacion':c,

       

       'descuento':n,'ahorro':n,'preciocatalogo':n,'realuuvendidas':n,

       

       'realanulmnneto':n1,

       'realdevmnneto':n1,'realuuanuladas':n1,"realuudevueltas":n1,"realuufaltantes":n1,

       'realvtamnfaltneto':n1,'realvtamnfaltcatalogo':n1, 'realuuvendidas':n1,       

       'realvtamnneto':nn,'realvtamncatalogo':nn}

       

boxes = {

    i : n1        

    for i in vec    

}    



agg_c.update(boxes)  





train = train.groupby(['IdConsultora','campana']).agg(agg_c)

train.columns=['F_' + '_'.join(col).strip() for col in train.columns.values]

train.reset_index(inplace=True)



test = test.groupby(['IdConsultora','campana']).agg(agg_c)

test.columns=['F_' + '_'.join(col).strip() for col in test.columns.values]

test.reset_index(inplace=True)



k_test = k_test.groupby(['IdConsultora','campana']).agg(agg_c)

k_test.columns=['F_' + '_'.join(col).strip() for col in k_test.columns.values]

k_test.reset_index(inplace=True)





tr_dat = pd.merge(tr_dat, train ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

te_dat = pd.merge(te_dat, test ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

k_dat = pd.merge(k_dat, k_test ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])



del train, test, k_test , detallecamp, producto, ids, vec



#tr_dat.to_csv('/content/drive/My Drive/BELCORP/historico_train.csv')

#te_dat.to_csv('/content/drive/My Drive/BELCORP/historico_test.csv')

#k_dat.to_csv('/content/drive/My Drive/BELCORP/historico_k_test.csv')



te_dat["Target"] = test["Target"]



te_dat.to_csv('/content/drive/My Drive/BELCORP/historico_test_1.csv')



###############################################################################################

################################        MODELO XGBOOSST       #################################

###############################################################################################  





x_train = tr_dat

y_train = x_train['Target']

x_train = x_train.drop(['Target'], axis=1)

x_train = x_train.drop(['IdConsultora'], axis=1)

x_train = x_train.drop(['campana'], axis=1)



x_test = te_dat

y_test = x_test['Target']

x_test = x_test.drop(['Target'], axis=1)

x_test = x_test.drop(['IdConsultora'], axis=1)

x_test = x_test.drop(['campana'], axis=1)

    



##GS1

xg_reg =  xgb.XGBClassifier(colsample_bytree = 0.7,

                gamma = 0,                 

                learning_rate = 0.1,

                max_depth = 6,

                min_child_weight = 1.5,

                n_estimators = 90, #Cantidad de arboles                                                                   

                n_thread = 3,

                n_jobs = 36, 

                reg_alpha = 0.5,

                reg_lambda = 5)     





xg_reg.fit(x_train.values,y_train.values)

preds = pd.DataFrame(xg_reg.predict_proba(x_test.values))

preds = pd.Series(preds[1])



preds_train = pd.DataFrame(xg_reg.predict_proba(x_train.values))

preds_train = pd.Series(preds_train[1])



from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, preds) 

roc_auc_score(y_train, preds_train)



roc_auc_score(y_test, preds)



x_test = k_dat

x_test = x_test.drop(['IdConsultora'], axis=1)

x_test = x_test.drop(['campana'], axis=1)



k_preds = pd.DataFrame(xg_reg.predict_proba(x_test.values))

k_preds = pd.Series(k_preds[1])



historico = pd.DataFrame()

historico["IdConsultora"] = te_dat["IdConsultora"]

historico["predsTrain"] = preds

historico["predsKagle"] = k_preds







##--- MODELO AGRUPADO LIGHGBM



# -*- coding: utf-8 -*-

"""Agrupado_LGBM.ipynb



Automatically generated by Colaboratory.



Original file is located at

    https://colab.research.google.com/drive/1Fl9C1MHfDpt0INaLOSuEu4crD4nNYxvR

"""



import pandas as pd

import pickle as pickle

from datetime import datetime

import numpy as np

import pkg_resources

from scipy.stats import uniform, randint

import matplotlib.pyplot as plt

import math

from math import log

import sklearn as sklearn

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.metrics import r2_score

from xgboost import plot_importance



pd.options.display.html.table_schema=True #Para ver con barra los dataframes



#----- DEFINIR PERIODOS

tr_p = 201807 

te_p = 201808



# tr_p = 201815 #9M 

# te_p = 201816



# tr_p = 201818 #6M

# te_p = 201901 



# tr_p = 201903 #3M

# te_p = 201904



###############################################################################################

########################        BASE DE DATOS CAMPAÑA CONSULTORA      #########################

############################################################################################### 



k_test = pd.read_csv('/content/drive/My Drive/BELCORP/predict_submission.csv', sep=",")

campaña = pd.read_csv('/content/drive/My Drive/BELCORP/campana_consultora.csv')



del campaña["Unnamed: 0"]



#Id's para el train, SIN TODOS EN TRAIN

# ids = pd.DataFrame()

# ids["IdConsultora"] = k_test.idconsultora

# tr_ids = campaña[["IdConsultora","Flagpasopedido"]][campaña.campana == 201906]

# tr_ids = tr_ids.rename(columns={'Flagpasopedido':'Target'})



#Id's para el train, CON TODOS EN TRAIN

ids = pd.DataFrame()

ids["IdConsultora"] = k_test.idconsultora

tr_ids = campaña[["IdConsultora","Flagpasopedido"]][campaña.campana == 201906]

tr_ids = tr_ids.rename(columns={'Flagpasopedido':'Target'})

tr_ids = pd.merge(ids, tr_ids ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

tr_ids["Target"].fillna(0, inplace=True)



#----- TRAIN Y TEST

train = pd.merge(tr_ids, campaña ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

test = pd.merge(ids, campaña ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

train = train[(train.campana >= int(tr_p)) & (train.campana <= 201905)] #filas: 1,052,514 - columnas: 22

test = test[(test.campana >= int(te_p)) & (test.campana <= 201906)] #filas: 1,416,970 - columnas: 21



del campaña



#----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS



vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['IdConsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.9:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)

      

#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)    



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)



vec_cat = []

#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

for i in range(len(discretas)):

  a = discretas[i]

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):

          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1

          

          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1

          

          vec_cat.append(a + '_' + str(b[j]))  

      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)



#----- CREACION DE RATIOS

#---- PARA TRAIN

#-- cantidadlogueos

rt = train[["IdConsultora", 'campana','cantidadlogueos']].copy()

rt.sort_values(['IdConsultora', 'campana'], inplace=True)

rt = rt.reset_index(drop=True)



for col in rt.columns:

    if(col in ["IdConsultora","campana"]):

        continue

    rt[col+'_0'] = rt.groupby(['IdConsultora'])[col].shift(0)

    rt[col+'_1'] = rt.groupby(['IdConsultora'])[col].shift(-1)

    rt[col+'_2'] = rt.groupby(['IdConsultora'])[col].shift(-2)

    rt[col+'_3'] = rt.groupby(['IdConsultora'])[col].shift(-3)

    rt[col+'_4'] = rt.groupby(['IdConsultora'])[col].shift(-4)

    rt[col+'_5'] = rt.groupby(['IdConsultora'])[col].shift(-5)

    rt[col+'_6'] = rt.groupby(['IdConsultora'])[col].shift(-6)

    rt[col+'_7'] = rt.groupby(['IdConsultora'])[col].shift(-7)

    rt[col+'_8'] = rt.groupby(['IdConsultora'])[col].shift(-8)

    rt[col+'_9'] = rt.groupby(['IdConsultora'])[col].shift(-9)

    rt[col+'_10'] = rt.groupby(['IdConsultora'])[col].shift(-10)

    rt[col+'_11'] = rt.groupby(['IdConsultora'])[col].shift(-11)



columnas = rt.columns

for col in columnas:

    if(col[-2:] in ["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11"]):

      continue

    if(col[-3:] in ["_10","_11"]):

      continue 

    if(col in ["IdConsultora","campana"]):

      continue

    rt[col+'_total12'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']+rt[col+'_9']+rt[col+'_10']+rt[col+'_11']

    rt[col+'_total9'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']

    rt[col+'_total6'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']

    rt[col+'_total3'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2']



    rt[col+'_ratio12'] = rt[col+'_0'] / rt[col+'_total12'] 

    rt[col+'_ratio9'] = rt[col+'_0'] / rt[col+'_total9'] 

    rt[col+'_ratio6'] = rt[col+'_0'] / rt[col+'_total6'] 

    rt[col+'_ratio3'] = rt[col+'_0'] / rt[col+'_total3'] 



    rt[col+'_ratio3_6'] = rt[col+'_total3'] / rt[col+'_total6']

    rt[col+'_ratio3_9'] = rt[col+'_total3'] / rt[col+'_total9']

    rt[col+'_ratio3_12'] = rt[col+'_total3'] / rt[col+'_total12']  



    del rt[col+'_0']

    del rt[col+'_1']

    del rt[col+'_2']

    del rt[col+'_3']

    del rt[col+'_4']

    del rt[col+'_5']

    del rt[col+'_6']

    del rt[col+'_7']

    del rt[col+'_8']

    del rt[col+'_9']

    del rt[col+'_10'] 

    del rt[col+'_11']

    del rt[col]

    

rt = rt.fillna(0)    

train = pd.merge(train, rt ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

vec_ratios = [x for x in rt.columns if not x in ['IdConsultora','campana','cantidadlogueos']]

del rt



#--- PARA TEST

rt = test[["IdConsultora", 'campana','cantidadlogueos']].copy()

rt.sort_values(['IdConsultora', 'campana'], inplace=True)

rt = rt.reset_index(drop=True)



for col in rt.columns:

    if(col in ["IdConsultora","campana"]):

        continue

    rt[col+'_0'] = rt.groupby(['IdConsultora'])[col].shift(0)

    rt[col+'_1'] = rt.groupby(['IdConsultora'])[col].shift(-1)

    rt[col+'_2'] = rt.groupby(['IdConsultora'])[col].shift(-2)

    rt[col+'_3'] = rt.groupby(['IdConsultora'])[col].shift(-3)

    rt[col+'_4'] = rt.groupby(['IdConsultora'])[col].shift(-4)

    rt[col+'_5'] = rt.groupby(['IdConsultora'])[col].shift(-5)

    rt[col+'_6'] = rt.groupby(['IdConsultora'])[col].shift(-6)

    rt[col+'_7'] = rt.groupby(['IdConsultora'])[col].shift(-7)

    rt[col+'_8'] = rt.groupby(['IdConsultora'])[col].shift(-8)

    rt[col+'_9'] = rt.groupby(['IdConsultora'])[col].shift(-9)

    rt[col+'_10'] = rt.groupby(['IdConsultora'])[col].shift(-10)

    rt[col+'_11'] = rt.groupby(['IdConsultora'])[col].shift(-11)



columnas = rt.columns

for col in columnas:

    if(col[-2:] in ["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11"]):

      continue

    if(col[-3:] in ["_10","_11"]):

      continue 

    if(col in ["IdConsultora","campana"]):

      continue

    rt[col+'_total12'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']+rt[col+'_9']+rt[col+'_10']+rt[col+'_11']

    rt[col+'_total9'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']

    rt[col+'_total6'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']

    rt[col+'_total3'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2']



    rt[col+'_ratio12'] = rt[col+'_0'] / rt[col+'_total12'] 

    rt[col+'_ratio9'] = rt[col+'_0'] / rt[col+'_total9'] 

    rt[col+'_ratio6'] = rt[col+'_0'] / rt[col+'_total6'] 

    rt[col+'_ratio3'] = rt[col+'_0'] / rt[col+'_total3'] 



    rt[col+'_ratio3_6'] = rt[col+'_total3'] / rt[col+'_total6']

    rt[col+'_ratio3_9'] = rt[col+'_total3'] / rt[col+'_total9']

    rt[col+'_ratio3_12'] = rt[col+'_total3'] / rt[col+'_total12']  



    del rt[col+'_0']

    del rt[col+'_1']

    del rt[col+'_2']

    del rt[col+'_3']

    del rt[col+'_4']

    del rt[col+'_5']

    del rt[col+'_6']

    del rt[col+'_7']

    del rt[col+'_8']

    del rt[col+'_9']

    del rt[col+'_10'] 

    del rt[col+'_11']

    del rt[col]

    

rt = rt.fillna(0)    

test = pd.merge(test, rt ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

del rt



#----- GENERAR FUNCION PARA CREAR ACUMULADO

c=['count','nunique']

n=['mean','max','min','sum','std']

n1=["sum"]

n2=['mean','sum','std']

n3=['max']

n4 = ['mean']

nn=['mean','max','min','sum','std','quantile']



#Para este caso no lo hago para variables CATEGORICAS , todas SON CONTINUAS!!!!!, tratamiento n

agg_c={'campana':n3,

    'Flagpasopedido':n2, 'flagactiva':n2, 'flagpasopedidocuidadopersonal':n2, 'flagpasopedidomaquillaje':n2, 

       'flagpasopedidotratamientocorporal':n2, 'flagpasopedidotratamientofacial':n2, 'flagpedidoanulado':n2, 

       'flagpasopedidofragancias':n2, 'flagpasopedidoweb':n2, 

       

       'cantidadlogueos':nn,

       

       #'flagdigital':n2,

       'flagdispositivo':n2, 'flagofertadigital':n2, 'flagsuscripcion':n2  

       

}





boxes = {

    i : n1        

    for i in vec_cat    

}  



agg_c.update(boxes)  



boxes = {

    i : n4        

    for i in vec_ratios    

}  



agg_c.update(boxes)



train = train.groupby(['IdConsultora']).agg(agg_c)

train.columns=['C_' + '_'.join(col).strip() for col in train.columns.values]

train.reset_index(inplace=True)



test = test.groupby(['IdConsultora']).agg(agg_c)

test.columns=['C_' + '_'.join(col).strip() for col in test.columns.values]

test.reset_index(inplace=True)



tr_ids = pd.merge(tr_ids, train ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

ids = pd.merge(ids, test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



del train, test



###############################################################################################

#######################        BASE DE DATOS MAESTRO CONSULTORA        ########################

###############################################################################################  



consultora = pd.read_csv('/content/drive/My Drive/BELCORP/maestro_consultora.csv',  encoding='latin-1')



del consultora["Unnamed: 0"]



train = pd.merge(tr_ids[['IdConsultora']], consultora ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

test = pd.merge(ids[['IdConsultora']], consultora ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



del consultora



def mesToNumero(codmes):

  anio = codmes//100

  mes = codmes%100

  return anio*18+mes

def numeroToMes(codmes):

  anio = codmes//18

  mes = codmes%18

  if(mes==0):

    anio-=1

    mes=18

  return anio*100+mes

def restarMes(codmes):

  return numeroToMes(mesToNumero(codmes)-1)





# d = campaña[["IdConsultora","campana","Flagpasopedido"]]

# d['campana_1'] = d['campana'].apply(restarMes)

# del d['campana']

# d = d.rename(columns={'campana_1': 'campana'}) 

# d = d.rename(columns={'Flagpasopedido': 'Target'})



#----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS

vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['IdConsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.9:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)



#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)     



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)



vec_cat = []

#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

for i in range(len(discretas)):

  a = discretas[i]

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):

          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1

          

          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1

          

          vec_cat.append(a + '_' + str(b[j]))  

      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)



tr_ids = pd.merge(tr_ids, train ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

ids = pd.merge(ids, test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



del train, test



tr_ids.keys()



tr_ids["Dif_ultimopedido"] = tr_ids["C_campana_max"].apply(mesToNumero) - tr_ids["campanaultimopedido"].apply(mesToNumero)

tr_ids["Dif_primerpedido"] = tr_ids["C_campana_max"].apply(mesToNumero) - tr_ids["campanaprimerpedido"].apply(mesToNumero)

tr_ids["Dif_ingreso"] = tr_ids["C_campana_max"].apply(mesToNumero) - tr_ids["campanaingreso"].apply(mesToNumero)



ids["Dif_ultimopedido"] = ids["C_campana_max"].apply(mesToNumero) - ids["campanaultimopedido"].apply(mesToNumero)

ids["Dif_primerpedido"] = ids["C_campana_max"].apply(mesToNumero) - ids["campanaprimerpedido"].apply(mesToNumero)

ids["Dif_ingreso"] = ids["C_campana_max"].apply(mesToNumero) - ids["campanaingreso"].apply(mesToNumero)



del tr_ids["C_campana_max"]

del tr_ids["campanaultimopedido"]

del tr_ids["campanaprimerpedido"]

del tr_ids["campanaingreso"]



del ids["C_campana_max"]

del ids["campanaultimopedido"]

del ids["campanaprimerpedido"]

del ids["campanaingreso"]



###############################################################################################

############################        BASE DE DATOS CAMPAÑA        ##############################

###############################################################################################  



producto = pd.read_csv('/content/drive/My Drive/BELCORP/maestro_producto.csv',  encoding='latin-1')

detallecamp = pd.read_csv('/content/drive/My Drive/BELCORP/dtt_fvta_cl.csv',  encoding='latin-1')



train = pd.merge(tr_ids[["IdConsultora"]], detallecamp ,how='left', left_on=['IdConsultora'], right_on=['idconsultora'])

test = pd.merge(ids[["IdConsultora"]], detallecamp ,how='left', left_on=['IdConsultora'], right_on=['idconsultora'])



del train["idconsultora"]

del test["idconsultora"]

del detallecamp



train = pd.merge(train, producto[["idproducto","unidadnegocio","marca","categoria"]] ,

                how='left', left_on=['idproducto'], right_on=['idproducto'])



test = pd.merge(test, producto[["idproducto","unidadnegocio","marca","categoria"]] ,

                how='left', left_on=['idproducto'], right_on=['idproducto'])



del producto



#----- TRAIN Y TEST

train = train[(train.campana >= int(tr_p)) & (train.campana <= 201905)] #filas: 1,052,514 - columnas: 22

test = test[(test.campana >= int(te_p)) & (test.campana <= 201906)] #filas: 1,416,970 - columnas: 21



#----- DROPEAR VARIABLES MONÓTONAS, DROPEAR VARIABLES MISSING, CLASIFICAR VARIABLES CONTINUAS Y DISCRETAS

vars_miss = []

monotonas = []

discretas = []

num_discr = []

continuas = []



for i in train.columns:  

  if not i in ['IdConsultora','Target','campana']:

    if (len(train[pd.isnull(train[i]) == True])/len(train)) >= 0.9:

      vars_miss.append(i)

    elif len(train[i].unique()) == 1:

      monotonas.append(i)  

    elif (len(train[i].unique()) > 1) & (len(train[i].unique()) < 50 ): 

      discretas.append(i)

      num_discr.append(len(train[i].unique()))

    elif len(train[i].unique()) >= 50: 

      continuas.append(i)



#---- DROPEAR VARIABLES CON MÄS DE 80% DE MISSINGS

train = train.drop(vars_miss, axis=1)    

test = test.drop(vars_miss, axis=1)     



#---- DROPEAR VARIABLES MONÓTONAS

train = train.drop(monotonas, axis=1)    

test = test.drop(monotonas, axis=1)



#len(train.canalingresoproducto.unique())#6 dummies

#len(train.grupooferta.unique())#5 dummies

#len(train.unidadnegocio.unique())#5 dummies

#len(train.marca.unique())#4 dummies

#len(train.categoria.unique())#19 Crear (count.unique) 

#len(train.palancapersonalizacion.unique())#271 Crear (count.unique)



#----- TRANSFORMACIÓN DE VARIABLES CATEGORICAS

discretas = ['descuento','canalingresoproducto','grupooferta','realuuanuladas','realuudevueltas','realuufaltantes',

             'unidadnegocio','marca']

vec_cat = []

for i in range(len(discretas)):

  a = discretas[i]

  n = num_discr[i] 

  t = type((train[a][pd.isnull(train[a]) == False].reset_index(drop=True))[0])

  

  if (t is str):

      b = train[a].unique()   

      for j in range(len(b)-1):



          train[a + '_' + str(b[j])] = 0

          train[a + '_' + str(b[j])][(train[a] == b[j]) == True] = 1



          test[a + '_' + str(b[j])] = 0

          test[a + '_' + str(b[j])][(test[a] == b[j]) == True] = 1

       

          vec_cat.append(a + '_' + str(b[j]))  

      train.drop([a], axis='columns', inplace=True)

      test.drop([a], axis='columns', inplace=True)



#len(train.canalingresoproducto.unique())#6 dummies

#len(train.grupooferta.unique())#5 dummies

#len(train.unidadnegocio.unique())#5 dummies

#len(train.marca.unique())#4 dummies

#len(train.categoria.unique())#19 Crear (count.unique) 

#len(train.palancapersonalizacion.unique())#271 Crear (count.unique)



#----- GENERAR FUNCION PARA CREAR ACUMULADO MENSUAL

c=['count','nunique']

n=['mean','max','min','sum','std']

n1=["sum"]

n2=['mean','sum','std']

n3=['max']

n4=['mean']

n5=['mean','count']

nn=['mean','max','min','sum','std','quantile']



#Para este caso no lo hago para variables CATEGORICAS , todas SON CONTINUAS!!!!!, tratamiento n

agg_c={'codigotipooferta':c,

  

    'descuento':n5, 'ahorro':n5,

     

    'idproducto':c, 'codigopalancapersonalizacion':c, 'palancapersonalizacion':c,

    

     'preciocatalogo':n5, 'realanulmnneto':n5, 'realdevmnneto':n5, 'realuuanuladas':n5, 'realuudevueltas':n5,

     'realuufaltantes':n5, 'realuuvendidas':n5, 'realvtamnfaltneto':n5, 'realvtamnneto':n5,

     'realvtamncatalogo':n5, 'realvtamnfaltcatalogo':n5, 

       

     'categoria':c,            

}





boxes = {

    i : n1        

    for i in vec_cat    

}  



agg_c.update(boxes)  





agg_c.update(boxes)



train = train.groupby(['IdConsultora','campana']).agg(agg_c)

train.columns=['P_' + '_'.join(col).strip() for col in train.columns.values]

train.reset_index(inplace=True)



test = test.groupby(['IdConsultora','campana']).agg(agg_c)

test.columns=['P_' + '_'.join(col).strip() for col in test.columns.values]

test.reset_index(inplace=True)



#----- CREACION DE RATIOS

#---- PARA TRAIN



#'descuento', 'ahorro','realuuvendidas','realvtamnneto','realuuanuladas'

rt = train[["IdConsultora", 'campana','P_descuento_mean','P_ahorro_mean','P_realuuvendidas_mean',

            'P_realvtamnneto_mean','P_realuuanuladas_mean']].copy()



rt.sort_values(['IdConsultora', 'campana'], inplace=True)

rt = rt.reset_index(drop=True)



for col in rt.columns:

    if(col in ["IdConsultora","campana"]):

        continue

    rt[col+'_0'] = rt.groupby(['IdConsultora'])[col].shift(0)

    rt[col+'_1'] = rt.groupby(['IdConsultora'])[col].shift(-1)

    rt[col+'_2'] = rt.groupby(['IdConsultora'])[col].shift(-2)

    rt[col+'_3'] = rt.groupby(['IdConsultora'])[col].shift(-3)

    rt[col+'_4'] = rt.groupby(['IdConsultora'])[col].shift(-4)

    rt[col+'_5'] = rt.groupby(['IdConsultora'])[col].shift(-5)

    rt[col+'_6'] = rt.groupby(['IdConsultora'])[col].shift(-6)

    rt[col+'_7'] = rt.groupby(['IdConsultora'])[col].shift(-7)

    rt[col+'_8'] = rt.groupby(['IdConsultora'])[col].shift(-8)

    rt[col+'_9'] = rt.groupby(['IdConsultora'])[col].shift(-9)

    rt[col+'_10'] = rt.groupby(['IdConsultora'])[col].shift(-10)

    rt[col+'_11'] = rt.groupby(['IdConsultora'])[col].shift(-11)



columnas = rt.columns

for col in columnas:

    if(col[-2:] in ["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11"]):

      continue

    if(col[-3:] in ["_10","_11"]):

      continue 

    if(col in ["IdConsultora","campana"]):

      continue

    rt[col+'_total12'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']+rt[col+'_9']+rt[col+'_10']+rt[col+'_11']

    rt[col+'_total9'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']

    rt[col+'_total6'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']

    rt[col+'_total3'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2']



    rt[col+'_ratio12'] = rt[col+'_0'] / rt[col+'_total12'] 

    rt[col+'_ratio9'] = rt[col+'_0'] / rt[col+'_total9'] 

    rt[col+'_ratio6'] = rt[col+'_0'] / rt[col+'_total6'] 

    rt[col+'_ratio3'] = rt[col+'_0'] / rt[col+'_total3'] 



    rt[col+'_ratio3_6'] = rt[col+'_total3'] / rt[col+'_total6']

    rt[col+'_ratio3_9'] = rt[col+'_total3'] / rt[col+'_total9']

    rt[col+'_ratio3_12'] = rt[col+'_total3'] / rt[col+'_total12']  



    del rt[col+'_0']

    del rt[col+'_1']

    del rt[col+'_2']

    del rt[col+'_3']

    del rt[col+'_4']

    del rt[col+'_5']

    del rt[col+'_6']

    del rt[col+'_7']

    del rt[col+'_8']

    del rt[col+'_9']

    del rt[col+'_10'] 

    del rt[col+'_11']

    del rt[col]

    

rt = rt.fillna(0)    

train = pd.merge(train, rt ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

vec_ratios = [x for x in rt.columns if not x in ['IdConsultora','campana']]

del rt



#--- PARA TEST

#'descuento', 'ahorro','realuuvendidas','realvtamnneto','realuuanuladas'

rt = test[["IdConsultora", 'campana','P_descuento_mean','P_ahorro_mean','P_realuuvendidas_mean',

            'P_realvtamnneto_mean','P_realuuanuladas_mean']].copy()

  

rt.sort_values(['IdConsultora', 'campana'], inplace=True)

rt = rt.reset_index(drop=True)



for col in rt.columns:

    if(col in ["IdConsultora","campana"]):

        continue

    rt[col+'_0'] = rt.groupby(['IdConsultora'])[col].shift(0)

    rt[col+'_1'] = rt.groupby(['IdConsultora'])[col].shift(-1)

    rt[col+'_2'] = rt.groupby(['IdConsultora'])[col].shift(-2)

    rt[col+'_3'] = rt.groupby(['IdConsultora'])[col].shift(-3)

    rt[col+'_4'] = rt.groupby(['IdConsultora'])[col].shift(-4)

    rt[col+'_5'] = rt.groupby(['IdConsultora'])[col].shift(-5)

    rt[col+'_6'] = rt.groupby(['IdConsultora'])[col].shift(-6)

    rt[col+'_7'] = rt.groupby(['IdConsultora'])[col].shift(-7)

    rt[col+'_8'] = rt.groupby(['IdConsultora'])[col].shift(-8)

    rt[col+'_9'] = rt.groupby(['IdConsultora'])[col].shift(-9)

    rt[col+'_10'] = rt.groupby(['IdConsultora'])[col].shift(-10)

    rt[col+'_11'] = rt.groupby(['IdConsultora'])[col].shift(-11)



columnas = rt.columns

for col in columnas:

    if(col[-2:] in ["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9","_10","_11"]):

      continue

    if(col[-3:] in ["_10","_11"]):

      continue 

    if(col in ["IdConsultora","campana"]):

      continue

    rt[col+'_total12'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']+rt[col+'_9']+rt[col+'_10']+rt[col+'_11']

    rt[col+'_total9'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']+rt[col+'_6']+rt[col+'_7']+rt[col+'_8']

    rt[col+'_total6'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2'] + rt[col+'_3']+ rt[col+'_4'] +rt[col+'_5']

    rt[col+'_total3'] = rt[col+'_0']+ rt[col+'_1'] +rt[col+'_2']



    rt[col+'_ratio12'] = rt[col+'_0'] / rt[col+'_total12'] 

    rt[col+'_ratio9'] = rt[col+'_0'] / rt[col+'_total9'] 

    rt[col+'_ratio6'] = rt[col+'_0'] / rt[col+'_total6'] 

    rt[col+'_ratio3'] = rt[col+'_0'] / rt[col+'_total3'] 



    rt[col+'_ratio3_6'] = rt[col+'_total3'] / rt[col+'_total6']

    rt[col+'_ratio3_9'] = rt[col+'_total3'] / rt[col+'_total9']

    rt[col+'_ratio3_12'] = rt[col+'_total3'] / rt[col+'_total12']  



    del rt[col+'_0']

    del rt[col+'_1']

    del rt[col+'_2']

    del rt[col+'_3']

    del rt[col+'_4']

    del rt[col+'_5']

    del rt[col+'_6']

    del rt[col+'_7']

    del rt[col+'_8']

    del rt[col+'_9']

    del rt[col+'_10'] 

    del rt[col+'_11']

    del rt[col]

    

rt = rt.fillna(0)    

test = pd.merge(test, rt ,how='left', left_on=['IdConsultora','campana'], right_on=['IdConsultora','campana'])

del rt



no_vec = vec_ratios.copy()

no_vec.append('IdConsultora')

no_vec.append('campana')

vec = [x for x in train.columns if not x in no_vec]



#----- GENERAR FUNCION PARA CREAR ACUMULADO PARA UN SOLO IDCONSULTORA

c=['count','nunique']

n=['mean','max','min','sum','std']

n1=["sum"]

n2=['mean','sum','std']

n3=['max']

m4=['mean']

nn=['mean','max','min','sum','std','quantile']



#Para este caso no lo hago para variables CATEGORICAS , todas SON CONTINUAS!!!!!, tratamiento n

agg_c={            

}



boxes = {

    i : n4        

    for i in vec    

}

agg_c.update(boxes) 



boxes = {

    i : nn        

    for i in vec_ratios    

}  

agg_c.update(boxes)



train = train.groupby(['IdConsultora']).agg(agg_c)

train.columns=['P_' + '_'.join(col).strip() for col in train.columns.values]

train.reset_index(inplace=True)



test = test.groupby(['IdConsultora']).agg(agg_c)

test.columns=['P_' + '_'.join(col).strip() for col in test.columns.values]

test.reset_index(inplace=True)



tr_ids = pd.merge(tr_ids, train ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])

ids = pd.merge(ids, test ,how='left', left_on=['IdConsultora'], right_on=['IdConsultora'])



del train, test



tr_ids.to_csv('/content/drive/My Drive/BELCORP/agrupado_train.csv')

ids.to_csv('/content/drive/My Drive/BELCORP/agrupado_test.csv')



###############################################################################################

##################################        MODELO        #######################################

###############################################################################################  



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



train = tr_ids

test = ids

#train.fillna(0, inplace=True)



features=[ x for x in train.columns if x not in ['IdConsultora','Target']]

target='Target'



kf_previo=StratifiedKFold(n_splits=5,random_state=256,shuffle=True)



i=1



r=[]



importancias=pd.DataFrame()



importancias['variable']=features



for train_index,test_index in kf_previo.split(train,train[target]):



    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,target].values.ravel())

    lgb_eval = lgb.Dataset(train.loc[test_index,features].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)



    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'auc'},

        "max_depth":6,

        "num_leaves":10,

        'learning_rate': 0.1,

    "min_child_samples": 100,

        'min_child_weight': 1.5,

        'feature_fraction': 0.5,

     "bagging_freq":1,

        'bagging_fraction': 0.9,

        "lambda_l1":0.5,

        "lambda_l2":5,

       # "scale_pos_weight":30,



        'verbose': 1    

    }

    



    lgbm3 = lgb.train(params,lgb_train,num_boost_round=1310,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25)

    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)

    train["TARGET_FOLD"+str(i)]=lgbm3.predict(train[features].values, num_iteration=lgbm3.best_iteration)

 

    importancias['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")



    

    print ("Fold_"+str(i))

    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration)))

    r.append(a)

    print (a)

    print ("")

    

    i=i+1



print ("mean: "+str(np.mean(np.array(r))))

print ("std: "+str(np.std(np.array(r))))





w=[x for x in test.columns if 'FOLD' in x]

test['flagpasopedido']=test[w].mean(axis=1)



# test[['IdConsultora','flagpasopedido']].to_csv('Submit/submit7.csv',index=False)



train['flagpasopedido']=train[w].mean(axis=1)



agrupado = pd.DataFrame()

agrupado["IdConsultora"] = train["IdConsultora"]

agrupado["predsTrain"] = train["flagpasopedido"]

agrupado["predsKagle"] = test["flagpasopedido"]







###############################################################################################

##################################       ENSAMBLE       #######################################

###############################################################################################  



train = agrupado[["IdConsultora"]]

train["preds_agrupado"] = agrupado[["predsTrain"]]

train["preds_historico"] = historico[["predsTrain"]]

train["Target"] = y_test["Target"]



test = agrupado[["IdConsultora"]]

test["preds_agrupado"] = agrupado[["predsKagle"]]

test["preds_historico"] = historico[["predsKagle"]]



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score





train = train

test = test

#train.fillna(0, inplace=True)



features=[ x for x in train.columns if x not in ['IdConsultora','Target']]

target='Target'





#--- MODELAMIENTO CON 5 FOLDS



kf_previo=StratifiedKFold(n_splits=5,random_state=256,shuffle=True)



i=1



r=[]



importancias=pd.DataFrame()



importancias['variable']=features



for train_index,test_index in kf_previo.split(train,train[target]):



    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,target].values.ravel())

    lgb_eval = lgb.Dataset(train.loc[test_index,features].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)



    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'auc'},

        "max_depth":6,

        "num_leaves":10,

        'learning_rate': 0.1,

    "min_child_samples": 100,

        'min_child_weight': 1.5,

        'feature_fraction': 0.5,

     "bagging_freq":1,

        'bagging_fraction': 0.9,

        "lambda_l1":0.5,

        "lambda_l2":5,

       # "scale_pos_weight":30,



        'verbose': 1    

    }

    

    params = {

        'task': 'train',

        'boosting_type': 'gbdt',

        'objective': 'binary',

        'metric': { 'auc'},

        "max_depth":6,

        "num_leaves":10,

        'learning_rate': 0.1,

    "min_child_samples": 100,

        'min_child_weight': 1.5,

        'feature_fraction': 0.5,

     "bagging_freq":1,

        'bagging_fraction': 0.9,

        "lambda_l1":5,

        "lambda_l2":5,

       # "scale_pos_weight":30,



        'verbose': 1    

    }

    lgbm3 = lgb.train(params,lgb_train,num_boost_round=1310,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25)

    test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)

 

    importancias['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")



    

    print ("Fold_"+str(i))

    a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration)))

    r.append(a)

    print (a)

    print ("")

    

    i=i+1



print ("mean: "+str(np.mean(np.array(r))))

print ("std: "+str(np.std(np.array(r))))





w=[x for x in test.columns if 'FOLD' in x]

test['flagpasopedido']=test[w].mean(axis=1)



test[['IdConsultora','flagpasopedido']].to_csv('/content/drive/My Drive/BELCORP/Submit/submit12.csv',index=False)


