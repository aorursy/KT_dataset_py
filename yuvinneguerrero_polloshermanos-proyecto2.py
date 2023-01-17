#Importacion de librerias

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import statistics
import seaborn as sns
import sklearn as skl
import scipy.stats as st
from sklearn.model_selection import cross_validate #importo libreria para cross validation
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#importacion de datasets
Original_Train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
Original_Test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
Original_Train.info() #exploracion inicial
Original_Test.info() #exploracion inicial
Original_Train['ExterQual'].value_counts()
exterqual_nums={'ExterQual': {"Ex":5, "Gd":4,'TA':3,'Fa':2}}
Original_Train.replace(exterqual_nums,inplace=True)
Original_Test['ExterQual'].value_counts()
exterqual_nums={'ExterQual': {"Ex":5, "Gd":4,'TA':3,'Fa':2}}
Original_Test.replace(exterqual_nums,inplace=True)
data = [Original_Train, Original_Test] #creacion de gurpo para el loop
for dataset in data:
    dataset['BaseArea'] = dataset['GrLivArea'] + dataset['GarageArea'] #creacion de variable relatives
    dataset['Qual_Time'] = (dataset['YearBuilt']-dataset['YearRemodAdd']) 
    dataset['Exqual_YearB'] = (dataset['YearBuilt']*dataset['ExterQual'])
Original_Train.head() #visualizacion
# establecer grupos de variables cuantitatvas y cualitativas
quantitative_ori = [f for f in Original_Train.columns if Original_Train.dtypes[f] != 'object']
#remover saleprice y id de las variable cualitativas
quantitative_ori.remove('SalePrice')
quantitative_ori.remove('Id')
qualitative_ori = [f for f in Original_Train.columns if Original_Train.dtypes[f] == 'object']
# distribución de las variables caulitativas
#Original_melt= pd.melt(Original_Train, value_vars=quantitative_ori) # melt el data set original para obtener las variables y sus valores
#graph = sns.FacetGrid(Original_melt, col="variable",  col_wrap=2, sharex=False, sharey=False) #establecer los ejes del grafico
#graph = graph.map(sns.distplot, "value") #impirmir el graph y mapear las distribuciones de cada valor

#referencia: https://www.kaggle.com/dgawlik/house-prices-eda
cova=Original_Train.cov()
cova['ExterQual'].head()
corr=Original_Train.corr()
corr_extq=corr['ExterQual'].sort_values(ascending=False)
corr_extq.head()
saleprice_corr=pd.DataFrame(corr['SalePrice'])
saleprice_corr.sort_values('SalePrice',ascending=False, inplace=True)
saleprice_corr.reset_index(inplace=True)
saleprice_corr.rename(columns={'index':'Variable'}, inplace=True)
saleprice_corr.head(10)
s=corr.abs().unstack()
so =pd.DataFrame(s.sort_values())
so
fig, ax = plt.subplots()
fig.set_size_inches(14, 7)
ax.set_title('Features correlation', )
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, vmin=-1, vmax=1, annot=False, fmt=".2f", cmap='RdBu', center=0, ax=ax)
Original_Esta=Original_Test.describe()
Original_Esta
#encontrar el total de valores perdidos de cada variable
train_total = Original_Train.isnull().sum().sort_values(ascending=False)
#expresar el porcentaje de valores perdidos por variable
train_percent_1 = Original_Train.isnull().sum()/Original_Train.isnull().count()*100
#redondear los decimales del porcentaje
train_percent_2 = (round(train_percent_1, 1)).sort_values(ascending=False)
#crear un nuevo dataframe que muestre el total y porcentaje
train_missing_data = pd.DataFrame([train_total, train_percent_2], index=["Total",'Porcentaje']).T
train_missing_data #visualizacion
missing_train = Original_Train.isnull().sum()
missing_train = missing_train[missing_train > 0]
missing_train.sort_values(inplace=True)
missing_train.plot.bar()
#plt.set_title('Missing Values')
#encontrar el total de valores perdidos de cada variable para el set de test
Test_total = Original_Test.isnull().sum().sort_values(ascending=False)
#expresar el porcentaje de valores perdidos por variable
Test_percent_1 = Original_Test.isnull().sum()/Original_Test.isnull().count()*100
#redondear los decimales del porcentaje
Test_percent_2 = (round(Test_percent_1, 1)).sort_values(ascending=False)
#crear un nuevo dataframe que muestre el total y porcentaje
Test_missing_data = pd.DataFrame([Test_total, Test_percent_2], index=["Total",'Porcentaje']).T
Test_missing_data.head(10) #visualizacion
missing_test = Original_Test.isnull().sum()
missing_test = missing_test[missing_test > 0]
missing_test.sort_values(inplace=True)
missing_test.plot.bar()
#plt.set_title('Missing Values')

# crear un nuevo dataframe de set de datos editados
Train_Edited=pd.DataFrame(Original_Train.copy())
Test_Edited=pd.DataFrame(Original_Test.copy())
Test_Edited['BsmtQual'].unique()
data = [Train_Edited, Test_Edited] #creacion de gurpo para el loop
for dataset in data:
    pool_qual = {"Ex": 4, "Gd": 3, "TA":2, "Fa":1} #crear un diccionario de correspondencia
    dataset['PoolQC'] = dataset['PoolQC'].map(pool_qual)  #cambiar los valores origniales con los del dic
    dataset['PoolQC']=dataset['PoolQC'].fillna(0) #cambiar los valores nulos por 0
    dataset['PoolQC']=dataset['PoolQC'].astype(int)
Test_Edited['PoolQC'].isnull().sum() #contar la cantidad de valores nulos
Test_Edited['PoolQC'].unique()
data = [Train_Edited, Test_Edited] #creacion de gurpo para el loop
for dataset in data:
    fence_qual = {"GdPrv": 4, "MnPrv": 3, "GdWo":2, "MnWw":1} #crear un diccionario de correspondencia
    dataset['Fence'] = dataset['Fence'].map(fence_qual)  #cambiar los valores origniales con los del dic
    dataset['Fence']=dataset['Fence'].fillna(0) #cambiar los valores nulos por 0
    dataset['Fence']=dataset['Fence'].astype(int)
Test_Edited['Fence'].isnull().sum() #contar la cantidad de valores nulos
Test_Edited['Fence'].unique()
data = [Train_Edited, Test_Edited] #creacion de gurpo para el loop
for dataset in data:
    alley_qual = {"Grvl": 2, "Pave": 1} #crear un diccionario de correspondencia
    dataset['Alley'] = dataset['Alley'].map(alley_qual)  #cambiar los valores origniales con los del dic
    dataset['Alley']=dataset['Alley'].fillna(0) #cambiar los valores nulos por 0
    dataset['Alley']=dataset['Alley'].astype(int)
Test_Edited['Alley'].isnull().sum() #contar la cantidad de valores nulos
Test_Edited['Alley'].unique()
Train_Edited['Shed']=np.where(Original_Train['MiscFeature']=='Shed',1,0)
Train_Edited['Gar2']=np.where(Original_Train['MiscFeature']=='Gar2',1,0)
Train_Edited['Othr']=np.where(Original_Train['MiscFeature']=='Othr',1,0)
Train_Edited['TenC']=np.where(Original_Train['MiscFeature']=='TenC',1,0)
Train_Edited
#data = [Train_Edited, Test_Edited] #creacion de gurpo para el loop
#for dataset in data:
#    misc_fea = {"Shed": 4, "Gar2": 3, "Othr": 2, "TenC": 1} #crear un diccionario de correspondencia
#    dataset['MiscFeature'] = dataset['MiscFeature'].map(misc_fea)  #cambiar los valores origniales con los del dic
#    dataset['MiscFeature']=dataset['MiscFeature'].fillna(0) #cambiar los valores nulos por 0
#    dataset['MiscFeature']=dataset['MiscFeature'].astype(int)
#Train_Edited['MiscFeature'].isnull().sum() #contar la cantidad de valores nulos
Train_Edited['Shed'].unique()
testcor=Train_Edited.corr()
testcor[['Shed', 'Gar2', 'Othr', 'TenC']].loc['SalePrice':].head(1)
data = [Train_Edited, Test_Edited] #creacion de gurpo para el loop
for dataset in data:
    fire_qua = {"Ex": 5, "Gd": 4, "TA":3, "Fa":2, "Po":1}  #crear un diccionario de correspondencia
    dataset['FireplaceQu'] = dataset['FireplaceQu'].map(fire_qua)  #cambiar los valores origniales con los del dic
    dataset['FireplaceQu']=dataset['FireplaceQu'].fillna(0) #cambiar los valores nulos por 0
    dataset['FireplaceQu']=dataset['FireplaceQu'].astype(int)
Train_Edited['FireplaceQu'].isnull().sum() #contar la cantidad de valores nulos
Test_Edited['BsmtQual'].unique()
Train_Edited
# reemplazar los valores perdidos de LotFrontage con la mediana
Train_Edited['LotFrontage'].fillna(Original_Train['LotFrontage'].median(), inplace = True)
# reemplazar los valores perdidos de LotFrontage con la mediana
Test_Edited['LotFrontage'].fillna(Original_Test['LotFrontage'].median(), inplace = True)
Test_Edited['BsmtFullBath'].fillna(Original_Test['BsmtFullBath'].median(), inplace = True)
Test_Edited['BsmtHalfBath'].fillna(Original_Test['BsmtHalfBath'].median(), inplace = True)
Test_Edited['TotalBsmtSF'].fillna(Original_Test['TotalBsmtSF'].median(), inplace = True)
Test_Edited['GarageCars'].fillna(Original_Test['GarageCars'].median(), inplace = True)
Test_Edited['GarageArea'].fillna(Original_Test['GarageArea'].median(), inplace = True)
Test_Edited['BsmtUnfSF'].fillna(Original_Test['BsmtUnfSF'].median(), inplace = True)
Test_Edited['BsmtFinSF2'].fillna(Original_Test['BsmtFinSF2'].median(), inplace = True)
Test_Edited['BsmtFinSF1'].fillna(Original_Test['BsmtFinSF1'].median(), inplace = True)
Test_Edited['BaseArea'].fillna(Original_Test['BaseArea'].median(), inplace = True)
Train_Edited.drop(columns=['Shed', 'Gar2', 'Othr', 'TenC', 'MiscFeature', 'GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True) #eliminar la variable MiscFeature
#encontrar el total de valores perdidos de cada variable en el nuevo dataframe de train
train_total_edited = Train_Edited.isnull().sum().sort_values(ascending=False)
#expresar el porcentaje de valores perdidos por variable
train_percent_1_edited = Train_Edited.isnull().sum()/Train_Edited.isnull().count()*100
#redondear los decimales del porcentaje
train_percent_2_edited = (round(train_percent_1_edited, 1)).sort_values(ascending=False)
#crear un nuevo dataframe que muestre el total y porcentaje
train_missing_data_edited = pd.DataFrame([train_total_edited, train_percent_2_edited], index=["Total",'Porcentaje']).T
train_missing_data_edited.head(15)
Test_Edited.drop(columns=['MiscFeature', 'GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True) #eliminar la variable MiscFeature
#encontrar el total de valores perdidos de cada variable en el nuevo dataframe de Test
Test_total_edited = Test_Edited.isnull().sum().sort_values(ascending=False)
#expresar el porcentaje de valores perdidos por variable
Test_percent_1_edited = Test_Edited.isnull().sum()/Test_Edited.isnull().count()*100
#redondear los decimales del porcentaje
Test_percent_2_edited = (round(Test_percent_1_edited, 1)).sort_values(ascending=False)
#crear un nuevo dataframe que muestre el total y porcentaje
Test_missing_data_edited = pd.DataFrame([Test_total_edited, Test_percent_2_edited], index=["Total",'Porcentaje']).T
Test_missing_data_edited.head()
Original_Train['BsmtQual'].unique()

# crear un nuevo dataframe 
Train_pca=Train_Edited.drop(columns=['Id']) #eliminar las columnas Id y SalePrice
Train_pca
# establecer grupos de variables cuantitatvas y cualitativas
quantitative = [f for f in Train_pca.columns if Train_pca.dtypes[f] != 'object']
qualitative = [f for f in Train_pca.columns if Train_pca.dtypes[f] == 'object']
quantitative_SP = [f for f in Train_pca.columns if Train_pca.dtypes[f] != 'object']
quantitative_SP.remove('SalePrice')
#normalizar datos
Train_N=skl.preprocessing.StandardScaler().fit(Train_pca[quantitative]).transform(Train_pca[quantitative].astype(float))
Train_N=pd.DataFrame(Train_N.copy(), columns=Train_pca[quantitative].columns) # renombrar columanas con los nombres del DF original
Train_N #visualizar DF
#correr modelo PCA 
Train_N_SP=Train_N.copy().drop(columns='SalePrice')
pcs = skl.decomposition.PCA()
pcs.fit(Train_N_SP)
#obtener estadisticas de los componentes principales
pcsSummary_df = pd.DataFrame({'Standard deviation': np.sqrt(pcs.explained_variance_),
                           'Proportion of variance': pcs.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)})
pcsSummary_df = pcsSummary_df.transpose() #transponer DF
pcsSummary_df.columns = ['PC{}'.format(i) for i in range(1, len(pcsSummary_df.columns)+1)] #establecer el nombre de las columnas
pcsSummary_df.round(4) #precision de 4 decimales
pcsSummary_df.iloc[:,:20] #mostrar unicamente los 20 primeros PC
# encontrar los pesos de cada PC
pcsComponents_df = pd.DataFrame(pcs.components_.transpose(), columns=pcsSummary_df.columns, 
                                index=Train_N_SP[quantitative_SP].columns)
pcsComponents_df.iloc[:10,:13] #mostrar los pesos de los 13 primeros componentes
houses_red_df = Train_N_SP[quantitative_SP].dropna(axis=0)
houses_red_df = Train_N_SP[quantitative_SP].reset_index(drop=True)

scores = pd.DataFrame(pcs.fit_transform(skl.preprocessing.scale(houses_red_df.dropna(axis=0))), 
                      columns=[f'PC{i}' for i in range(1, 43)])
houses_pca_df = pd.concat([houses_red_df['MSSubClass'].dropna(axis=0), scores[['PC1', 'PC2']]], axis=1)
ax = houses_pca_df.plot.scatter(x='PC1', y='PC2', figsize=(6, 6))
points = houses_pca_df[['PC1','PC2','MSSubClass']]

texts = []
plt.show()
#obtener los valores transformados de cada uno de PC para todos los registros del set de datos
transform_df = pd.DataFrame(pcs.transform(Train_N_SP[quantitative_SP]), 
                      columns=pcsSummary_df.columns)
transform_df_13=transform_df.iloc[:,:13].copy() # DF con 13 componentes principales
transform_df_20=transform_df.iloc[:,:20].copy() # DF con 20 componentes principales
transform_df_20.head()
# PCA para el set de test

# crear un nuevo dataframe 
Test_pca=Test_Edited.drop(columns=['Id']) #eliminar las columnas Id y SalePrice
Test_pca


#normalizar datos
Test_N=skl.preprocessing.StandardScaler().fit(Test_pca[quantitative_SP]).transform(Test_pca[quantitative_SP].astype(float))
Test_N=pd.DataFrame(Test_N.copy(), columns=Test_pca[quantitative_SP].columns) # renombrar columanas con los nombres del DF original
Test_N #visualizar DF


#correr modelo PCA 
Test_N_SP=Test_N.copy()
pcs_test = skl.decomposition.PCA()
pcs_test.fit(Test_N_SP)
#obtener estadisticas de los componentes principales
pcsSummary_test_df = pd.DataFrame({'Standard deviation': np.sqrt(pcs_test.explained_variance_),
                           'Proportion of variance': pcs_test.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs_test.explained_variance_ratio_)})
pcsSummary_test_df = pcsSummary_test_df.transpose() #transponer DF
pcsSummary_test_df.columns = ['PC{}'.format(i) for i in range(1, len(pcsSummary_test_df.columns)+1)] #establecer el nombre de las columnas
pcsSummary_test_df.round(4) #precision de 4 decimales
pcsSummary_test_df.iloc[:,:20] #mostrar unicamente los 20 primeros PC

pcsComponents_test_df = pd.DataFrame(pcs_test.components_.transpose(), columns=pcsSummary_test_df.columns, 
                                index=Test_N_SP[quantitative_SP].columns)
pcsComponents_test_df.iloc[:10,:13] #mostrar los pesos de los 13 primeros componentes


houses_red_df_test = Test_N_SP[quantitative_SP].dropna(axis=0)
houses_red_df_test = Test_N_SP[quantitative_SP].reset_index(drop=True)

scores_test = pd.DataFrame(pcs_test.fit_transform(skl.preprocessing.scale(houses_red_df_test.dropna(axis=0))), 
                      columns=[f'PC{i}' for i in range(1, 43)])
houses_pca_df_test = pd.concat([houses_red_df_test['MSSubClass'].dropna(axis=0), scores_test[['PC1', 'PC2']]], axis=1)
ax = houses_pca_df_test.plot.scatter(x='PC1', y='PC2', figsize=(6, 6))
points = houses_pca_df_test[['PC1','PC2','MSSubClass']]

texts = []
plt.show()

#obtener los valores transformados de cada uno de PC para todos los registros del set de datos
transform_df_test = pd.DataFrame(pcs_test.transform(Test_N_SP[quantitative_SP]), 
                      columns=pcsSummary_test_df.columns)
transform_df_test_13=transform_df_test.iloc[:,:13].copy() # DF con 13 componentes principales
transform_df_test_20=transform_df_test.iloc[:,:20].copy() # DF con 20 componentes principales
transform_df_test_20.head()

PCA_X_train_13 = transform_df_13 #definimos los feautures del set train para los modelos de prediccion
PCA_Y_train_13 = Train_Edited["SalePrice"] #definimos el target
PCA_X_test_13 = transform_df_test_13 #definimos sobre que set de datos se desean realizar las prediccione

#random forest model
PCA_RF_13 = RandomForestClassifier(n_estimators=100)  #llamamos al primer modelo definiendo el numero de estimadores
PCA_RF_13.fit(PCA_X_train_13, PCA_Y_train_13) #asignamos el set de train
PCA_RF_EXIST_13 = PCA_RF_13.predict(PCA_X_train_13)
PCA_Y_prediction_13 = PCA_RF_13.predict(PCA_X_test_13) #asignamos sobre que deseamos calcular el predictor
PCA_ACC_RF_13 = PCA_RF_13.score(PCA_X_train_13, PCA_Y_train_13) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_RF_13= skl.metrics.mean_absolute_error(PCA_RF_EXIST_13, PCA_Y_train_13)
PCA_MSE_RF_13= skl.metrics.mean_squared_error(PCA_RF_EXIST_13, PCA_Y_train_13)
PCA_MSE_RF_13
# adaboosreg model
X_13=PCA_X_train_13
y_13=PCA_Y_train_13
PCA_AB_13=skl.ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
PCA_AB_13.fit(X_13,y_13)
PCA_Y_pred_ADAB_13=PCA_AB_13.predict(PCA_X_test_13)
PCA_AB_EXIST_13 = PCA_AB_13.predict(PCA_X_train_13)
PCA_ACC_AB_13= PCA_AB_13.score(PCA_X_train_13, PCA_Y_train_13) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_AB_13= skl.metrics.mean_absolute_error(PCA_AB_EXIST_13, PCA_Y_train_13)
PCA_MSE_AB_13= skl.metrics.mean_squared_error(PCA_AB_EXIST_13, PCA_Y_train_13)
#referencia: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
#linear regresion model
PCA_LR_13=skl.linear_model.LinearRegression()
PCA_LR_13.fit(X_13, y_13)
PCA_Y_pred_LR_13= PCA_LR_13.predict(PCA_X_test_13)
PCA_LR_EXIST_13 = PCA_LR_13.predict(PCA_X_train_13)
PCA_ACC_LR_13 = PCA_LR_13.score(PCA_X_train_13, PCA_Y_train_13) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_LR_13= skl.metrics.mean_absolute_error(PCA_LR_EXIST_13, PCA_Y_train_13)
PCA_MSE_LR_13= skl.metrics.mean_squared_error(PCA_LR_EXIST_13, PCA_Y_train_13)
print('Intercept (b0):',PCA_LR_13.intercept_)
# For retrieving the slope:
print('****************')
print('(bi):',PCA_LR_13.coef_)
print('****************')
PCA_X_train_20 = transform_df_20 #definimos los feautures del set train para los modelos de prediccion
PCA_Y_train_20 = Train_Edited["SalePrice"] #definimos el target
PCA_X_test_20 = transform_df_test_20 #definimos sobre que set de datos se desean realizar las prediccione

#random forest model
PCA_RF_20 = RandomForestClassifier(n_estimators=100)  #llamamos al primer modelo definiendo el numero de estimadores
PCA_RF_20.fit(PCA_X_train_20, PCA_Y_train_20) #asignamos el set de train
PCA_RF_EXIST_20 = PCA_RF_20.predict(PCA_X_train_20)
PCA_Y_prediction_20 = PCA_RF_20.predict(PCA_X_test_20) #asignamos sobre que deseamos calcular el predictor
PCA_ACC_RF_20 = PCA_RF_20.score(PCA_X_train_20, PCA_Y_train_20) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_RF_20= skl.metrics.mean_absolute_error(PCA_RF_EXIST_20, PCA_Y_train_20)
PCA_MSE_RF_20= skl.metrics.mean_squared_error(PCA_RF_EXIST_20, PCA_Y_train_20)
# adaboosreg model
X_20=PCA_X_train_20
y_20=PCA_Y_train_20
PCA_AB_20=skl.ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
PCA_AB_20.fit(X_20,y_20)
PCA_Y_pred_ADAB_20=PCA_AB_20.predict(PCA_X_test_20)
PCA_AB_EXIST_20 = PCA_AB_20.predict(PCA_X_train_20)
PCA_ACC_AB_20= PCA_AB_20.score(PCA_X_train_20, PCA_Y_train_20) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_AB_20= skl.metrics.mean_absolute_error(PCA_AB_EXIST_20, PCA_Y_train_20)
PCA_MSE_AB_20= skl.metrics.mean_squared_error(PCA_AB_EXIST_20, PCA_Y_train_20)
#referencia: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
#linear regresion model
PCA_LR_20=skl.linear_model.LinearRegression()
PCA_LR_20.fit(X_20, y_20)
PCA_Y_pred_LR_20= PCA_LR_20.predict(PCA_X_test_20)
PCA_LR_EXIST_20 = PCA_LR_20.predict(PCA_X_train_20)
PCA_ACC_LR_20 = PCA_LR_20.score(PCA_X_train_20, PCA_Y_train_20) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_LR_20= skl.metrics.mean_absolute_error(PCA_LR_EXIST_20, PCA_Y_train_20)
PCA_MSE_LR_20= skl.metrics.mean_squared_error(PCA_LR_EXIST_20, PCA_Y_train_20)
print('Intercept (b0):',PCA_LR_20.intercept_)
# For retrieving the slope:
print('****************')
print('(bi):',PCA_LR_20.coef_)
print('****************')
#cross validate linear regression PCA
PCA_LR_CV_SCORE = cross_validate(PCA_LR_13, PCA_X_train_13, PCA_Y_train_13, cv=4,scoring=['neg_mean_absolute_error','neg_mean_squared_error'],return_estimator=True) #llamo a cross validate definiendo feautures del dataframe auxiliar, la variable de salida, el valor de k, y las metricas deseadas


PCA_LR_CV_METRICS=pd.DataFrame(PCA_LR_CV_SCORE, columns=['test_neg_mean_squared_error']) #almaceno las metricas obtenidas en un dataframe
PCA_LR_CV_METRICS.rename(columns={'test_neg_mean_squared_error':'MSE'}, inplace=True) #cambio los nombres predeterminados por nombres mas simples
PCA_LR_CV_METRICS_2=pd.DataFrame() #creo un nuevo dataframe

for i in PCA_LR_CV_SCORE['estimator']: 
    PCA_LR_CV_METRICS_2.loc[i,'MAPE']=((i.predict(PCA_X_train_13)-PCA_Y_train_13).abs()/PCA_Y_train_13).sum()/PCA_X_train_13['PC1'].count()*100 ##guardo en el dataframe, el MAPE ajustado de cada modelo a meida itera
PCA_LR_CV_METRICS_2.reset_index(drop=True, inplace=True) #borro el indice para cambiar el codigo del modelo por numeros iniciando en 0
PCA_LR_CV_METRICS_FINAL=pd.DataFrame(PCA_LR_CV_METRICS_2[['MAPE']].copy()) #genero un dataframe que copie las metricas del pirmer dataframe de metricas 
PCA_LR_CV_METRICS_FINAL[['MSE']]=PCA_LR_CV_METRICS[['MSE']].copy() #añado las columnas del segundo dataframe de metricas al dataframe final

PCA_LR_CV_METRICS_FINAL=abs(PCA_LR_CV_METRICS_FINAL) #cambio todos los valores del dataframe final por sus valores absolutos para evitar negativos

PCA_LR_CV_METRICS_FINAL.rename(index={0: 'Fold_1',1: 'Fold_2',2: 'Fold_3',3: 'Fold_4' }, inplace=True) #renombro los indices segun el numero de fold
PCA_LR_CV_METRICS_FINAL #visualzo el datagrame final
#saving de dataframes en archivos csv excluyendo la columna de index para datos CON y SIN AGE
Pred_RF_PCA_13=pd.DataFrame()
Pred_RF_PCA_13['Id']=Original_Test['Id']
Pred_RF_PCA_13['SalePrice']=PCA_Y_prediction_13
Pred_RF_PCA_13.to_csv('Pred_RF_PCA_13.csv',index=False)

Pred_LR_PCA_13=pd.DataFrame()
Pred_LR_PCA_13['Id']=Original_Test['Id']
Pred_LR_PCA_13['SalePrice']=PCA_Y_pred_LR_13
Pred_LR_PCA_13.to_csv('Pred_LR_PCA_13.csv',index=False)

Pred_AB_PCA_13=pd.DataFrame()
Pred_AB_PCA_13['Id']=Original_Test['Id']
Pred_AB_PCA_13['SalePrice']=PCA_Y_pred_ADAB_13
Pred_AB_PCA_13.to_csv('Pred_AB_PCA_13.csv',index=False)

Pred_RF_PCA_20=pd.DataFrame()
Pred_RF_PCA_20['Id']=Original_Test['Id']
Pred_RF_PCA_20['SalePrice']=PCA_Y_prediction_20
Pred_RF_PCA_20.to_csv('Pred_RF_PCA_20.csv',index=False)

Pred_LR_PCA_20=pd.DataFrame()
Pred_LR_PCA_20['Id']=Original_Test['Id']
Pred_LR_PCA_20['SalePrice']=PCA_Y_pred_LR_20
Pred_LR_PCA_20.to_csv('Pred_LR_PCA_20.csv',index=False)

Pred_AB_PCA_20=pd.DataFrame()
Pred_AB_PCA_20['Id']=Original_Test['Id']
Pred_AB_PCA_20['SalePrice']=PCA_Y_pred_ADAB_20
Pred_AB_PCA_20.to_csv('Pred_AB_PCA_20.csv',index=False)
obj_df = Train_Edited.select_dtypes(include=['object']).copy() #analysis de object variables
obj_df.head()
corr_2=Train_Edited.corr()
saleprice_corr_2=pd.DataFrame(corr_2['SalePrice'])
saleprice_corr_2.sort_values('SalePrice',ascending=False, inplace=True)
saleprice_corr_2.reset_index(inplace=True)
saleprice_corr_2.rename(columns={'index':'Variable'}, inplace=True)
saleprice_corr_2.head(10)
qualitative_model=['MSZoning', 'Neighborhood', 'MasVnrType', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType'] #seleccion de variables qualitativas en base a conocimientos previos
quantitative_model=['OverallQual','BaseArea','GrLivArea','Exqual_YearB','ExterQual','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF'] #seleccion de variables cuantitativas en funcion de su correlacion con sale price

Train_Edited[qualitative_model].info() #analisis de mssing values para variables categoricas en set de entrenamiento
Test_Edited[qualitative_model].info() #analisis de mssing values para variables categoricas en set de prueba
Train_Edited[quantitative_model].info() #analisis de mssing values para variables cuantitativas en set de entrenamiento
Test_Edited[quantitative_model].info() #analisis de mssing values para variables cuantitativas en set de entrenamiento
Train_Edited_m1=pd.DataFrame()
Train_Edited_m1[qualitative_model]=Train_Edited[qualitative_model].copy()
Train_Edited_m1[quantitative_model]=Train_Edited[quantitative_model].copy()
Train_Edited_m1['SalePrice']=Train_Edited['SalePrice'].copy()

Test_Edited_m1=pd.DataFrame()
Test_Edited_m1[qualitative_model]=Test_Edited[qualitative_model].copy()
Test_Edited_m1[quantitative_model]=Test_Edited[quantitative_model].copy()

Train_Edited_m1
data_m1 = [Train_Edited_m1,Test_Edited_m1] #creacion de gurpo para el loop
for dataset in data_m1:
    msz = {"FV": 5, "RL": 4, "RH":3, "RM":2, "C (all)":1}  #crear un diccionario de correspondencia
    dataset['MSZoning'] = dataset['MSZoning'].map(msz)  #cambiar los valores origniales con los del dic
    dataset['MSZoning']=dataset['MSZoning'].fillna(dataset['MSZoning'].median())
    dataset['MSZoning']=dataset['MSZoning'].astype(int)   
for dataset in data_m1:
    NB = {"CollgCr": 25, "OldTown": 24, "Edwards":23, "Somerst":22, "Gilbert":21,"NridgHt": 20, "Sawyer": 19, "NWAmes":18, "SawyerW":17, "BrkSide":16,"Crawfor": 15, "Mitchel": 14, "NoRidge":13, "Timber":12, "IDOTRR":10,"ClearCr": 9, "StoneBr":8, "SWISU":7, "Blmngtn":6,"MeadowV": 5, "BrDale": 4, "Veenker":3, "NPKVill":2, "Blueste":1}  #crear un diccionario de correspondencia
    dataset['Neighborhood'] = dataset['Neighborhood'].map(NB)  #cambiar los valores origniales con los del dic
    dataset['Neighborhood']=dataset['Neighborhood'].fillna(dataset['Neighborhood'].median())
    dataset['Neighborhood']=dataset['Neighborhood'].astype(int)   
for dataset in data_m1:
    MVT = {"Stone": 4, "BrkFace": 3, "BrkCmn":2, "None":1}  #crear un diccionario de correspondencia
    dataset['MasVnrType'] = dataset['MasVnrType'].map(MVT)  #cambiar los valores origniales con los del dic
    dataset['MasVnrType']=dataset['MasVnrType'].fillna(dataset['MasVnrType'].median())
    dataset['MasVnrType']=dataset['MasVnrType'].astype(int)   
for dataset in data_m1:
    BMQ = {"Ex": 4, "Gd": 3, "TA":2, "Fa":1}  #crear un diccionario de correspondencia
    dataset['BsmtQual'] = dataset['BsmtQual'].map(BMQ)  #cambiar los valores origniales con los del dic
    dataset['BsmtQual']=dataset['BsmtQual'].fillna(dataset['BsmtQual'].median())
    dataset['BsmtQual']=dataset['BsmtQual'].astype(int)   
for dataset in data_m1:
    CA = {"Y": 1, "N": 0}  #crear un diccionario de correspondencia
    dataset['CentralAir'] = dataset['CentralAir'].map(CA)  #cambiar los valores origniales con los del dic
    dataset['CentralAir']=dataset['CentralAir'].fillna(dataset['CentralAir'].median())
    dataset['CentralAir']=dataset['CentralAir'].astype(int)   
for dataset in data_m1:
    EL = {"SBrkr": 5, "FuseA": 4,"FuseF":3,"FuseP":2,"Mix":1}  #crear un diccionario de correspondencia
    dataset['Electrical'] = dataset['Electrical'].map(EL)  #cambiar los valores origniales con los del dic
    dataset['Electrical']=dataset['Electrical'].fillna(dataset['Electrical'].median())
    dataset['Electrical']=dataset['Electrical'].astype(int)   
for dataset in data_m1:
    KQ = {"Ex": 4,"Gd":3,"TA":2,"Fa":1}  #crear un diccionario de correspondencia
    dataset['KitchenQual'] = dataset['KitchenQual'].map(KQ)  #cambiar los valores origniales con los del dic
    dataset['KitchenQual']=dataset['KitchenQual'].fillna(dataset['KitchenQual'].median())
    dataset['KitchenQual']=dataset['KitchenQual'].astype(int)   
for dataset in data_m1:
    KQ = {"Con": 9,"New":8,"CWD":7,"WD":6,"ConLw": 5,"COD":4,"ConLD":3,"ConLI":2,"Oth": 1}  #crear un diccionario de correspondencia
    dataset['SaleType'] = dataset['SaleType'].map(KQ)  #cambiar los valores origniales con los del dic
    dataset['SaleType']=dataset['SaleType'].fillna(dataset['SaleType'].median())
    dataset['SaleType']=dataset['SaleType'].astype(int)  
for dataset in data_m1:
    dataset['OverallQual']=dataset['OverallQual'].fillna(dataset['OverallQual'].median())
    dataset['OverallQual']=dataset['OverallQual'].astype(int)  
for dataset in data_m1:
    dataset['BaseArea']=dataset['BaseArea'].fillna(dataset['BaseArea'].median())
    dataset['BaseArea']=dataset['BaseArea'].astype(int)  
for dataset in data_m1:
    dataset['GrLivArea']=dataset['GrLivArea'].fillna(dataset['GrLivArea'].median())
    dataset['GrLivArea']=dataset['GrLivArea'].astype(int)  
for dataset in data_m1:
    dataset['Exqual_YearB']=dataset['Exqual_YearB'].fillna(dataset['Exqual_YearB'].median())
    dataset['Exqual_YearB']=dataset['Exqual_YearB'].astype(int)  
for dataset in data_m1:
    dataset['ExterQual']=dataset['ExterQual'].fillna(dataset['ExterQual'].median())
    dataset['ExterQual']=dataset['ExterQual'].astype(int)  
for dataset in data_m1:
    dataset['GarageCars']=dataset['GarageCars'].fillna(dataset['GarageCars'].median())
    dataset['GarageCars']=dataset['GarageCars'].astype(int)  
for dataset in data_m1:
    dataset['GarageArea']=dataset['GarageArea'].fillna(dataset['GarageArea'].median())
    dataset['GarageArea']=dataset['GarageArea'].astype(int)  
for dataset in data_m1:
    dataset['TotalBsmtSF']=dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].median())
    dataset['TotalBsmtSF']=dataset['TotalBsmtSF'].astype(int)  
for dataset in data_m1:
    dataset['1stFlrSF']=dataset['1stFlrSF'].fillna(dataset['1stFlrSF'].median())
    dataset['1stFlrSF']=dataset['1stFlrSF'].astype(int)  
X_Train_Final_m1=pd.DataFrame()
X_Train_Final_m1[qualitative_model]=Train_Edited_m1[qualitative_model].copy()
X_Train_Final_m1[quantitative_model]=Train_Edited_m1[quantitative_model].copy()
Y_Train_Final_m1=Train_Edited_m1['SalePrice']

X_Test_Final_m1=pd.DataFrame()
X_Test_Final_m1[qualitative_model]=Test_Edited_m1[qualitative_model].copy()
X_Test_Final_m1[quantitative_model]=Test_Edited_m1[quantitative_model].copy() 
X_Train_Final_m1
X_Test_Final_m1
Y_Train_Final_m1
#random forest model
AP2_RF = RandomForestClassifier(n_estimators=100)  #llamamos al primer modelo definiendo el numero de estimadores
AP2_RF.fit(X_Train_Final_m1, Y_Train_Final_m1) #asignamos el set de train
AP2_RF_EXIST= AP2_RF.predict(X_Train_Final_m1)
AP2_RF_PRED = AP2_RF.predict(X_Test_Final_m1) #asignamos sobre que deseamos calcular el predictor
AP2_ACC_RF = AP2_RF.score(X_Train_Final_m1, Y_Train_Final_m1) * 100 #sacamos el porcetaje de certeza del modelo
AP2_MAE_RF= skl.metrics.mean_absolute_error(AP2_RF_EXIST, Y_Train_Final_m1)
AP2_MSE_RF= skl.metrics.mean_squared_error(AP2_RF_EXIST, Y_Train_Final_m1)
# adaboosreg model
AP2_AB=skl.ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
AP2_AB.fit(X_Train_Final_m1,Y_Train_Final_m1)
AP2_AB_EXIST= AP2_AB.predict(X_Train_Final_m1)
AP2_AB_PRED=AP2_AB.predict(X_Test_Final_m1)
AP2_ACC_AB = AP2_AB.score(X_Train_Final_m1, Y_Train_Final_m1) * 100 #sacamos el porcetaje de certeza del modelo
AP2_MAE_AB= skl.metrics.mean_absolute_error(AP2_AB_EXIST, Y_Train_Final_m1)
AP2_MSE_AB= skl.metrics.mean_squared_error(AP2_AB_EXIST, Y_Train_Final_m1)
#referencia: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
#linear regresion model
AP2_LR=skl.linear_model.LinearRegression()
AP2_LR.fit(X_Train_Final_m1,Y_Train_Final_m1)
AP2_LR_EXIST= AP2_LR.predict(X_Train_Final_m1)
AP2_LR_PRED= AP2_LR.predict(X_Test_Final_m1)
AP2_ACC_LR = AP2_LR.score(X_Train_Final_m1, Y_Train_Final_m1) * 100 #sacamos el porcetaje de certeza del modelo
AP2_MAE_LR= skl.metrics.mean_absolute_error(AP2_LR_EXIST, Y_Train_Final_m1)
AP2_MSE_LR= skl.metrics.mean_squared_error(AP2_LR_EXIST, Y_Train_Final_m1)
print('Intercept (b0):',AP2_LR.intercept_)
# For retrieving the slope:
print('(bi):',AP2_LR.coef_)
print('****************')
#cross validate para linear regression approach 2
AP2_LR_CV_SCORE = cross_validate(AP2_LR, X_Train_Final_m1, Y_Train_Final_m1, cv=4,scoring=['neg_mean_absolute_error','neg_mean_squared_error'],return_estimator=True) #llamo a cross validate definiendo feautures del dataframe auxiliar, la variable de salida, el valor de k, y las metricas deseadas


AP2_LR_CV_METRICS=pd.DataFrame(AP2_LR_CV_SCORE, columns=['test_neg_mean_squared_error']) #almaceno las metricas obtenidas en un dataframe
AP2_LR_CV_METRICS.rename(columns={'test_neg_mean_squared_error':'MSE'}, inplace=True) #cambio los nombres predeterminados por nombres mas simples
AP2_LR_CV_METRICS_2=pd.DataFrame() #creo un nuevo dataframe

for i in AP2_LR_CV_SCORE['estimator']: 
    AP2_LR_CV_METRICS_2.loc[i,'MAPE']=((i.predict(X_Train_Final_m1)-Y_Train_Final_m1).abs()/Y_Train_Final_m1).sum()/X_Train_Final_m1['Electrical'].count()*100 ##guardo en el dataframe, el MAPE ajustado de cada modelo a meida itera
AP2_LR_CV_METRICS_2.reset_index(drop=True, inplace=True) #borro el indice para cambiar el codigo del modelo por numeros iniciando en 0
AP2_LR_CV_METRICS_FINAL=pd.DataFrame(AP2_LR_CV_METRICS_2[['MAPE']].copy()) #genero un dataframe que copie las metricas del pirmer dataframe de metricas 
AP2_LR_CV_METRICS_FINAL[['MSE']]=AP2_LR_CV_METRICS[['MSE']].copy() #añado las columnas del segundo dataframe de metricas al dataframe final

AP2_LR_CV_METRICS_FINAL=abs(AP2_LR_CV_METRICS_FINAL) #cambio todos los valores del dataframe final por sus valores absolutos para evitar negativos

AP2_LR_CV_METRICS_FINAL.rename(index={0: 'Fold_1',1: 'Fold_2',2: 'Fold_3',3: 'Fold_4' }, inplace=True) #renombro los indices segun el numero de fold
AP2_LR_CV_METRICS_FINAL #visualzo el datagrame final
 #saving de dataframes en archivos csv
Pred_RF_AP2=pd.DataFrame()
Pred_RF_AP2['Id']=Original_Test['Id']
Pred_RF_AP2['SalePrice']=AP2_RF_PRED
Pred_RF_AP2.to_csv('Pred_RF_AP2.csv',index=False)

Pred_LR_AP2=pd.DataFrame()
Pred_LR_AP2['Id']=Original_Test['Id']
Pred_LR_AP2['SalePrice']=AP2_LR_PRED
Pred_LR_AP2.to_csv('Pred_LR_AP2.csv',index=False)

Pred_AB_AP2=pd.DataFrame()
Pred_AB_AP2['Id']=Original_Test['Id']
Pred_AB_AP2['SalePrice']=AP2_AB_PRED
Pred_AB_AP2.to_csv('Pred_AB_AP2.csv',index=False)
#crear un data frame de train con las variables categoricas codificadas y las variables cuantitativas
X_Train_Final_m2=pd.DataFrame(X_Train_Final_m1[qualitative_model].copy())
columns_3=Train_pca[quantitative_SP].columns
for i in columns_3:
    X_Train_Final_m2[i]=Train_pca[i].copy()
X_Train_Final_m2
#crear un data frame de test con las variables categoricas codificadas y las variables cuantitativas
X_Test_Final_m2=pd.DataFrame(X_Test_Final_m1[qualitative_model].copy())
columns_3=Test_pca[quantitative_SP].columns
for i in columns_3:
    X_Test_Final_m2[i]=Test_pca[i].copy()
X_Test_Final_m2.info()
# crear una variable con lo valores del outcome del set de train
Y_Train_Final_m2=Y_Train_Final_m1.copy()

# normalizar datos
X_Train_Final_m2_N=skl.preprocessing.StandardScaler().fit(X_Train_Final_m2).transform(X_Train_Final_m2.astype(float))
X_Train_Final_m2_N=pd.DataFrame(X_Train_Final_m2_N.copy(), columns=X_Train_Final_m2.columns) # renombrar columanas con los nombres del DF original

X_Test_Final_m2_N=skl.preprocessing.StandardScaler().fit(X_Test_Final_m2).transform(X_Test_Final_m2.astype(float))
X_Test_Final_m2_N=pd.DataFrame(X_Test_Final_m2_N.copy(), columns=X_Test_Final_m2.columns) # renombrar columanas con los nombres del DF original
#correr modelo PCA 
pcs_train_3 = skl.decomposition.PCA()
pcs_train_3.fit(X_Train_Final_m2_N)
#obtener estadisticas de los componentes principales
pcsSummary_train_3_df = pd.DataFrame({'Standard deviation': np.sqrt(pcs_train_3.explained_variance_),
                           'Proportion of variance': pcs_train_3.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs_train_3.explained_variance_ratio_)})
pcsSummary_train_3_df = pcsSummary_train_3_df.transpose() #transponer DF
pcsSummary_train_3_df.columns = ['PC{}'.format(i) for i in range(1, len(pcsSummary_train_3_df.columns)+1)] #establecer el nombre de las columnas
pcsSummary_train_3_df.round(4) #precision de 4 decimales
pcsSummary_train_3_df.iloc[:,:30] #mostrar unicamente los 20 primeros PC
pcsComponents_train_3_df = pd.DataFrame(pcs_train_3.components_.transpose(), columns=pcsSummary_train_3_df.columns, 
                                index=X_Train_Final_m2_N.columns)
pcsComponents_train_3_df.iloc[:10,:25] #mostrar los pesos de los 13 primeros componentes
houses_red_df_train_3 = X_Train_Final_m2_N.dropna(axis=0)
houses_red_df_train_3 = X_Train_Final_m2_N.reset_index(drop=True)
scores_train_3 = pd.DataFrame(pcs_train_3.fit_transform(skl.preprocessing.scale(houses_red_df_train_3.dropna(axis=0))), 
                      columns=[f'PC{i}' for i in range(1, 51)])
#obtener los valores transformados de cada uno de PC para todos los registros del set de datos
transform_df_train_3 = pd.DataFrame(pcs_train_3.transform(X_Train_Final_m2_N), 
                      columns=pcsSummary_train_3_df.columns)
transform_df_train_3=transform_df_train_3.iloc[:,:25] #25 componentes principales
transform_df_train_3
#correr modelo PCA 
pcs_test_3 = skl.decomposition.PCA()
pcs_test_3.fit(X_Test_Final_m2_N)
#obtener estadisticas de los componentes principales
pcsSummary_test_3_df = pd.DataFrame({'Standard deviation': np.sqrt(pcs_test_3.explained_variance_),
                           'Proportion of variance': pcs_test_3.explained_variance_ratio_,
                           'Cumulative proportion': np.cumsum(pcs_test_3.explained_variance_ratio_)})
pcsSummary_test_3_df = pcsSummary_test_3_df.transpose() #transponer DF
pcsSummary_test_3_df.columns = ['PC{}'.format(i) for i in range(1, len(pcsSummary_test_3_df.columns)+1)] #establecer el nombre de las columnas
pcsSummary_test_3_df.round(4) #precision de 4 decimales
pcsSummary_test_3_df.iloc[:,:20] #mostrar unicamente los 20 primeros PC
pcsComponents_test_3_df = pd.DataFrame(pcs_test_3.components_.transpose(), columns=pcsSummary_test_3_df.columns, 
                                index=X_Test_Final_m2_N.columns)
pcsComponents_test_3_df.iloc[:10,:13] #mostrar los pesos de los 13 primeros componentes
houses_red_df_test_3 = X_Test_Final_m2_N.dropna(axis=0)
houses_red_df_test_3 = X_Test_Final_m2_N.reset_index(drop=True)
scores_test_3 = pd.DataFrame(pcs_test_3.fit_transform(skl.preprocessing.scale(houses_red_df_test_3.dropna(axis=0))), 
                      columns=[f'PC{i}' for i in range(1, 51)])
#obtener los valores transformados de cada uno de PC para todos los registros del set de datos
transform_df_test_3 = pd.DataFrame(pcs_test_3.transform(X_Test_Final_m2_N), 
                      columns=pcsSummary_test_3_df.columns)
transform_df_test_3=transform_df_test_3.iloc[:,:25] # 25 pc
transform_df_test_3
PCA_X_train_3 = transform_df_train_3 #definimos los feautures del set train para los modelos de prediccion
PCA_Y_train_3 = Train_Edited["SalePrice"] #definimos el target
PCA_X_test_3 = transform_df_test_3 #definimos sobre que set de datos se desean realizar las prediccione

PCA_RF_3 = RandomForestClassifier(n_estimators=100)  #llamamos al primer modelo definiendo el numero de estimadores
PCA_RF_3.fit(PCA_X_train_3, PCA_Y_train_3) #asignamos el set de train
PCA_RF_EXIST_3 = PCA_RF_3.predict(PCA_X_train_3)
PCA_Y_prediction_3 = PCA_RF_3.predict(PCA_X_test_3) #asignamos sobre que deseamos calcular el predictor
PCA_ACC_RF_3 = PCA_RF_3.score(PCA_X_train_3, PCA_Y_train_3) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_RF_3= skl.metrics.mean_absolute_error(PCA_RF_EXIST_3, PCA_Y_train_3)
PCA_MSE_RF_3= skl.metrics.mean_squared_error(PCA_RF_EXIST_3, PCA_Y_train_3)
PCA_MSE_RF_3
from sklearn.model_selection import train_test_split
# partición de set de datos para poder obtener el mejor n
PCA_split=PCA_X_train_3.copy()
PCA_split['Sale Price']= Train_Edited["SalePrice"]
PCA_train_split, PCA_test_split = train_test_split(PCA_split,test_size=0.25, random_state=1)
print('Training : ', PCA_train_split.shape) #imprimir la dimensionalidad del set de entrenamiento
print('Test : ', PCA_test_split.shape) #imprimir dimensionalidad del set de test
X_train_split=PCA_train_split.drop(columns='Sale Price')
y_train_split=PCA_train_split['Sale Price']
X_test_split=PCA_test_split.drop(columns='Sale Price')
y_test_split=PCA_test_split['Sale Price']
results=[]
for k in range(1,100):
    PCA_RF_3 =  RandomForestClassifier(n_estimators=k).fit(X_train_split,y_train_split)
    results.append({
        'k': k,
        'MAE': skl.metrics.mean_absolute_error(y_test_split, PCA_RF_3.predict(X_test_split)),
        'MSE': skl.metrics.mean_squared_error(y_test_split, PCA_RF_3.predict(X_test_split))
    })
# Convert results to a pandas data frame
results = pd.DataFrame(results)
results.sort_values(["k",'MAE','MSE'],ascending=True).head(10)
print('El mejor valor de n es: \n', results.sort_values(['MAE','MSE'],ascending=True).iloc[0,0:3])
PCA_RF_3 = RandomForestClassifier(n_estimators=91)  #llamamos al primer modelo definiendo el numero de estimadores
PCA_RF_3.fit(PCA_X_train_3, PCA_Y_train_3) #asignamos el set de train
PCA_RF_EXIST_3 = PCA_RF_3.predict(PCA_X_train_3)
PCA_Y_prediction_3 = PCA_RF_3.predict(PCA_X_test_3) #asignamos sobre que deseamos calcular el predictor
PCA_ACC_RF_3 = PCA_RF_3.score(PCA_X_train_3, PCA_Y_train_3) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_RF_3= skl.metrics.mean_absolute_error(PCA_RF_EXIST_3, PCA_Y_train_3)
PCA_MSE_RF_3= skl.metrics.mean_squared_error(PCA_RF_EXIST_3, PCA_Y_train_3)
PCA_MSE_RF_3
# adaboosreg model
PCA_AB_3=skl.ensemble.AdaBoostRegressor(random_state=0, n_estimators=100)
PCA_AB_3.fit(PCA_X_train_3,PCA_Y_train_3)
PCA_Y_pred_ADAB_3=PCA_AB_3.predict(PCA_X_test_3)
PCA_AB_EXIST_3 = PCA_AB_3.predict(PCA_X_train_3)
PCA_ACC_AB_3= PCA_AB_3.score(PCA_X_train_3, PCA_Y_train_3) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_AB_3= skl.metrics.mean_absolute_error(PCA_AB_EXIST_3, PCA_Y_train_3)
PCA_MSE_AB_3= skl.metrics.mean_squared_error(PCA_AB_EXIST_3, PCA_Y_train_3)
#referencia: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
results_ab=[]
for b in range(1,100):
    PCA_AB_rf =  skl.ensemble.AdaBoostRegressor(n_estimators=b).fit(X_train_split,y_train_split)
    results_ab.append({
        'b': b,
        'MAE': skl.metrics.mean_absolute_error(y_test_split, PCA_AB_rf.predict(X_test_split)),
        'MSE': skl.metrics.mean_squared_error(y_test_split, PCA_AB_rf.predict(X_test_split))
    })
# Convert results to a pandas data frame
results_ab = pd.DataFrame(results_ab)
results_ab.sort_values(['MAE'],ascending=True).head()
# adaboosreg model
PCA_AB_3=skl.ensemble.AdaBoostRegressor(random_state=0, n_estimators=84)
PCA_AB_3.fit(PCA_X_train_3,PCA_Y_train_3)
PCA_Y_pred_ADAB_3=PCA_AB_3.predict(PCA_X_test_3)
PCA_AB_EXIST_3 = PCA_AB_3.predict(PCA_X_train_3)
PCA_ACC_AB_3= PCA_AB_3.score(PCA_X_train_3, PCA_Y_train_3) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_AB_3= skl.metrics.mean_absolute_error(PCA_AB_EXIST_3, PCA_Y_train_3)
PCA_MSE_AB_3= skl.metrics.mean_squared_error(PCA_AB_EXIST_3, PCA_Y_train_3)
#referencia: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
#linear regresion model
PCA_LR_3=skl.linear_model.LinearRegression()
PCA_LR_3.fit(PCA_X_train_3, PCA_Y_train_3)
PCA_Y_pred_LR_3= PCA_LR_3.predict(PCA_X_test_3)
PCA_LR_EXIST_3 = PCA_LR_3.predict(PCA_X_train_3)
PCA_ACC_LR_3 = PCA_LR_3.score(PCA_X_train_3, PCA_Y_train_3) * 100 #sacamos el porcetaje de certeza del modelo
PCA_MAE_LR_3= skl.metrics.mean_absolute_error(PCA_LR_EXIST_3, PCA_Y_train_3)
PCA_MSE_LR_3= skl.metrics.mean_squared_error(PCA_LR_EXIST_3, PCA_Y_train_3)
#saving de dataframes en archivos csv excluyendo la columna de index para datos CON y SIN AGE
Pred_RF_PCA_3=pd.DataFrame()
Pred_RF_PCA_3['Id']=Original_Test['Id']
Pred_RF_PCA_3['SalePrice']=PCA_Y_prediction_3
Pred_RF_PCA_3.to_csv('Pred_RF_PCA_3.csv',index=False)

Pred_LR_PCA_3=pd.DataFrame()
Pred_LR_PCA_3['Id']=Original_Test['Id']
Pred_LR_PCA_3['SalePrice']=PCA_Y_pred_LR_3
Pred_LR_PCA_3.to_csv('Pred_LR_PCA_3.csv',index=False)

Pred_AB_PCA_3=pd.DataFrame()
Pred_AB_PCA_3['Id']=Original_Test['Id']
Pred_AB_PCA_3['SalePrice']=PCA_Y_pred_ADAB_3
Pred_AB_PCA_3.to_csv('Pred_AB_PCA_3.csv',index=False)
Accuracy_df = pd.DataFrame({'Model': ['Random Forest','Linear Reg','ADA Boost'],
                        'AP-2 Acc':[AP2_ACC_RF, AP2_ACC_LR, AP2_ACC_AB],
                        '13 PCA Acc':[PCA_ACC_RF_13,PCA_ACC_LR_13,PCA_ACC_AB_13],
                        '20 PCA Acc':[PCA_ACC_RF_20,PCA_ACC_LR_20,PCA_ACC_AB_20],
                        '25 PCA Acc':[PCA_ACC_RF_3,PCA_ACC_LR_3,PCA_ACC_AB_3],
                        'AP-2 MAE':[AP2_MAE_RF,AP2_MAE_LR,AP2_MAE_AB],
                        '13 PCA MAE':[PCA_MAE_RF_13,PCA_MAE_LR_13,PCA_MAE_AB_13],  
                        '20 PCA MAE':[PCA_MAE_RF_20,PCA_MAE_LR_20,PCA_MAE_AB_20],
                        '25 PCA MAE':[PCA_MAE_RF_3,PCA_MAE_LR_3,PCA_MAE_AB_3],
                        'AP-2 MSE': [AP2_MSE_RF, AP2_MSE_LR, AP2_MSE_AB],
                        '13 PCA MSE':[PCA_MSE_RF_13,PCA_MSE_LR_13,PCA_MSE_AB_13],
                        '20 PCA MSE':[PCA_MSE_RF_20,PCA_MSE_LR_20,PCA_MSE_AB_20],
                        '25 PCA MSE':[PCA_MSE_RF_3,PCA_MSE_LR_3,PCA_MSE_AB_3],   
                        })
Accuracy_df               
