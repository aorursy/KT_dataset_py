import pandas as pd
import sklearn
import numpy as np
import warnings
import matplotlib as plt
%matplotlib inline
# Lendo a base de treino
trainDB = pd.read_csv("../input/train.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN"
                     )

# Lendo a base de teste
testDB = pd.read_csv("../input/test.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN"
                    )
# Plot da base de treino
trainDB.iloc[0:20,:]
# Tamanho da base de treino
size_train = trainDB.shape
size_train
# Total de cômodos

total_r = trainDB['total_rooms'] 
r_max = total_r.max()
r_min = total_r.min()
print('O maximo de cômodos em uma regiao é',r_max ,'e o mínimo é ',r_min)

# Divisão de classes -> 0 - 1k, 1k - 3k, 3k - 5k, 5k - 10k, 10k - 40k
clas = np.array([0,0,0,0,0])
for i in range(len(total_r)):
    if total_r[i] <= 1000:
        clas[0] = clas[0] + 1
    if total_r[i] > 1000 and total_r[i] <= 3000:
        clas[1] = clas[1] + 1
    if total_r[i] > 3000 and total_r[i] <= 5000:
        clas[2] = clas[2] + 1
    if total_r[i] > 50000 and total_r[i] <= 10000:
        clas[3] = clas[3] + 1
    if total_r[i] > 10000:
        clas[4] = clas[4] + 1
lab = '<1k','1k<X<3k','3k<X<5k','5k<X<10k','>10k'
plt.pyplot.pie(clas,labels = lab);
# Total de dormitórios

total_b = trainDB['total_bedrooms'] 
r_max = total_b.max()
r_min = total_b.min()
print('O maximo de quartos em uma regiao eh',r_max,'e o mínimo eh ',r_min)

# Classes division -> 0 - 300, 300 - 500, 500 - 1k, 1k - 2.5k, 2.5k - Max
clas = np.array([0,0,0,0,0])
for i in range(len(total_r)):
    if total_b[i] <= 300:
        clas[0] = clas[0] + 1
    if total_b[i] > 300 and total_b[i] <= 5000:
        clas[1] = clas[1] + 1
    if total_b[i] > 500 and total_b[i] <= 1000:
        clas[2] = clas[2] + 1
    if total_b[i] > 1000 and total_b[i] <= 2500:
        clas[3] = clas[3] + 1
    if total_b[i] > 2500:
        clas[4] = clas[4] + 1
lab = 'X<300','300<X<500','500<X<1k','1k<X<2.5k','2.5k<X'
plt.pyplot.pie(clas,labels = lab);
# Testando correlação entre total de dormitórios/total de cômodos

rel_rooms_bedrooms = total_r/total_b
rel_max = rel_rooms_bedrooms.max()
rel_min = rel_rooms_bedrooms.min()
print('Max = ',rel_max,' Min = ',rel_min)
values = trainDB['median_house_value']
# Observando a dispersão
plt.pyplot.scatter(rel_rooms_bedrooms,values);
# Relação entre renda média e valor da propriedade
med_in = trainDB['median_income']
# Observando a dispersão
plt.pyplot.scatter(med_in,values);
# Relação entre renda per capita e valor da propriedade
pc_income = med_in/trainDB['population']
# Observando a dispersão
plt.pyplot.scatter(pc_income,values);
# Escolhendo X e Y para os regressores
size_train = trainDB.shape
# Testando a melhores colunas para usar regressor knn
knn_trainX = trainDB[["longitude","total_rooms","total_bedrooms","population","households","median_income"]]
# Testando a melhores colunas para usar regressor lasso
lasso_trainX = trainDB[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]
# Testando a melhores colunas para usar regressor ridge
ridge_trainX = trainDB[["longitude","median_age","total_rooms","total_bedrooms","population","households","median_income"]]

trainY = trainDB.iloc[:,(size_train[1])-1]
from sklearn.metrics import mean_squared_error as mse
import math
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
# Regressão de KNN

from sklearn.neighbors import KNeighborsRegressor
# Escolha da vizinhança
neigh = KNeighborsRegressor(n_neighbors=2)
# Fitting
neigh.fit(knn_trainX,trainY) 
# Predict
knn_predict = neigh.predict(knn_trainX)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
# RMLSE (Estimativa do desempenho do regressor)
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
# Regressão de LASSO
warnings.filterwarnings("ignore")

from sklearn import linear_model
# Escolha do regressor
clf = linear_model.Lasso(alpha=0.5)
# fitting to data
clf.fit(lasso_trainX,trainY)
# predicting
lasso_predict = clf.predict(lasso_trainX)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
# inverting negative numbers
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
# RMSLE (Estimativa do desempenho do regressor)
print(rmsle(df_l.Y_real,df_l.Y_pred))
# Regressão de Ridge

from sklearn.linear_model import Ridge
# Escolhendo regressor
clf = Ridge(alpha=1.0)
# fitting to data
clf.fit(ridge_trainX,trainY) 
# predicting
ridge_predict = clf.predict(ridge_trainX)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
# RMSLE (Estimativa do desempenho do regressor)
print(rmsle(df_r.Y_real,df_r.Y_pred))

new_trainX = trainDB.iloc[:,1:9]
# renda per capita
new_trainX.insert(5,'per_capita_income',pc_income,allow_duplicates=False)
# Total de cômodos/total de dormitórios
new_trainX.insert(5,'total_rooms_total_bedrooms',rel_rooms_bedrooms,allow_duplicates=False)
# população por propriedade
p_p_h = new_trainX.population/new_trainX.households
new_trainX.insert(5,'people_per_house',p_p_h,allow_duplicates=False)
new_trainX.iloc[0:10,:]
# Select K Best e f_classif
from sklearn.feature_selection import SelectKBest,f_classif
# kNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Definindo selector para knn
selector = SelectKBest(score_func=f_classif, k=9)
# Treinando o selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids_knn = selector.get_support(indices = True)
knn_trainX_trans = new_trainX.iloc[:,ids_knn]
#KNN
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(knn_trainX_trans,trainY) 
knn_predict = neigh.predict(knn_trainX_trans)
df_knn = pd.DataFrame({'Y_real':trainY[:],'Y_pred':knn_predict[:]})
# Estimativa do desempenho do regressor
print(rmsle(df_knn.Y_real,df_knn.Y_pred))
# Definindo selector para Lasso
selector = SelectKBest(score_func=f_classif, k=5)
# Treinando o selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids = selector.get_support(indices = True)
lasso_trainX_trans = new_trainX.iloc[:,ids]
#Lasso
warnings.filterwarnings("ignore")
from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.5)
clf.fit(lasso_trainX_trans,trainY)
lasso_predict = clf.predict(lasso_trainX_trans)
df_l = pd.DataFrame({'Y_real':trainY[:],'Y_pred':lasso_predict[:]})
for i in range (len(df_l.Y_real)):
    if df_l.Y_pred[i] < 0:
        aux = df_l.Y_pred[i]*(-1)
        df_l.ix[i,'Y_pred'] = aux
# Estimativa do desempenho do regressor
print(rmsle(df_l.Y_real,df_l.Y_pred))
# Definindo selector para Ridge
selector = SelectKBest(score_func=f_classif, k=5)
# Trainindo o selector
trainX_select = selector.fit_transform(new_trainX,trainY)
ids = selector.get_support(indices = True)
ridge_trainX_trans = new_trainX.iloc[:,ids]
# RIDGE
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(ridge_trainX_trans,trainY) 
ridge_predict = clf.predict(ridge_trainX_trans)
df_r = pd.DataFrame({'Y_real':trainY[:],'Y_pred':ridge_predict[:]})
for i in range (len(df_r.Y_real)):
    if df_r.Y_pred[i] < 0:
        aux = df_r.Y_pred[i]*(-1)
        df_r.ix[i,'Y_pred'] = aux
# Estimativa do desempenho do regressor
print(rmsle(df_r.Y_real,df_r.Y_pred))
# Submission
testX = testDB.iloc[:,1:9]
# renda per capita
pc_income = testX.median_income/testX.population
testX.insert(5,'per_capita_income',pc_income,allow_duplicates=False)
# Total de dormitórios / Total de cômodos
rel_rooms_bedrooms = testX.total_bedrooms/testX.total_rooms
testX.insert(5,'total_rooms_total_bedrooms',rel_rooms_bedrooms,allow_duplicates=False)
# população por propriedade
p_p_h = testX.population/testX.households
testX.insert(5,'people_per_house',p_p_h,allow_duplicates=False)
# Regressor
knn_testX_trans = testX.iloc[:,ids_knn]
knn_predict = neigh.predict(knn_testX_trans)

submission_id =  testDB.Id
submission_pred = knn_predict
sub = pd.DataFrame({'Id':submission_id[:],'median_house_value':submission_pred[:]})
sub.to_csv('submission.csv', index = False)