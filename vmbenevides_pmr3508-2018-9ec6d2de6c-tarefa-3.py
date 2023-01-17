from sklearn.metrics import make_scorer
import pandas as pd
import math
import scipy
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
train = pd.read_csv ("../input/california/train.csv")
train.shape 
train.head ()
train = pd.read_csv ("../input/california/train.csv")
#________________________________________________________________________________________________________________________________
# REGIÕES CARAS

#Vale do Silício
vale_silicio = {'latitude':37.84, 'longitude':-122.05}
train['dist_vs'] = (((train['longitude']-vale_silicio['longitude'])**2+(train['latitude']-vale_silicio['latitude'])**2)**(1/2))

#Beverly Hills
beverly_hills = {'latitude':33.933333, 'longitude':-118.4}
train['dist_bh'] = (((train['longitude']-beverly_hills['longitude'])**2+(train['latitude']-beverly_hills['latitude'])**2)**(1/2))

#Corona Del Mar
corona_del_mar = {'latitude':33.5978595,'longitude':-117.8730142}
train['dist_cdm'] = (((train['longitude']-corona_del_mar['longitude'])**2+(train['latitude']-corona_del_mar['latitude'])**2)**(1/2))

#Montecito
montecito = {'latitude':34.43666 ,'longitude':-119.63208}
train['dist_mt'] = (((train['longitude']-montecito['longitude'])**2+(train['latitude']-montecito['latitude'])**2)**(1/2))

#Atherton
atherton = {'latitude':37.458611,'longitude': -122.2}
train['dist_at'] = (((train['longitude']-atherton['longitude'])**2+(train['latitude']-atherton['latitude'])**2)**(1/2))

#Woodside
wdside = {'latitude':37.420833, 'longitude':-122.259722}
train['dist_wd'] = (((train['longitude']-wdside['longitude'])**2+(train['latitude']-wdside['latitude'])**2)**(1/2))

#Hidden Hills
hdhills = {'latitude': 34.1675, 'longitude':-118.660833}
train['dist_hh'] = (((train['longitude']-hdhills['longitude'])**2+(train['latitude']-hdhills['latitude'])**2)**(1/2))

#Los Altos
latos = {'latitude':37.368056, 'longitude':-122.0975}
train['dist_lal'] = (((train['longitude']-latos['longitude'])**2+(train['latitude']-latos['latitude'])**2)**(1/2))

#Belvedere
bvde = {'latitude':37.872778, 'longitude':-122.464444}
train['dist_bvd'] = (((train['longitude']-bvde['longitude'])**2+(train['latitude']-bvde['latitude'])**2)**(1/2))

#Santa monica
stm = {'latitude':34.021944, 'longitude': -118.481389}
train['dist_stm'] = (((train['longitude']-stm['longitude'])**2+(train['latitude']-stm['latitude'])**2)**(1/2))

#Ross
ross = {'latitude':37.9625,'longitude': -122.555}
train['dist_ros'] = (((train['longitude']-ross['longitude'])**2+(train['latitude']-ross['latitude'])**2)**(1/2))

#Fremont
fremont = {'latitude':37.548333, 'longitude':-121.988611}
train['dist_frm'] = (((train['longitude']-fremont['longitude'])**2+(train['latitude']-fremont['latitude'])**2)**(1/2))

#HILLSBOROUGH
hbr = {'latitude':37.560278, 'longitude': -122.356389}
train['dist_hbr'] = (((train['longitude']-hbr['longitude'])**2+(train['latitude']-hbr['latitude'])**2)**(1/2))

#Newport beach
npb = {'latitude':33.616667, 'longitude':-117.8975}
train['dist_npb'] = (((train['longitude']-npb['longitude'])**2+(train['latitude']-npb['latitude'])**2)**(1/2))

#Pelican Hill
plh = {'latitude':33.5901144,'longitude':-117.8445537}
train['dist_plh'] = (((train['longitude']-plh['longitude'])**2+(train['latitude']-plh['latitude'])**2)**(1/2))


#____________________________________________________________________________________________________________________________________________
#REGIÕES BARATAS

#Blythe
blt = {'latitude': 33.610278, 'longitude': -114.596389}
train['dist_blt'] = (((train['longitude']-blt['longitude'])**2+(train['latitude']-blt['latitude'])**2)**(1/2))

#Twentynine Palms
tnp = {'latitude':34.138333, 'longitude': -116.0725}
train['dist_tnp'] = (((train['longitude']-tnp['longitude'])**2+(train['latitude']-tnp['latitude'])**2)**(1/2))

#Arvin
avn = {'latitude':35.209167, 'longitude': -118.828333}
train['dist_avn'] = (((train['longitude']-avn['longitude'])**2+(train['latitude']-avn['latitude'])**2)**(1/2))

#Woodlake
wdl = {'latitude':36.416389, 'longitude':-119.099444}
train['dist_wdl'] = (((train['longitude']-wdl['longitude'])**2+(train['latitude']-wdl['latitude'])**2)**(1/2))

#Parlier
plr = {'latitude':36.611667, 'longitude':-119.526944}
train['dist_plr'] = (((train['longitude']-plr['longitude'])**2+(train['latitude']-plr['latitude'])**2)**(1/2))

#California City
ccc =  {'latitude': 35.125833, 'longitude': -117.985833}
train['dist_ccc'] = (((train['longitude']-ccc['longitude'])**2+(train['latitude']-ccc['latitude'])**2)**(1/2))

#Farmersville
fmv = {'latitude':36.301111, 'longitude': -119.2075}
train['dist_fmv'] = (((train['longitude']-fmv['longitude'])**2+(train['latitude']-fmv['latitude'])**2)**(1/2))

#Lindsay
lds = {'latitude':36.2, 'longitude':-119.083333}
train['dist_lds'] = (((train['longitude']-lds['longitude'])**2+(train['latitude']-lds['latitude'])**2)**(1/2))

#Wasco
wsc = {'latitude':35.594167, 'longitude':-119.340833}
train['dist_wsc'] = (((train['longitude']-wsc['longitude'])**2+(train['latitude']-wsc['latitude'])**2)**(1/2))

#Corcoran
ccn = {'latitude':36.098056, 'longitude':-119.560278}
train['dist_ccn'] = (((train['longitude']-ccn['longitude'])**2+(train['latitude']-ccn['latitude'])**2)**(1/2))

#_________________________________________________________________________________________________________________________________
#Colocando o target no final do dataframe
a = train['median_house_value']
train = train.drop (columns = ['median_house_value'])
train['median_house_value'] = a
train.head ()
plt.matshow(train.corr())
train.corr().iloc[1:-1, -1].plot('bar')
abs(train.corr().iloc[1:-1, -1]).plot('bar')
test = pd.read_csv ("../input/california/test.csv")
test.head ()
test = pd.read_csv ("../input/california/test.csv")
#________________________________________________________________________________________________________________________________
# REGIÕES CARAS

#Vale do Silício
vale_silicio = {'latitude':37.84, 'longitude':-122.05}
test['dist_vs'] = (((test['longitude']-vale_silicio['longitude'])**2+(test['latitude']-vale_silicio['latitude'])**2)**(1/2))

#Beverly Hills
beverly_hills = {'latitude':33.933333, 'longitude':-118.4}
test['dist_bh'] = (((test['longitude']-beverly_hills['longitude'])**2+(test['latitude']-beverly_hills['latitude'])**2)**(1/2))

#Corona Del Mar
corona_del_mar = {'latitude':33.5978595,'longitude':-117.8730142}
test['dist_cdm'] = (((test['longitude']-corona_del_mar['longitude'])**2+(test['latitude']-corona_del_mar['latitude'])**2)**(1/2))

#Montecito
montecito = {'latitude':34.43666 ,'longitude':-119.63208}
test['dist_mt'] = (((test['longitude']-montecito['longitude'])**2+(test['latitude']-montecito['latitude'])**2)**(1/2))

#Atherton
atherton = {'latitude':37.458611,'longitude': -122.2}
test['dist_at'] = (((test['longitude']-atherton['longitude'])**2+(test['latitude']-atherton['latitude'])**2)**(1/2))

#Woodside
wdside = {'latitude':37.420833, 'longitude':-122.259722}
test['dist_wd'] = (((test['longitude']-wdside['longitude'])**2+(test['latitude']-wdside['latitude'])**2)**(1/2))

#Hidden Hills
hdhills = {'latitude': 34.1675, 'longitude':-118.660833}
test['dist_hh'] = (((test['longitude']-hdhills['longitude'])**2+(test['latitude']-hdhills['latitude'])**2)**(1/2))

#Los Altos
latos = {'latitude':37.368056, 'longitude':-122.0975}
test['dist_lal'] = (((test['longitude']-latos['longitude'])**2+(test['latitude']-latos['latitude'])**2)**(1/2))

#Belvedere
bvde = {'latitude':37.872778, 'longitude':-122.464444}
test['dist_bvd'] = (((test['longitude']-bvde['longitude'])**2+(test['latitude']-bvde['latitude'])**2)**(1/2))

#Santa monica
stm = {'latitude':34.021944, 'longitude': -118.481389}
test['dist_stm'] = (((test['longitude']-stm['longitude'])**2+(test['latitude']-stm['latitude'])**2)**(1/2))

#Ross
ross = {'latitude':37.9625,'longitude': -122.555}
test['dist_ros'] = (((test['longitude']-ross['longitude'])**2+(test['latitude']-ross['latitude'])**2)**(1/2))

#Fremont
fremont = {'latitude':37.548333, 'longitude':-121.988611}
test['dist_frm'] = (((test['longitude']-fremont['longitude'])**2+(test['latitude']-fremont['latitude'])**2)**(1/2))

#HILLSBOROUGH
hbr = {'latitude':37.560278, 'longitude': -122.356389}
test['dist_hbr'] = (((test['longitude']-hbr['longitude'])**2+(test['latitude']-hbr['latitude'])**2)**(1/2))

#Newport beach
npb = {'latitude':33.616667, 'longitude':-117.8975}
test['dist_npb'] = (((test['longitude']-npb['longitude'])**2+(test['latitude']-npb['latitude'])**2)**(1/2))

#Pelican Hill
plh = {'latitude':33.5901144,'longitude':-117.8445537}
test['dist_plh'] = (((test['longitude']-plh['longitude'])**2+(test['latitude']-plh['latitude'])**2)**(1/2))


#____________________________________________________________________________________________________________________________________________
#REGIÕES BARATAS

#Blythe
blt = {'latitude': 33.610278, 'longitude': -114.596389}
test['dist_blt'] = (((test['longitude']-blt['longitude'])**2+(test['latitude']-blt['latitude'])**2)**(1/2))

#Twentynine Palms
tnp = {'latitude':34.138333, 'longitude': -116.0725}
test['dist_tnp'] = (((test['longitude']-tnp['longitude'])**2+(test['latitude']-tnp['latitude'])**2)**(1/2))

#Arvin
avn = {'latitude':35.209167, 'longitude': -118.828333}
test['dist_avn'] = (((test['longitude']-avn['longitude'])**2+(test['latitude']-avn['latitude'])**2)**(1/2))

#Woodlake
wdl = {'latitude':36.416389, 'longitude':-119.099444}
test['dist_wdl'] = (((test['longitude']-wdl['longitude'])**2+(test['latitude']-wdl['latitude'])**2)**(1/2))

#Parlier
plr = {'latitude':36.611667, 'longitude':-119.526944}
test['dist_plr'] = (((test['longitude']-plr['longitude'])**2+(test['latitude']-plr['latitude'])**2)**(1/2))

#California City
ccc =  {'latitude': 35.125833, 'longitude': -117.985833}
test['dist_ccc'] = (((test['longitude']-ccc['longitude'])**2+(test['latitude']-ccc['latitude'])**2)**(1/2))

#Farmersville
fmv = {'latitude':36.301111, 'longitude': -119.2075}
test['dist_fmv'] = (((test['longitude']-fmv['longitude'])**2+(test['latitude']-fmv['latitude'])**2)**(1/2))

#Lindsay
lds = {'latitude':36.2, 'longitude':-119.083333}
test['dist_lds'] = (((test['longitude']-lds['longitude'])**2+(test['latitude']-lds['latitude'])**2)**(1/2))

#Wasco
wsc = {'latitude':35.594167, 'longitude':-119.340833}
test['dist_wsc'] = (((test['longitude']-wsc['longitude'])**2+(test['latitude']-wsc['latitude'])**2)**(1/2))

#Corcoran
ccn = {'latitude':36.098056, 'longitude':-119.560278}
test['dist_ccn'] = (((test['longitude']-ccn['longitude'])**2+(test['latitude']-ccn['latitude'])**2)**(1/2))

test.head ()
train.columns[1:34]
def RMSLE(Y, Ypred):
    n = len(Y)
    soma = 0
    Y = np.array(Y)
    for i in range(len(Y)):
        soma += ( math.log( abs(Ypred[i]) + 1 ) - math.log( Y[i] + 1 ) )**2
    return math.sqrt(soma / n)
scorer_rmsle = make_scorer(RMSLE)
Xtrain = train.iloc[:,1:34]
Ytrain = train.median_house_value
Xtest = test.iloc[:,1:34]
reglin = linear_model.LinearRegression()
reglin.fit (Xtrain, Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (reglin,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)

Ytest_pred1 = reglin.predict (Xtest)
for i in Ytest_pred1:
    if i < 0:
        Ytest_pred1 = abs(Ytest_pred1)
result1 = np.vstack((test['Id'], Ytest_pred1)).T.astype(int)
x1 = ["Id","median_house_value"]
Resultado1 = pd.DataFrame(columns = x1, data = result1)
ridge = linear_model.Ridge ()
ridge.fit (Xtrain,Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (ridge,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred2 = ridge.predict (Xtest)
Ytest_pred2 = abs(Ytest_pred2)
result2 = np.vstack((test['Id'], Ytest_pred2)).T.astype(int)
x2 = ["Id","median_house_value"]
Resultado2 = pd.DataFrame(columns = x2, data = result2)
lasso = linear_model.Lasso()
lasso.fit (Xtrain, Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (lasso,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred3 = ridge.predict (Xtest)
Ytest_pred3 = abs(Ytest_pred3)
result3 = np.vstack((test['Id'], Ytest_pred3)).T.astype(int)
x3 = ["Id","median_house_value"]
Resultado3 = pd.DataFrame(columns = x3, data = result3)
arvore = tree.DecisionTreeRegressor ()
arvore.fit (Xtrain,Ytrain)
scores = cross_val_score (arvore,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred4 = arvore.predict (Xtest)

result4 = np.vstack((test['Id'], Ytest_pred4)).T.astype(int)
x4 = ["Id","median_house_value"]
Resultado4 = pd.DataFrame(columns = x4, data = result4)
bot = BaggingRegressor(n_estimators = 200)
bot.fit (Xtrain,Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (bot,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred5 = bot.predict (Xtest)

result5 = np.vstack((test['Id'], Ytest_pred5)).T.astype(int)
x5 = ["Id","median_house_value"]
Resultado5 = pd.DataFrame(columns = x5, data = result5)
rdf = RandomForestRegressor (n_estimators = 200, max_features = 'log2', min_samples_leaf = 1)
rdf.fit (Xtrain, Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (rdf,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred6 = rdf.predict (Xtest)

result6 = np.vstack((test['Id'], Ytest_pred6)).T.astype(int)
x6 = ["Id","median_house_value"]
Resultado = pd.DataFrame(columns = x6, data = result6)
Resultado.to_csv("resultados_rdf.csv", index = False)
etr = ExtraTreesRegressor (n_estimators = 200)
etr.fit (Xtrain,Ytrain)
total = 0
c_val = 10 
scores = cross_val_score (etr,Xtrain,Ytrain, cv = c_val,scoring = scorer_rmsle)
total = 0
for j in scores:
    total += j
acuracia_esperada = total/c_val
print (acuracia_esperada)
Ytest_pred7 = rdf.predict (Xtest)

result7 = np.vstack((test['Id'], Ytest_pred7)).T.astype(int)
x7 = ["Id","median_house_value"]
Resultado = pd.DataFrame(columns = x7, data = result7)