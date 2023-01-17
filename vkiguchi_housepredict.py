import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from scipy import stats as st
import os
import matplotlib.colors as mcolors
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
trainfilepath = "../input/atividade-3-pmr3508/train.csv"
testfilepath = "../input/atividade-3-pmr3508/test.csv"
trainHouses = pd.read_csv(trainfilepath, sep=r'\s*,\s*', engine='python', na_values='?')
testHouses = pd.read_csv(testfilepath, sep=r'\s*,\s*', engine='python', na_values='?')
states = pd.read_csv("../input/averagestatecoordinates/states.csv", sep=r'\s*,\s*', engine='python', na_values='?')
states.head()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, p=2)

#knn.fit(states.drop("state", axis="columns"), states['state'])
knn.fit(states[["latitude", "longitude"]], states['state'])


#trainHouses["state"] = knn.predict(trainHouses.drop(trainHouses.columns.drop(["latitude", "longitude"]), axis="columns"))
#prediction = knn.predict(trainHouses.drop(trainHouses.columns.drop(["latitude", "longitude"]), axis="columns"))
prediction = knn.predict(trainHouses[["latitude", "longitude"]])

#print(prediction.value_counts())
print(np.unique(prediction,return_counts=True))
from numpy import radians, cos, sin, arcsin, sqrt
# Distance between two lat/lng coordinates in km using the Haversine formula
def getDistanceFromLatLng(lat1, lng1, lat2, lng2, miles=False): # use decimal degrees
    r=6371 # radius of the earth in km
    lat1=radians(lat1)
    lat2=radians(lat2)
    lat_dif=lat2-lat1
    lng_dif=radians(lng2-lng1)
    a=sin(lat_dif/2.0)**2+cos(lat1)*cos(lat2)*sin(lng_dif/2.0)**2
    d=2*r*arcsin(sqrt(a))
    return d
apple = (37.33182, -122.03118)
SF = (37.783333, -122.416667)

#trainHouses["distance_to_apple"] = np.sqrt((trainHouses["latitude"] - miami[0])**2 + (trainHouses["longitude"] - miami[1])**2)
#d_to_apple = np.sqrt((trainHouses["latitude"] - apple[0])**2 + (trainHouses["longitude"] - apple[1])**2)
d_to_apple = getDistanceFromLatLng(trainHouses["latitude"], trainHouses["longitude"], apple[0], apple[1])
#d_to_SF = np.sqrt((trainHouses["latitude"] - SF[0])**2 + (trainHouses["longitude"] - SF[1])**2)
d_to_SF = getDistanceFromLatLng(trainHouses["latitude"], trainHouses["longitude"], SF[0], SF[1])

plt.hist(d_to_apple, bins=100)
plt.show()
plt.hist2d(d_to_apple, trainHouses["median_house_value"], bins=100, norm=mcolors.PowerNorm(0.15))
plt.show()
plt.hist(d_to_SF, bins=100)
plt.show()
plt.hist2d(d_to_SF, trainHouses["median_house_value"], bins=100, norm=mcolors.PowerNorm(0.15))
plt.show()
LA = (35.0569, -118.25)
Beverly_Hills = (34.073056, -118.399444)

#d_to_LA = np.sqrt((trainHouses["latitude"] - LA[0])**2 + (trainHouses["longitude"] - LA[1])**2)
d_to_LA = getDistanceFromLatLng(trainHouses["latitude"], trainHouses["longitude"], LA[0], LA[1])
#d_to_BH = np.sqrt((trainHouses["latitude"] - Beverly_Hills[0])**2 + (trainHouses["longitude"] - Beverly_Hills[1])**2)
d_to_BH = getDistanceFromLatLng(trainHouses["latitude"], trainHouses["longitude"], Beverly_Hills[0], Beverly_Hills[1])

plt.hist(d_to_LA, bins=100)
plt.show()
plt.hist2d(d_to_LA, trainHouses["median_house_value"], bins=100, norm=mcolors.PowerNorm(0.15))
plt.show()

plt.hist(d_to_BH, bins=100)
plt.show()
plt.hist2d(d_to_BH, trainHouses["median_house_value"], bins=100, norm=mcolors.PowerNorm(0.15))
plt.show()
lat = states["latitude"][4]
long = states["longitude"][4]
#trainHouses["distance_to_state_center"] = np.sqrt((trainHouses["latitude"] - lat)**2 + (trainHouses["longitude"] - long)**2)
d_to_center = np.sqrt((trainHouses["latitude"] - lat)**2 + (trainHouses["longitude"] - long)**2)
d_to_center = getDistanceFromLatLng(trainHouses["latitude"], trainHouses["longitude"], lat, long)
plt.hist(d_to_center, bins=100)
plt.show()
plt.hist2d(d_to_center, trainHouses["median_house_value"], bins=100, norm=mcolors.PowerNorm(0.15))
plt.show()
trainHouses["distance_to_SF"] = d_to_SF
#testHouses["distance_to_SF"] = np.sqrt((testHouses["latitude"] - SF[0])**2 + (testHouses["longitude"] - SF[1])**2)
testHouses["distance_to_SF"] = getDistanceFromLatLng(testHouses["latitude"], testHouses["longitude"], SF[0], SF[1])
trainHouses["distance_to_LA"] = d_to_LA
#testHouses["distance_to_LA"] = np.sqrt((testHouses["latitude"] - LA[0])**2 + (testHouses["longitude"] - LA[1])**2)
testHouses["distance_to_LA"] = getDistanceFromLatLng(testHouses["latitude"], testHouses["longitude"], LA[0], LA[1])
trainHouses["distance_to_state_center"] = d_to_center
#testHouses["distance_to_state_center"] = np.sqrt((testHouses["latitude"] - lat)**2 + (testHouses["longitude"] - long)**2)
testHouses["distance_to_state_center"] = getDistanceFromLatLng(testHouses["latitude"], testHouses["longitude"], lat, long)
trainHouses["distance_to_beverly_hills"] = d_to_BH
testHouses["distance_to_beverly_hills"] = getDistanceFromLatLng(testHouses["latitude"], testHouses["longitude"], Beverly_Hills[0], Beverly_Hills[1])
trainHouses.shape
trainHouses.head()
plt.hist(trainHouses["median_house_value"], bins=100)
plt.show()
print(sorted(trainHouses["median_house_value"].value_counts(), reverse=True)[0])
trainHouses["oferta"] = trainHouses["households"]/trainHouses["population"]
testHouses["oferta"] = testHouses["households"]/testHouses["population"]
plt.hist2d(trainHouses["oferta"], trainHouses["median_house_value"], bins=120, norm=mcolors.PowerNorm(0.15))
plt.xlabel("oferta")
plt.show()

trainHouses["bedroom_per_room"] = trainHouses["total_bedrooms"]/trainHouses["total_rooms"]
testHouses["bedroom_per_room"] = testHouses["total_bedrooms"]/testHouses["total_rooms"]
plt.hist2d(trainHouses["bedroom_per_room"], trainHouses["median_house_value"], bins=120, norm=mcolors.PowerNorm(0.15))
plt.xlabel("bedroom_per_room")
plt.show()

trainHouses["riqueza"] = trainHouses["median_income"]*trainHouses["population"]
testHouses["riqueza"] = testHouses["median_income"]*testHouses["population"]
plt.hist2d(trainHouses["riqueza"], trainHouses["median_house_value"], bins=120, norm=mcolors.PowerNorm(0.15))
plt.xlabel("riqueza")
plt.show()
tfeatures = trainHouses.columns.drop(["Id", "median_house_value"])
for label in tfeatures:
    #plt.scatter(trainHouses[label], trainHouses["median_house_value"].transform(np.log))
    plt.hist2d(trainHouses[label], trainHouses["median_house_value"], bins=120, norm=mcolors.PowerNorm(0.15))
    plt.xlabel(label)
    plt.show()

trainHouses.min()
trainHouses.max()
print(trainHouses.corr()["median_house_value"])
XtrainHouses = trainHouses.drop(["Id", "median_house_value"], axis="columns")
XtrainHousesB = trainHouses.drop(["Id", "median_house_value", "median_age", "total_rooms", "total_bedrooms", "population", "households"], axis="columns")
YtrainHouses = trainHouses["median_house_value"]

XtestHouses = testHouses.drop("Id", axis="columns")
XtestHousesB = testHouses.drop(["Id", "median_age", "total_rooms", "total_bedrooms", "population", "households"], axis="columns")
#a = trainHouses[trainHouses["median_house_value"].transform(lambda x: x<=500000)]
#a = trainHouses[trainHouses["median_house_value"] <=500000]
plt.hist(trainHouses[trainHouses["median_house_value"] <=500000]["median_house_value"], bins=100)
plt.show()
culledTrainHouses = trainHouses[trainHouses["median_house_value"] <=500000]
XculledTrainHouses = culledTrainHouses.drop(["Id", "median_house_value"], axis="columns")
YculledTrainHouses = culledTrainHouses.median_house_value

#RMSLE = SQRT((1/n)*Sum((log(1+Y_hat)-log(Y + 1))^2)
def rmsle(real, predicted):
    summ=0.0
    for x in range(len(predicted)):
        if(predicted[x]<0 or real[x]<0): #check for negative values
            print("Bad Value")
            raise ValueError
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        summ = summ + (p - r)*(p - r)
    return np.sqrt(summ/len(predicted))
def rmsle2(real, predicted):
    summ = sum((np.log(real[x]+1)-np.log(predicted[x]+1))**2 for x in range(len(predicted)))
    return np.sqrt(summ/len(predicted))
msle = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_log_error, greater_is_better=False)
rmsle_scorer = sklearn.metrics.make_scorer(rmsle, greater_is_better=False)
def rmsle_score(estimador, x, y):
    y_pred = estimador.predict(x)
    #[print("pred", a) for a in y_pred if a<0]
    #y_pred = [max(a,0) for a in y_pred]
    #y_pred = [np.log(1+abs(a)) if a<0 else a for a in y_pred]
    #y_pred = [abs(a) for a in y_pred]
    y_pred = [max(15000, a) for a in y_pred]
    return np.sqrt(sklearn.metrics.mean_squared_log_error(y, y_pred))
    #return rmsle(y, y_pred)

from itertools import combinations

#model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=True))])
model = DecisionTreeRegressor(max_depth=20 ,min_samples_split=12, random_state=0)
linear = LinearRegression()

best = None
best_linear = None
#best_score = cross_val_score(model, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score).mean()
best_score = 50
best_linear_score = 50
#nums = list(range(1, 4))
#nums = nums + (list(range(len(XtrainHouses.columns)-3, len(XtrainHouses.columns)+1)))
nums = range(1, len(XtrainHouses.columns))
print(nums)
for r in nums:
    print("combinations with ", r, " features:")
    for x in combinations(XtrainHouses.columns, r):
        scores = cross_val_score(model, XtrainHouses[list(x)], YtrainHouses, cv=10, scoring=rmsle_score)
        scores_l = cross_val_score(linear, XtrainHouses[list(x)], YtrainHouses, cv=10, scoring=rmsle_score)
        score = scores.mean()
        score_l = scores_l.mean()
        if(score < best_score):
            print("Hello")
            print(x, score)
            best = x
            best_score = score
        if(score_linear < best_linear_score):
            print("Hello2")
            print(x, score_l)
            best_linear = x
            best_linear_score = score_l
print(best)
print(best_score)
print(best_linear)
print(best_linear_score)
from sklearn.linear_model import LinearRegression
# Create linear regression object
lregr = LinearRegression()

scores = cross_val_score(lregr, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XculledTrainHouses, YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XtrainHouses.drop(["households"], axis='columns'), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XculledTrainHouses.drop(["households"], axis='columns'), YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XtrainHousesB, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(lregr, XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

#lregr.fit(XtrainHouses, YtrainHouses)
#lregr.fit(XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses)
lregr.fit(XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses)


#housePredict = lregr.predict(XtestHouses)
#housePredict = lregr.predict(XtestHouses.drop(XtestHouses.columns.drop("median_income"), axis="columns"))
housePredict = lregr.predict(XtestHouses[['longitude', 'latitude', 'median_income']])

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: max(min(x, 500000), 15000))
houseLinearPredict = predictHouses["median_house_value"]

plt.scatter(XtrainHouses["median_income"], YtrainHouses,  color='black')
#plt.plot(XtrainHouses["median_income"], lregr.predict(XtrainHouses), color='blue', linewidth=3)
tpred = lregr.predict(XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"))
plt.plot(XtrainHouses["median_income"], tpred, color='blue', linewidth=3)
plt.plot(XtrainHouses["median_income"], list(map(lambda x: min(x, 500000),tpred)), color='red', linewidth=3)


savepath = "houseLinearPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=True))])

scores = cross_val_score(model, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XculledTrainHouses, YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses.drop(["households"], axis='columns'), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHousesB, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses[["median_income", "oferta", "riqueza", "distance_to_state_center"]], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

#model.fit(XtrainHouses, YtrainHouses)
#model.fit(XtrainHouses.drop(["households"], axis='columns'), YtrainHouses)
model.fit(XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses)
#model.fit(XtrainHousesB, YtrainHouses)
#model.fit(XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses)

#housePredict = model.predict(XtestHouses)
#housePredict = model.predict(XtestHouses.drop(["households"], axis='columns'))
housePredict = model.predict(XtestHouses[['longitude', 'latitude', 'median_income']])
#housePredict = model.predict(XtestHousesB)
#housePredict = model.predict(XtestHouses.drop(XtestHouses.columns.drop("median_income"), axis="columns"))

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseCubicPredict = predictHouses["median_house_value"]

savepath = "houseCubicPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=True))])
baggedModel = BaggingRegressor(model, n_estimators=75, bootstrap=True, bootstrap_features=True)

scores = cross_val_score(baggedModel, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(baggedModel, XculledTrainHouses, YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(baggedModel, XtrainHousesB, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(baggedModel, XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(baggedModel, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())


#baggedModel.fit(XtrainHousesB, YtrainHouses)
baggedModel.fit(XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses)

#housePredict = baggedModel.predict(XtestHousesB)
housePredict = baggedModel.predict(XtestHouses[['longitude', 'latitude', 'median_income']])

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
from sklearn.linear_model import SGDRegressor

sgdreg = SGDRegressor(loss='huber',penalty='l1', epsilon=0.05, max_iter=1000, tol=1e-4, learning_rate="optimal", early_stopping=True)


scores = cross_val_score(sgdreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

sgdreg.fit(XtrainHouses, YtrainHouses)

housePredict = sgdreg.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseSGDRpredict = predictHouses["median_house_value"]


savepath = "houseSGDRpredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV


lassoreg = Lasso()
lassolarscvreg = LassoLarsCV(cv=10)
lassocvreg = LassoCV(cv=10)


scores = cross_val_score(lassoreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(lassolarscvreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(lassocvreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())


#lassoreg.fit(XtrainHouses, YtrainHouses)
lassocvreg.fit(XtrainHouses, YtrainHouses)

#housePredict = lassoreg.predict(XtestHouses)
housePredict = lassocvreg.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseLassoPredict = predictHouses["median_house_value"]


savepath = "houseLassoPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

enetreg = ElasticNet()
enetcvreg = ElasticNetCV(cv=10)

scores = cross_val_score(enetreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(enetreg, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(enetcvreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())


#enetreg.fit(XtrainHouses, YtrainHouses)
enetcvreg.fit(XtrainHouses, YtrainHouses)

#housePredict = enetreg.predict(XtestHouses)
housePredict = enetcvreg.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseElasticNetPredict = predictHouses["median_house_value"]


savepath = "houseElasticNetPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

rnreg = RadiusNeighborsRegressor(radius=1.96, weights="distance")
#rnreg = RadiusNeighborsRegressor(radius=3, weights="distance")
#rnreg = RadiusNeighborsRegressor(radius=1.96)
#gsearch = GridSearchCV(RadiusNeighborsRegressor(), param_grid={"radius":[10/x for x in range(1,50)]}, scoring=rmsle_score, cv=10)

SS = StandardScaler()

normXtrainHouses = XtrainHouses.transform(lambda x: (x-x.mean())/x.std())
scaledXtrainHouses = SS.fit_transform(XtrainHouses)
#scores = cross_val_score(rnreg, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
scores = cross_val_score(rnreg, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(rnreg, normXtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(rnreg, scaledXtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
#scores = cross_val_score(gsearch, scaledXtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
#print(scores.mean())

#rnreg.fit(XtrainHouses, YtrainHouses)
#rnreg.fit(normXtrainHouses, YtrainHouses)
rnreg.fit(scaledXtrainHouses, YtrainHouses)
#gsearch.fit(scaledXtrainHouses, YtrainHouses)
#print(gsearch.cv_results_)
#gsearch.score()

#normXtestHouses = XtestHouses.transform(lambda x: (x-x.mean())/x.std())
scaledXtestHouses = SS.transform(XtestHouses)
#housePredict = rnreg.predict(normXtestHouses)
housePredict = rnreg.predict(scaledXtestHouses)
#housePredict = rnreg.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseRadiusPredict = predictHouses["median_house_value"]


savepath = "houseRadiusPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 17, weights="distance")
#knn = KNeighborsRegressor(n_neighbors = 6)
#knn = KNeighborsRegressor(n_neighbors = 15)

scores = cross_val_score(knn, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
#scores = cross_val_score(knn, normXtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())
scores = cross_val_score(knn, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())


#knn.fit(XtrainHouses, YtrainHouses)
knn.fit(normXtrainHouses, YtrainHouses)


#housePredict = knn.predict(normXtestHouses)
housePredict = knn.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseKNNpredict = predictHouses["median_house_value"]


savepath = "houseKNNpredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
tree = DecisionTreeRegressor(max_depth=50, min_samples_split=12, min_samples_leaf=2)
model = BaggingRegressor(tree, n_estimators=120, bootstrap=True, bootstrap_features=True)

scores = cross_val_score(model, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses.drop(["households"], axis='columns'), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XculledTrainHouses.drop(["households"], axis='columns'), YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(model, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

model.fit(XculledTrainHouses.drop(["households"], axis='columns'), YculledTrainHouses)

housePredict = model.predict(XtestHouses.drop(["households"], axis='columns'))

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseBaggedTreePredict = predictHouses["median_house_value"]

savepath = "houseBaggedTreePredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
forest = RandomForestRegressor(n_estimators=200, max_depth=50, min_samples_split=12, random_state=0, min_samples_leaf=2)
#forest = RandomForestRegressor(n_estimators=200, max_depth=50, min_samples_split=12, random_state=0, max_features="sqrt", min_samples_leaf=2)


#scores = cross_val_score(forest, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
#print(scores.mean())

#scores = cross_val_score(forest, XtrainHouses.drop(["households"], axis='columns'), YtrainHouses, cv=10, scoring=rmsle_score)
#print(scores.mean())

scores = cross_val_score(forest, XtrainHouses[["median_income", "oferta", "riqueza", "distance_to_state_center"]], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(forest, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(forest, XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

#forest.fit(XtrainHouses, YtrainHouses)
forest.fit(XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses)

#housePredict = forest.predict(XtestHouses)
housePredict = forest.predict(XtestHouses[['longitude', 'latitude', 'median_income']])


predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseForestPredict = predictHouses["median_house_value"]

savepath = "houseForestPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
from sklearn.ensemble import ExtraTreesRegressor

extraT = ExtraTreesRegressor(n_estimators=200, max_depth=150, min_samples_split=12, random_state=0, min_samples_leaf=2)

scores = cross_val_score(extraT, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(extraT, XculledTrainHouses, YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(extraT, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(extraT, XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

extraT.fit(XculledTrainHouses, YculledTrainHouses)
#extraT.fit(XtrainHouses.drop(["households"], axis='columns'), YtrainHouses)

housePredict = extraT.predict(XtestHouses)
#housePredict = extraT.predict(XtestHouses.drop(["households"], axis='columns'))


predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
houseExtraTreesPredict = predictHouses["median_house_value"]

savepath = "houseExtraTreesPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
#boosted = GradientBoostingRegressor(subsample=0.85, n_estimators=320, max_depth=2)
boosted = GradientBoostingRegressor(loss='huber', subsample=0.85, n_estimators=200, max_depth=2, alpha=0.25)
#boosted = GradientBoostingRegressor(loss='huber', subsample=0.85, n_estimators=5250, max_depth=2, alpha=0.25)
#boosted = GradientBoostingRegressor(loss='lad', subsample=0.85, n_estimators=320, max_depth=2)
#boosted = GradientBoostingRegressor(loss='quantile', subsample=0.85, n_estimators=320, max_depth=2, alpha=0.9)

scores = cross_val_score(boosted, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(boosted, XculledTrainHouses, YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(boosted, XtrainHouses.drop(["households"], axis='columns'), YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(boosted, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

boosted.fit(XtrainHouses, YtrainHouses)

housePredict = boosted.predict(XtestHouses)

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
GradientForestPredict = predictHouses["median_house_value"]

savepath = "GradientForestPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=True))])
boosted = AdaBoostRegressor(base_estimator=model,n_estimators=50)

scores = cross_val_score(boosted, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(boosted, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(boosted, XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

boosted.fit(XculledTrainHouses[['longitude', 'latitude', 'median_income']], YculledTrainHouses)

housePredict = boosted.predict(XtestHouses[['longitude', 'latitude', 'median_income']])

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: min(max(x, 15000), 500000))
AdaBoostForestPredict = predictHouses["median_house_value"]

savepath = "AdaBoostForestPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
neural = MLPRegressor(hidden_layer_sizes=(128,100, 50), learning_rate_init=0.006, max_iter=300, activation="relu", early_stopping=True)

scores = cross_val_score(neural, XtrainHouses, YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

scores = cross_val_score(neural, XtrainHouses[['longitude', 'latitude', 'median_income']], YtrainHouses, cv=10, scoring=rmsle_score)
print(scores.mean())

#scores = cross_val_score(neural, XtrainHousesB, YtrainHouses, cv=10, scoring=rmsle_score)
#print(scores.mean())

#scores = cross_val_score(neural, XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses, cv=10, scoring=rmsle_score)
#print(scores.mean())

neural.fit(XtrainHouses, YtrainHouses)
#neural.fit(XtrainHousesB, YtrainHouses)
#neural.fit(XtrainHouses.drop(XtrainHouses.columns.drop("median_income"), axis="columns"), YtrainHouses)

housePredict = neural.predict(XtestHouses)
#housePredict = neural.predict(XtestHousesB)
#housePredict = neural.predict(XtestHouses.drop(XtestHouses.columns.drop("median_income"), axis="columns"))

predictHouses = testHouses.apply(lambda x:x)
predictHouses["median_house_value"] = housePredict
predictHouses["median_house_value"] = predictHouses["median_house_value"].transform(lambda x: max(min(x, 500000), 0))
houseCubicPredict = predictHouses["median_house_value"]

savepath = "houseNeuralPredict.csv"
predictHouses.to_csv(savepath, index=False, columns=['Id', 'median_house_value'])
