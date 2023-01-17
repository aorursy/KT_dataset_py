# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model, metrics, model_selection, preprocessing, multiclass, svm
from scipy import interp
import os, datetime , time, sys
from itertools import cycle

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sha_pek = pd.read_csv('/kaggle/input/air-tickets-between-shanghai-and-beijing/sha-pek.csv')
sha_pek.head(5)
sha_pek['createDate'] = pd.to_datetime(sha_pek['createDate'])
sha_pek['departureDate'] = pd.to_datetime(sha_pek['departureDate'])
sha_pek_invalid =  sha_pek[((sha_pek['departureDate']-sha_pek['createDate'])/np.timedelta64(1,'s'))<0]
sha_pek_invalid.head(5)
sha_pek_valid = sha_pek[((sha_pek['departureDate']-sha_pek['createDate'])/np.timedelta64(1,'s'))>0]
min((sha_pek_valid['departureDate']-sha_pek_valid['createDate'])/np.timedelta64(1,'s'))
#EDA of whether the price column is actually the discounted price.

singleFlight_Filter = sha_pek_valid.loc[sha_pek_valid.flightNumber.isin(['MU5389']) & 
                  sha_pek_valid.cabinClass.isin(['C']) & 
                  sha_pek_valid.departureDate.isin(['2019-07-21 07:20:00'])]

singleFlight_Filter
#CALCULATING DEPARTURE TIME FROM THE DEPARTURE DATE COLUMN
sha_pek_valid['departureTime'] =pd.to_datetime(sha_pek_valid['departureDate']).dt.strftime('%H:%M')

#SELECTING NECESSARY COLUMNS FROM THE VALID DATASET
tabular_subset = sha_pek_valid[['flightNumber','departureTime','cabinClass','price','rate']]

#DETERMINING ORIGINAL PRICE OF EACH FLIGHT WHERE RATE = 1.0 which means no discount
tabular_subset = sha_pek_valid.loc[sha_pek_valid['rate']==1.0]

#FINAL OUTPUT
tabular_subset[['flightNumber','departureTime','cabinClass','price']]
##MAKING TIME EXECUTABLE FOR REGRESSION

tabular_subset['departureTime']=tabular_subset['departureTime'].str.replace(':','.')

#DETERMINING THE FLIGHT NAME BY TAKING THE FIRST TWO LETTERS ACCORDING TO THEIR NAMING CONVENTION
tabular_subset['flightType']=tabular_subset['flightNumber'].str.slice(0,2)

#GENERATING GRID PLOTS ACCORDING TO DIFFERENT FLIGHT NAMES COMPUTED FROM ABOVE

airlines = tabular_subset.flightType.to_list()
airlines = set(airlines)

fig = plt.figure(figsize=(100,100))

for al,num in zip(airlines,range(1,7)):
    al_flights = tabular_subset.loc[tabular_subset['flightType']==al].sort_values(by=['departureTime'])
    ax=fig.add_subplot(20,20,num)
    ax.plot(al_flights['departureTime'], al_flights['price'])
    ax.set_title(al)
    
plt.show()
#Linear Regression for Q5 (DEPARTURE TIME vs PRICE)

x=tabular_subset[['departureTime']].values
y=tabular_subset['price'].values
rm = linear_model.LinearRegression()
rm.fit(x,y)
sst = np.sum((y-np.mean(y))**2)
ssr = np.sum((rm.predict(x)-np.mean(y))**2)
sse = np.sum((rm.predict(x)-y)**2)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Total Sum of Squares is: ', sst)
print('The Residual Sum of Squares is: ',sse)
print('The Explained Sum of Squares is: ', ssr)
print('The R^2 from regressor: ', rm.score(x,y))
print('The R^2 from ssr/sst: ', ssr/sst)
#Plotting datedifference vs rate

cabins = sha_pek_valid.cabinClass.to_list()
cabins = set(cabins)

fig, ax = plt.subplots()

sha_pek_valid.plot(x='dateDifference',y='rate',ax=ax)
plt.show()
#Linear Regression Date Difference vs Rate

x=sha_pek_valid[['dateDifference']].values
y=sha_pek_valid['rate'].values
rm = linear_model.LinearRegression()
rm.fit(x,y)
sst = np.sum((y-np.mean(y))**2)
ssr = np.sum((rm.predict(x)-np.mean(y))**2)
sse = np.sum((rm.predict(x)-y)**2)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Coefficient is: ', rm.coef_)
print('The Intercept is: ', rm.intercept_)
print('The Total Sum of Squares is: ', sst)
print('The Residual Sum of Squares is: ',sse)
print('The Explained Sum of Squares is: ', ssr)
print('The R^2 from regressor: ', rm.score(x,y))
print('The R^2 from ssr/sst: ', ssr/sst)
#LINEAR REGRESSION OF THE ATTRIBUTES flightNumber, cabinClass, departureTime on Target: Price

lrAttributes = sha_pek_valid[['flightNumber','price','cabinClass','departureTime']]

#MAKING TIME EXECUTABLE FOR REGRESSION
lrAttributes['departureTime'] = lrAttributes['departureTime'].str.replace(':','.')

#CATEGORIZING OHC 
cabinClassEnc = pd.get_dummies(lrAttributes['cabinClass'])
flightNumberEnc = pd.get_dummies(lrAttributes['flightNumber'])

#CONCATENATING THE OHC to DATASET
lrAttributes = pd.concat([lrAttributes,cabinClassEnc,flightNumberEnc],axis=1)

#DELETING UNNECESSARY COLUMNS
lrAttributes = lrAttributes.drop(['flightNumber','cabinClass'],axis=1)

#SEPARATING PRICE vs THE REST
yAttributes = ['price']
xAttributes = list(set(list(lrAttributes.columns))-set(yAttributes))
xPrice = lrAttributes[xAttributes].values
yPrice = lrAttributes[yAttributes].values
xTrainPrice, xTestPrice, yTrainPrice, yTestPrice = model_selection.train_test_split(xPrice, 
                                                                                    yPrice, 
                                                                                    test_size=0.2, 
                                                                                    random_state = 2020)

#LINEAR REGRESSION
rm = linear_model.LinearRegression()
rm.fit(xTrainPrice,yTrainPrice)
trainPredPrice = rm.predict(xTrainPrice)
testPredPrice = rm.predict(xTestPrice)

#EVALUATION METRICS
print('R^2 for Training Data: ', rm.score(xTrainPrice,yTrainPrice))
print('R^2 for Test Data: ', rm.score(xTestPrice,yTestPrice))
print('Explained Metrics Score Test Data: ', metrics.explained_variance_score(yTestPrice,testPredPrice))
print('Mean Absolute Error Test Data: ', metrics.mean_absolute_error(yTestPrice,testPredPrice))
print('Mean Squared Error Test Data: ', metrics.mean_squared_error(yTestPrice,testPredPrice))
print('Root Mean Squared Error Test Data: ', np.sqrt(metrics.mean_squared_error(yTestPrice,testPredPrice)))

#LINEAR REGRESSION OF THE ATTRIBUTES flightNumber, cabinClass, departureTime on Target: Rate

lrAttributes = sha_pek_valid[['flightNumber','rate','cabinClass','departureTime']]

#MAKING TIME EXECUTABLE FOR REGRESSION
lrAttributes['departureTime'] = lrAttributes['departureTime'].str.replace(':','.')

#CATEGORIZING OHC
cabinClassEnc = pd.get_dummies(lrAttributes['cabinClass'])
flightNumberEnc = pd.get_dummies(lrAttributes['flightNumber'])

#CONCATENATING THE OHC to DATASET
lrAttributes = pd.concat([lrAttributes,cabinClassEnc,flightNumberEnc],axis=1)

#DELETING UNNECESSARY COLUMNS
lrAttributes = lrAttributes.drop(['flightNumber','cabinClass'],axis=1)

#SEPARATING PRICE vs THE REST
yAttributes = ['rate']
xAttributes = list(set(list(lrAttributes.columns))-set(yAttributes))
xRate = lrAttributes[xAttributes].values
yRate = lrAttributes[yAttributes].values
xTrainRate, xTestRate, yTrainRate, yTestRate = model_selection.train_test_split(xRate, 
                                                                                    yRate, 
                                                                                    test_size=0.2, 
                                                                                    random_state = 2020)

#LINEAR REGRESSION
rm = linear_model.LinearRegression()
rm.fit(xTrainRate,yTrainRate)
trainPredRate = rm.predict(xTrainRate)
testPredRate = rm.predict(xTestRate)

#EVALUATION METRICS
print('R^2 for Training Data: ', rm.score(xTrainRate,yTrainRate))
print('R^2 for Test Data: ', rm.score(xTestRate,yTestRate))
print('Explained Metrics Score Test Data: ', metrics.explained_variance_score(yTestRate,testPredRate))
print('Mean Absolute Error Test Data: ', metrics.mean_absolute_error(yTestRate,testPredRate))
print('Mean Squared Error Test Data: ', metrics.mean_squared_error(yTestRate,testPredRate))
print('Root Mean Squared Error Test Data: ', np.sqrt(metrics.mean_squared_error(yTestRate,testPredRate)))

#TAKING RELEVANT COLUMNS INTO CONSIDERATION

binAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]
binAttributes['rate'][binAttributes['rate']!=1]=0

#CATEGORIZING ONE HOT ENCODER VALUES FOR CLASSIFICATION
fNumberEnc = pd.get_dummies(binAttributes['flightNumber'])
traAirportEnc = pd.get_dummies(binAttributes['traAirport'])
cClassEnc = pd.get_dummies(binAttributes['cabinClass'])
priceClassEnc = pd.get_dummies(binAttributes['priceClass'])

#COMBINING OHC VALUES TO THE ORIGINAL DATASET
binAttributes = pd.concat([binAttributes,fNumberEnc,traAirportEnc,cClassEnc,priceClassEnc], axis=1)
#DELETING UNNECESSARY COLUMNS
binAttributes = binAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE vs THE REST
yAttribute = ['rate']
xAttribute = list(set(list(binAttributes.columns))-set(yAttribute))

#LOGISTIC REGRESSION

xR = binAttributes[xAttribute].values
yR = binAttributes[yAttribute].values.astype(int)

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xR,yR,test_size=0.25,random_state=2021)

logR = linear_model.LogisticRegression(solver='lbfgs')
logR.fit(xTrain,yTrain)

testPred = logR.predict(xTest)
print('The Accuracy Score is: ', metrics.accuracy_score(yTest,testPred)) #0.97216
#TAKING NECESSARY COLUMNS INTO CONSIDERATION

rocAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]
rocAttributes = rocAttributes.loc[rocAttributes['rate'] != 1]

#CATEGORIZING DATA ACCORDING TO THE QUESTION

for i in range(0,len(rocAttributes['rate'].values)):
    if rocAttributes['rate'].values[i] < 1 and rocAttributes['rate'].values[i] > 0.75:
        rocAttributes['rate'].values[i] = 1
    elif rocAttributes['rate'].values[i] <= 0.75 and rocAttributes['rate'].values[i] > 0.5:
        rocAttributes['rate'].values[i] = 2
    elif rocAttributes['rate'].values[i] <=0.5:
        rocAttributes['rate'].values[i] = 3

#CREATING ONE HOT ENCODER VALUES FOR CLASSIFICATION
fNumberEnc_roc = pd.get_dummies(rocAttributes['flightNumber'])
traAirportEnc_roc = pd.get_dummies(rocAttributes['traAirport'])
cClassEnc_roc = pd.get_dummies(rocAttributes['cabinClass'])
priceClassEnc_roc = pd.get_dummies(rocAttributes['priceClass'])

#DELETING REDUNDANT DATA
rocAttributes = pd.concat([rocAttributes,fNumberEnc_roc,traAirportEnc_roc,cClassEnc_roc,priceClassEnc_roc], axis=1)
rocAttributes = rocAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE VS THE REST OF THE ARGUMENTS
yAttr = ['rate']
xAttr = list(set(list(rocAttributes.columns))-set(yAttribute))

#ADDING NOISY FEATURES
xR1 = rocAttributes[xAttr].values
yR1 = rocAttributes[yAttr].values.astype(int)
yR1 = preprocessing.label_binarize(yR1,classes=[1,2,3])

nClasses = yR1.shape[1]
randomState = np.random.RandomState(0)

#SPLITTING TRAIN AND TEST BY 70:30 RATIO
xTrainR, xTestR, yTrainR, yTestR = model_selection.train_test_split(xR1, yR1, test_size=0.3, random_state=0)
#OVR LOGISTIC REGRESSION AND TEST PREDICTION

cfier = multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0))
yScore = cfier.fit(xTrainR,yTrainR).decision_function(xTestR)
#COMPUTING ROC CURVE AND ROC AREA FOR EACH CLASS

fpr=dict()
tpr=dict()
roc_auc=dict()


for i in range(nClasses):
    fpr[i], tpr[i], _ = metrics.roc_curve(yTestR[:,i],yScore[:,i])
    roc_auc[i] = metrics.auc(fpr[i],tpr[i])

    
#AGGREGATE ALL FALSE POSITIVE RATES    
allFpr = np.unique(np.concatenate([fpr[i] for i in range(nClasses)]))

#INTERPOLATE ALL ROC CURVES AT THESE POINTS
meanTpr = np.zeros_like(allFpr)
for i in range(nClasses):
    meanTpr += interp(allFpr, fpr[i], tpr[i])

#AVERAGE IT AND CALCULATE AUC
meanTpr /= nClasses

fpr["macro"] = allFpr
tpr["macro"] = meanTpr
roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

fpr["micro"], tpr["micro"], _ = metrics.roc_curve(yTestR.ravel(), yScore.ravel())
roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])


#ROC Curves
plt.figure()
lw=2
plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})' ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nClasses), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi_Class ROC')
plt.legend(loc="lower right")
plt.show()
#TAKING NECESSARY COLUMNS INTO CONSIDERATION
stackAttributes = sha_pek_valid[['flightNumber','traAirport','cabinClass','priceClass','rate']]

#CATEGORIZING DATA ACCORDING TO THE QUESTION

for i in range(0,len(stackAttributes['rate'].values)):
    if stackAttributes['rate'].values[i] == 1:
        stackAttributes['rate'].values[i] = 0
    elif stackAttributes['rate'].values[i] < 1 and stackAttributes['rate'].values[i] > 0.75:
        stackAttributes['rate'].values[i] = 1
    elif stackAttributes['rate'].values[i] <= 0.75 and stackAttributes['rate'].values[i] > 0.5:
        stackAttributes['rate'].values[i] = 2
    elif stackAttributes['rate'].values[i] <=0.5:
        stackAttributes['rate'].values[i] = 3

#CREATING ONE HOT ENCODER VALUES FOR CLASSIFICATION

fNumberEnc_stack = pd.get_dummies(stackAttributes['flightNumber'])
traAirportEnc_stack = pd.get_dummies(stackAttributes['traAirport'])
cClassEnc_stack = pd.get_dummies(stackAttributes['cabinClass'])
priceClassEnc_stack = pd.get_dummies(stackAttributes['priceClass'])

#DELETING REDUNDANT DATA TO AVOID PERFORMANCE ISSUES
stackAttributes = pd.concat([stackAttributes,fNumberEnc_stack,traAirportEnc_stack,cClassEnc_stack,priceClassEnc_stack], axis=1)
stackAttributes = stackAttributes.drop(['flightNumber','traAirport','cabinClass','priceClass'], axis=1)

#SEPARATING RATE VS THE REST OF THE ARGUMENTS
yAttr2 = ['rate']
xAttr2 = list(set(list(stackAttributes.columns))-set(yAttribute))

#LOGISTIC REGRESSION
xR2 = stackAttributes[xAttr].values
yR2 = stackAttributes[yAttr].values.astype(int)

xTrainS, xTestS, yTrainS, yTestS = model_selection.train_test_split(xR2, yR2, test_size=0.3, random_state=0)

clsfier = linear_model.LogisticRegression(solver='lbfgs',multi_class='multinomial')
clsfier.fit(xTrainS,yTrainS)

testPredS = clsfier.predict(xTestS)
confMat = metrics.confusion_matrix(yTestS,testPredS,[0,1,2,3])
confMat
fig = plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(confMat)
fig.colorbar(cax)
plt.title('Confusion Matrix')
ax.set_xticklabels(['']+[0,1,2,3])
ax.set_yticklabels(['']+[0,1,2,3])
plt.xlabel('Predictions')
plt.ylabel('Actuals')
print(metrics.classification_report(yTestS, testPredS))
print('The Training Score of OVR is: ',cfier.score(xTrainR,yTrainR))
print('The Training Score of Multinomial is: ', clsfier.score(xTrainS,yTrainS))