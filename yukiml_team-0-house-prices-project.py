!pip install regressors
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import linear_model as lm

from regressors import stats

import statsmodels.formula.api as sm

from sklearn.preprocessing import PolynomialFeatures,FunctionTransformer 

from sklearn.linear_model import LogisticRegression

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut



import os

print(os.listdir("../input"))
q1=pd.read_csv("../input/housetrain.csv")[:1000]

q1.head(10)
x1 = q1[["YearBuilt","SalePrice"]]

x2 = q1[["LotArea","SalePrice"]]

x3 = q1[["YearRemodAdd","SalePrice"]]

x4 = q1[["GarageYrBlt","SalePrice"]]

x5 = q1[["GarageArea","SalePrice"]]

x6 = q1[["GrLivArea","SalePrice"]]

x7 = q1[["TotRmsAbvGrd","SalePrice"]]

x8 = q1[["YrSold","SalePrice"]]



print("Correlation 1:\n",x1.corr())

print("Correlation 2:\n",x2.corr())

print("Correlation 3:\n",x3.corr())

print("Correlation 4:\n",x4.corr())

print("Correlation 5:\n",x5.corr())

print("Correlation 6:\n",x6.corr())

print("Correlation 7:\n",x7.corr())

print("Correlation 8:\n",x8.corr())



# Most relevant data

# GrLivArea > GarageArea > TotRmsAbvGrd > YearBuild > YearRemodAdd
# Clean data

a = q1[["GrLivArea","SalePrice","YearBuilt","YearRemodAdd","GarageArea","TotRmsAbvGrd"]]

print(a.isnull().values.any())

print(a.isnull().sum())

a = a.dropna()

print("Check for NaN/null values:\n",a.isnull().values.any())

print("Number of NaN/null values:\n",a.isnull().sum())
# 1. Fit a linear model

inputDF = a[["GrLivArea"]]

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print(model.intercept_, model.coef_)
# 2. Draw a scatterplot with the linear model as a line

y = model.predict(inputDF)

plt.scatter(inputDF,outcomeDF)

plt.plot(inputDF,y, color="blue")

plt.show()
# 4. Predict

xnew = pd.DataFrame(np.hstack(np.array([[1710],[1262],[1786],[1717],[2198]])))

xnew.columns=["GrLivArea"]

ynew = model.predict(xnew)

print(ynew)
#6 calculate the sum of squares of residuals for your model

predicted = model.predict(a[["GrLivArea"]])

print(np.sum((a[["GrLivArea"]]-predicted)**2))
# 1. Select 5 variables from your dataset. For each, draw a boxplot and analyze your observations.

fig = plt.figure(5, figsize=(20, 20))

cols = ["YearBuilt","YearRemodAdd","GrLivArea","SalePrice","GarageArea"]

for i in range(0,len(cols)):

    ax = fig.add_subplot(231+i)

    ax.boxplot(a[cols[i]])

plt.show()
#2 Draw a scatterplot for each pair and make your visual observations.

fig = plt.figure(6, figsize=(20, 20))

cols = ["YearBuilt","YearRemodAdd","GrLivArea","SalePrice"]

count = 0

for i in range(0,len(cols)):

    for j in range(i+1,len(cols)):

        ax = fig.add_subplot(431+count)

        ax.scatter(a[cols[i]],a[cols[j]])

        count += 1

plt.show()

#Regression Coefficients, Adjusted R-Squared and P-value calculation  



print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
est = sm.ols(formula="SalePrice ~ GrLivArea", data=a).fit()

print(est.summary())
# Polynomial regression (Quadratic)

inputDF = a[["GrLivArea"]]

poly_features = PolynomialFeatures ( degree = 2 , include_bias = False ) 

inputDF = poly_features . fit_transform ( inputDF ) 

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
# Polynomial regression (Cubic)

inputDF = a[["GrLivArea"]]

poly_features = PolynomialFeatures ( degree = 3 , include_bias = False ) 

inputDF = poly_features . fit_transform ( inputDF ) 

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
# Polynomial regression (Quartic)

inputDF = a[["GrLivArea"]]

poly_features = PolynomialFeatures ( degree = 4 , include_bias = False ) 

inputDF = poly_features . fit_transform ( inputDF ) 

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)



print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
#'GrLivArea' and 'GarageArea' as independent variable and 'SalePrice' as dependent variable.



inputDF = a[["GrLivArea","GarageArea"]]

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)

print(model.intercept_, model.coef_)
#Regression Coefficients, Adjusted R-Squared and P-value calculation  

print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea", data=a).fit()

print(est.summary())
est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea+TotRmsAbvGrd", data=a).fit()

print(est.summary())
inputDF = a[["GrLivArea"]]

transformer = FunctionTransformer(np.log1p, validate=True)

inputDF = transformer.transform ( inputDF )

inputDF = pd.concat([pd.DataFrame(inputDF),a[["GarageArea"]]],axis=1, join='inner')

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)

print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
inputDF = a[["GarageArea"]]

transformer = FunctionTransformer(np.log1p, validate=True)

inputDF = transformer.transform ( inputDF )

inputDF = pd.concat([pd.DataFrame(inputDF),a[["GrLivArea"]]],axis=1, join='inner')

outcomeDF = a[["SalePrice"]]

model = lm.LinearRegression()

results = model.fit(inputDF,outcomeDF)

print("Regression Coefficients: \n",model.intercept_, model.coef_)

print("Adjusted R-Squared:\n",stats.adj_r2_score(model, inputDF, outcomeDF))

print("P-value:\n",stats.coef_pval(model, inputDF, outcomeDF))
d=pd.read_csv("../input/housetrain.csv")

d.head()
#GrLivArea > GarageArea > TotRmsAbvGrd > YearBuild > YearRemodAdd

inter1 = sm.ols(formula="SalePrice ~ YrSold+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd",data=d).fit()

print(inter1.summary())
inter2 = sm.ols(formula="SalePrice ~ YrSold*YearBuilt*GrLivArea*GarageArea*TotRmsAbvGrd",data=d).fit()

print(inter2.summary())
inter3 = sm.ols(formula="SalePrice ~ YearBuilt + I(YearBuilt*YearBuilt) + I(YearBuilt*YearBuilt*YearBuilt)",data=d).fit()

print(inter3.summary())
inter4 = sm.ols(formula="SalePrice ~ GrLivArea + I(GrLivArea*GrLivArea) + I(GrLivArea*GrLivArea*GrLivArea)",data=d).fit()

print(inter4.summary())
df = pd.read_csv("../input/housetrain.csv")

inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea", "TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]

outputDF = df[["SalePrice"]]



R = 0

Feature = list()

for i in range(1,9):

    model = sfs(LinearRegression(),k_features=i,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

    model.fit(inputDF,outputDF)

    inputDFtemp = df[list(model.k_feature_names_)]

    outcomeDFtemp = df[["SalePrice"]]

    modelnew = lm.LinearRegression()

    results = modelnew.fit(inputDFtemp,outcomeDFtemp)

    if stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp) >= R:

        R = stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp)

        feature = list(model.k_feature_names_)

print("Final feature",feature)

print("Final R Square",R)



df = pd.read_csv("../input/housetrain.csv")

inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea", "TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]

outputDF = df[["SalePrice"]]



R = 0

Feature = list()

for i in range(1,9):

    model = sfs(LinearRegression(),k_features=i,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

    model.fit(inputDF,outputDF)

    inputDFtemp = df[list(model.k_feature_names_)]

    outcomeDFtemp = df[["SalePrice"]]

    modelnew = lm.LinearRegression()

    results = modelnew.fit(inputDFtemp,outcomeDFtemp)

    if stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp) >= R:

        R = stats.adj_r2_score(modelnew, inputDFtemp, outcomeDFtemp)

        feature = list(model.k_feature_names_)

print("Final feature",feature)

print("Final R Square",R)
inputDF = df[["YrSold","YearBuilt","GrLivArea","GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]

outputDF = df[["SalePrice"]]

model = LinearRegression()

loocv = LeaveOneOut()



rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = loocv))

print(rmse.mean())
inputDF = df[["YrSold","YearBuilt","GrLivArea", "GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]

outputDF = df[["SalePrice"]]

model = LinearRegression()

kf = KFold(5, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
inputDF = df[["YrSold","YearBuilt","GrLivArea","GarageArea","TotRmsAbvGrd","OverallQual","OverallCond","BedroomAbvGr"]]

outputDF = df[["SalePrice"]]

model = LinearRegression()

kf = KFold(10, shuffle=True, random_state=42).get_n_splits(inputDF)

rmse = np.sqrt(-cross_val_score(model, inputDF, outputDF, scoring="neg_mean_squared_error", cv = kf))

print(rmse.mean())
est = sm.ols(formula="SalePrice ~ YrSold+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd+OverallQual+OverallCond+BedroomAbvGr", data=d).fit()

print(est.summary())
est = sm.ols(formula="SalePrice ~ I(YrSold*YrSold*YrSold)+I(YrSold*YrSold)+YearBuilt+GrLivArea+GarageArea+TotRmsAbvGrd+OverallQual+OverallCond+BedroomAbvGr", data=d).fit()

print(est.summary())
est = sm.ols(formula="SalePrice ~ GrLivArea+GarageArea", data=a).fit()

print(est.summary())