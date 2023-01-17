import pandas as pd # data processing
import numpy as np # linear algebra
import seaborn as sns # plotting interactive graphs
import matplotlib.pyplot as plt # plotting the graph
import warnings

# allow plots to appear directly in the notebook
%matplotlib inline  
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
boston=load_boston()
#Adding the feature names to the dataframe
df=pd.DataFrame(boston.data,columns=boston.feature_names)

#Adding target variable to dataframe
df['MEDV'] = boston.target 
print("Total number of rows in dataset = {}".format(df.shape[0]))
print("Total number of columns in dataset = {}".format(df.shape[1]))
df.head(5)
df.describe()
df.dtypes
df.nunique()
df.isnull().sum()
def outlier_treatement(base_dataset):
    for i in base_dataset.columns:
        x=np.array(base_dataset[i])
        qr1=np.quantile(x,0.25)
        qr3=np.quantile(x,0.75)
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        for p in x:
            if p < ltv or p > utv:
                y.append(np.median(x))
            else:
                y.append(p)
        base_dataset[i]=y
        print(i)
    return base_dataset       
#df=outlier_treatement(df)
plt.figure(figsize=(8,5))
sns.distplot(df['MEDV'], bins=30)
plt.show()
df.corr()
plt.figure(figsize=(12,10))
sns.heatmap( df.corr(), annot=True );
X = df.drop(['MEDV'],axis=1)
y=df['MEDV']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=42)
print('X_train: ', X_train.shape)
print('X_test: ', X_test.shape)
print('y_train: ',y_train.shape)
print('y_test: ',y_test.shape)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred_train = linreg.predict(X_train)
y_pred_test=linreg.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model_result=[]
MAE_train=mean_absolute_error(y_train,y_pred_train)
MSE_train=mean_squared_error(y_train,y_pred_train)
RMSE_train = np.sqrt(mean_squared_error(y_train,y_pred_train))
r2_train = r2_score(y_train,y_pred_train)

MAE_test=mean_absolute_error(y_test,y_pred_test)
MSE_test=mean_squared_error(y_test,y_pred_test)
RMSE_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
r2_test = r2_score(y_test,y_pred_test)

model_result.append([MAE_train,MSE_train,RMSE_train,r2_train,MAE_test,MSE_test,RMSE_test,r2_test])
model_result=pd.DataFrame(model_result)
model_result.index=['LinearRegression']
model_result.columns=['MAE_train','MSE_train','RMSE_train','r2_train','MAE_test','MSE_test','RMSE_test','r2_test']
model_result
plt.scatter(y_train, y_pred_train)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
sns.distplot(y_train-y_pred_train)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
#Correlation with output variable
cor_target = abs(cor["MEDV"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features
print(df[["LSTAT","PTRATIO"]].corr())
print(df[["RM","LSTAT"]].corr())
print(df[["RM","PTRATIO"]].corr())
X = pd.DataFrame(np.c_[df['LSTAT'], df['PTRATIO']], columns = ['LSTAT','PTRATIO'])
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
linreg.fit(X_train,y_train)
y_pred_train = linreg.predict(X_train)
y_pred_test=linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred_train)))
r2 = r2_score(y_train, y_pred_train)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_pred_test = linreg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
r2 = r2_score(y_test, y_pred_test)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
# Creating scaled set to be used in model to improve our results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred_train = linreg.predict(X_train)
y_pred_test=linreg.predict(X_test)
y_train_predict = linreg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred_train)))
r2 = r2_score(y_train, y_pred_train)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_pred_test = linreg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred_test)))
r2 = r2_score(y_test, y_pred_test)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
from sklearn.ensemble import RandomForestRegressor
randomreg = RandomForestRegressor()
randomreg.fit(X_train,y_train)
y_train_pred = randomreg.predict(X_train)
y_test_pred = randomreg.predict(X_test)
print('R^2:',r2_score(y_train, y_train_pred))
print('Adjusted R^2:',1 - (1-r2_score(y_train, y_train_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',mean_absolute_error(y_train, y_train_pred))
print('MSE:',mean_squared_error(y_train, y_train_pred))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_train_pred)))
y_train_predict = randomreg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_pred)))
r2 = r2_score(y_train, y_train_pred)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_pred = randomreg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_pred)))
r2 = r2_score(y_test, y_test_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
X=df.iloc[:,df.columns!='MEDV']
y=df.MEDV
import statsmodels.api as sm
#Adding constant column of ones, mandatory for sm.OLS model
X_1 = sm.add_constant(X)
#Fitting sm.OLS model
model = sm.OLS(y,X_1).fit()
model.pvalues
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
X=df[selected_features_BE]
from sklearn.preprocessing import StandardScaler
def standard_scaler(X):
    ss= StandardScaler()
    for i in X.columns:
        ss.fit(X[i].values.reshape(-1,1))
        x=ss.transform(X[i].values.reshape(-1,1))
        X[i]=x
    return X      
standard_scaler(X)
X1=df.iloc[:,df.columns!='MEDV']
y1=df.MEDV
from sklearn.linear_model import LassoCV
reg=LassoCV()
reg.fit(X1,y1)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X1,y1))
coef = pd.Series(reg.coef_, index = X1.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
X1=df[coef[coef !=0].index]
y=df['MEDV']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.30,random_state=1)
from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train,y_train)
r2_train = r2_score(y_train, ls.predict(X_train))
r2_test = r2_score(y_test, ls.predict(X_test))
r2_train,r2_test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=1)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import svm

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
models = [LinearRegression, Lasso, RandomForestRegressor,XGBRegressor,svm.SVR]
model_results=[]
for i in models:
    ln=i()
    ln.fit(X_train,y_train)
    MAE_train=mean_absolute_error(y_train, ln.predict(X_train))
    MSE_train = mean_squared_error(y_train, ln.predict(X_train))
    RMSE_train = np.sqrt(mean_squared_error(y_train, ln.predict(X_train)))
    r2_train = r2_score(y_train, ln.predict(X_train)) 
    MAE_test = mean_absolute_error(y_test, ln.predict(X_test)) 
    MSE_test = mean_squared_error(y_test, ln.predict(X_test))
    RMSE_test = np.sqrt(mean_squared_error(y_test, ln.predict(X_test))) 
    r2_test = r2_score(y_test, ln.predict(X_test))
    model_results.append([MAE_train,MSE_train,RMSE_train,r2_train,MAE_test,MSE_test,RMSE_test,r2_test])
model_results=pd.DataFrame(model_results)
model_results.columns = ['MAE_train','MSE_train','RMSE_train','r2_train','MAE_test','MSE_test','RMSE_test','r2_test']
model_results.index =   ['LinearRegression', 'Lasso','RandomForest','XGBoost','SVM']
model_results