import pandas as pd
import numpy as np
import matplotlib as mp
%matplotlib inline
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
pd.options.display.max_columns = None
pd.options.display.max_rows = None
sp = pd.read_csv('../input/sp.csv')
sp.head()
sp1 = sp
sp1.iloc[:,0:4] = sp1.iloc[:,0:4].astype("category")
sp1.select_dtypes(include="category").describe()
sp1.select_dtypes(include="int64").describe()
#Relation entre les varaibles quantitatives
#recupérer les quantitative et faire un pairplot
pairplot = sp1.iloc[:,[0,2,3,5,6,7]]
#pairplot.to_csv("pairplot.csv")
sns.pairplot(pairplot, hue ='gender')
plt.show()
#on remarque qu'il y à bien une liansion linaire entre les différentes notes
#obtenues. 
pairplot.head()
sns.lmplot(x= "math score", y = "reading score", data = sp1, hue="gender")
plt.show()
#This lmplot show us that it may have a diff between gender exam score
#We will try to figure it out through an linear regression
#Let's see this diff with boxplot : math score by gender specially
plt.figure(figsize=(6,6))# addding figsize
fig = sns.boxplot(x= 'gender', y='math score', data= pairplot)
plt.ylim(0,120)
plt.yticks(np.arange(0, max(pairplot['math score'])+10, 5))#setting graduations
plt.show()
#Let's look how math score are distribute by gender and taking accound the kind of lunch that the student use to have
b=sns.FacetGrid(pairplot, col="gender", hue ="lunch")
b.map(sns.distplot,"math score")
plt.ylim(0,0.035)
plt.show()
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.distplot(sp1["math score"],ax=axes[0, 0],color='red')
sns.distplot(sp1["reading score"],ax=axes[0, 1],)
sns.distplot(sp1["writing score"],ax=axes[1, 0], color = "green")
plt.show()
df = pairplot
df1 = df
#Feautures engeneering on "parental level of education" feature
sp1["parental level of education"].value_counts()
#create two cat of parental degree : undergraduate 0 and graduate 1
#undergraduate : some college, associate, high school, some high school
#graduate : bachelor's degree, master's degree
df.head()
def p_grad(x):
    if x == "bachelor's degree" or x == "master's degree ":
        return 1
    else:
        return 0

df["p_grad"] = df["parental level of education"].apply(p_grad)
def sex(x):
    if x == "female":
        return 0
    else:
        return 1

df["sex"] = df["gender"].apply(sex)
df = df.astype({"sex": int}) # i convert to int because i've notice that it still category after using the apply function.
#Note that we can also use astype("categoty").cat.codes function in order to transform gender to categorial varaible 
#but with this small function "def sex(x)" you know excatly what is female and male.
def lunch_grad(x):
    if x == "standard":
        return 1
    else:
        return 0

df["lunch_grad"] = df["lunch"].apply(lunch_grad)
df = df.astype({"lunch_grad": int})
#New look of the data and the features created
df.head()
#selecting numeric columns
df = df._get_numeric_data()
#Just want to have close look on correlation degree between features
df.corr(method='spearman')#then ask for 
sns.heatmap(df.corr(method='pearson'))
sns.heatmap(df.corr(method='spearman'))
plt.show()
#Creating the independantes features call x and the target call y
x= df.drop(df.columns[[0,2]], axis = 1)
y = df.iloc[:,0]
#Using train_test_split from skealearn to split the dataset into train and test samples
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
#as statmodel don't automatically add constant in the model we have to do it so
x_train_sm = sm.add_constant(x_train) # addind constant
model_sm = sm.OLS(y_train, x_train_sm)# def the model
model_sm_fit = model_sm.fit()#fiting the model
view_res =model_sm_fit.summary()#store the result in view_res
#Lokk on the result 
view_res
x_test_sm =  sm.add_constant(x_test)
y_pred_of_x_test = model_sm_fit.predict(x_test_sm)
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred_of_x_test)))
## Sickit Learn OLS
from sklearn.metrics import mean_squared_error, r2_score
model_sklr = linear_model.LinearRegression()
#using x_train and y_train to build and fit the model
results = model_sklr.fit(x_train, y_train)
y_pred_skl = model_sklr.predict(x_test)
# The mean squared error
print('Mean squared error: %.2f'% mean_squared_error(y_test, y_pred_skl))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'% r2_score(y_test, y_pred_skl))

