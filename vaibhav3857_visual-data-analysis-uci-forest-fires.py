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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fires = pd.read_csv('../input/forest-forest-dataset/forestfires.csv')

fires['areaclass'] = [0 if val==0.0 else 1 for val in fires['area']]

fires.head()
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt 



y = fires['areaclass']

x = fires.drop(['areaclass','area','month','day','X','Y'],axis=1)



xnorm = (x - x.min()/x.max()-x.min())



# 2-dimensional PCA

pca = PCA(n_components=2)

trans = pd.DataFrame(pca.fit_transform(xnorm))



plt.scatter(trans[y==0][0], trans[y==0][1], label='non-fire area', c='green')

plt.scatter(trans[y==1][0], trans[y==1][1], label='fire area', c='blue')

plt.legend()

plt.show()
x.columns
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(x)

x = scaler.transform(x)



pca = PCA()

xnew = pca.fit_transform(x)



def myplot(score,coeff,labels=None):

    xs = score[:,0]

    ys = score[:,1]

    n = coeff.shape[0]

    scalex = 1.0/(xs.max() - xs.min())

    scaley = 1.0/(ys.max() - ys.min())

    plt.scatter(xs * scalex,ys * scaley, c = y)

    for i in range(n):

        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.2)

        if labels is None:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')

        else:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')



plt.xlim(-1,1)

plt.ylim(-1,1)

plt.xlabel("PC{}".format(1))

plt.ylabel("PC{}".format(2))

plt.grid()



#Call the function. Use only the 2 PCs.

myplot(xnew[:,0:2],np.transpose(pca.components_[0:2, :]))

plt.show()
pca.explained_variance_ratio_

# print(abs( pca.components_ ))

# np.transpose(pca.components_[0:2, :]).shape[0]
model = PCA(n_components=4).fit(x)

X_pc = model.transform(x)



# number of components

n_pcs= model.components_.shape[0]



# get the index of the most important feature on EACH component

# LIST COMPREHENSION HERE

most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]



initial_feature_names = ['FFMC','DMC','DC','ISI','temp','RH','wind','rain']

# get the names

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]



# LIST COMPREHENSION HERE AGAIN

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}



# build the dataframe

df = pd.DataFrame(dic.items())
df
from pandas.plotting import parallel_coordinates



plotcols = ['temp','RH','wind','rain','FFMC','DMC','DC','ISI']

data_norm = pd.concat([xnorm[plotcols],y],axis=1)

parallel_coordinates(data_norm,'areaclass')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder



# considering only relevant columns

linearcols = ['month','temp','RH','wind','rain','FFMC','DMC','DC','ISI']

datafires = fires[linearcols]



# label encoding for 'month' column

le = LabelEncoder()

xdata = datafires.apply(le.fit_transform)

xreg = xdata
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score



yreg = fires['area']



xtrain, xtest, ytrain, ytest = train_test_split(xreg,yreg)



#fittiing the model

regressor = LinearRegression(fit_intercept=False)

regressor.fit(xtrain,ytrain)

yregpred = regressor.predict(xtest)



#results

print('Coefficient of determination r^2: %.2f' % r2_score(ytest,yregpred))

print('RMSE: %.2f' % mean_squared_error(ytest,yregpred,squared=False))
lrresults = pd.DataFrame(yregpred,ytest)

lrresults.reset_index()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline



poly_model = make_pipeline(PolynomialFeatures(5),LinearRegression())

poly_model.fit(xtrain,ytrain)

ypolyfit = poly_model.predict(xtest)



plt.plot(xtest,ypolyfit)
fires['logarea'] = np.log10(fires['area'])

fires.replace([np.inf,-np.inf],0.0,inplace=True)



ylog = fires['logarea']

xlog = xdata



xltrain,xltest,yltrain,yltest = train_test_split(xlog,ylog)

logregressor = LinearRegression()

logregressor.fit(xltrain,yltrain)

ylogpred = logregressor.predict(xltest)



#results

print('Coefficient of determination r^2: %.2f' % r2_score(yltest,ylogpred))

print('RMSE: %.2f' % mean_squared_error(yltest,ylogpred,squared=False))
pcacols = ['temp','RH','wind','rain']

datax = fires[pcacols]

datay = fires['logarea']



pcaxtrain, pcaxtest, pcaytrain, pcaytest = train_test_split(datax,datay)

pcaregressor = LinearRegression()

pcaregressor.fit(pcaxtrain,pcaytrain)

pcaypred = pcaregressor.predict(pcaxtest)



#results

print('Coefficient of determination r^2: %.2f' % r2_score(pcaytest,pcaypred))

print('RMSE: %.2f' % mean_squared_error(pcaytest,pcaypred,squared=False))
print('Coefficients: \n',pcaregressor.coef_)

print('Intercept: \n',pcaregressor.intercept_)
cols = ['temp','RH']

X = fires[cols].values.reshape(-1,2)

Y = fires['logarea']

xvtrain,xvtest,yvtrain,yvtest = train_test_split(X,Y)



x = xvtrain[:,0]

y = xvtrain[:,1]

z = yvtrain



xx_pred, yy_pred = np.meshgrid(xvtest[:,0], xvtest[:,1])

model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T



ols = LinearRegression()

model = ols.fit(xvtrain, yvtrain)

predicted = model.predict(model_viz)



############################################ Evaluate ############################################



r2 = model.score(xvtrain, yvtrain)



############################################## Plot ################################################



plt.style.use('default')



fig = plt.figure(figsize=(12, 4))



ax1 = fig.add_subplot(131, projection='3d')

ax2 = fig.add_subplot(132, projection='3d')

ax3 = fig.add_subplot(133, projection='3d')



axes = [ax1, ax2, ax3]



for ax in axes:

    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)

    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')

    ax.set_xlabel('Temperature', fontsize=12)

    ax.set_ylabel('Relative Humidity', fontsize=12)

    ax.set_zlabel('log(forest fire area)', fontsize=12)

    ax.locator_params(nbins=4, axis='x')

    ax.locator_params(nbins=5, axis='x')



ax1.view_init(elev=28, azim=120)

ax2.view_init(elev=4, azim=114)

ax3.view_init(elev=60, azim=165)



fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()
import seaborn as sns



df = fires.iloc[:,4:-3]



corr = df.corr(method='spearman')



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(6, 5))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=.5)



fig.suptitle('Correlation matrix of features', fontsize=10)

fig.tight_layout()