import numpy as np

import pandas as pd

from scipy.stats import norm

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsRegressor
data = pd.read_csv('../input/diamonds/diamonds.csv')

np.random.seed(42)

data.head()
data.info()
df_train,  final_data, df_test, final_result = train_test_split(data.drop('price', axis=1), data.price, test_size=0.1)
df = df_train.join(df_test)

df = df.drop('Unnamed: 0', axis=1)
df['log_price'] = np.log1p(df.price)

print(round(df.corr(),2))

sns.heatmap(df.corr())
sns.pairplot(df.head(5000))
df.drop(df[df.carat>3.5].index, inplace = True)

df.drop(df[df.x<1].index, inplace = True)

df.drop(df[df.y<1].index, inplace = True)

df.drop(df[df.y>20].index, inplace = True)

df.drop(df[df.z<1].index, inplace = True)

df.drop(df[df.z>15].index, inplace = True)

df.drop(df[df.depth<50].index, inplace = True)

df.drop(df[df.depth>77].index, inplace = True)

df.drop(df[df.table<50].index, inplace = True)

df.drop(df[df.table>74].index, inplace = True)

df.shape
print(df[ 'cut'].describe())

print(df['cut'].unique())

print(df[ 'color'].describe())

print(df['color'].unique())

print(df[ 'clarity'].describe())

print(df['clarity'].unique())
fig, axs = plt.subplots(ncols=3, figsize=(18,4))

sns.boxplot(y='price', x = 'cut', data = df, ax = axs[0]).set_title('Price distribution by cut type')

sns.countplot( x = 'cut',   data = df,  ax = axs[1]).set_title('Frequencies of different cut types')

sns.boxplot( y = 'x', x = 'cut',   data = df,  ax = axs[2]).set_title('Size distribution of different cut types')
plt.figure(figsize=(17,7))

sns.scatterplot('x','log_price', hue= 'cut', data = df[0:10000]).set_title('Price vs x_dimension, by cut')
fig, axs = plt.subplots(ncols=2, figsize=(15,4))

sns.scatterplot('table','x', hue= 'cut', data = df, ax = axs[0]).set_title('Price vs table by cut')

sns.scatterplot('depth','x', hue= 'cut', data = df, ax = axs[1]).set_title('Dimension vs table by cut')
plt.figure(figsize=(10,10))

sns.scatterplot('table','depth', hue= 'cut', data = df).set_title('Depth vs table by cut')

plt.xlim(50,70)

plt.ylim(55,70)
fig, axs = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(y='price', x = 'color', data = df, ax = axs[0]).set_title('Price distribution by color')

sns.countplot( x = 'color',   data = df,  ax = axs[1]).set_title('Frequencies of different colors')

sns.boxplot( y = 'x', x = 'color',   data = df,  ax = axs[2]).set_title('Size distribution of different colors')

sns.countplot(x='color', hue = 'cut', data = df, ax = axs[3]).set_title('Distribution of cuts for different colors')
plt.figure(figsize=(17,7))

sns.scatterplot('x','log_price', hue= 'color', data = df).set_title('Price vs x-size graph, by color')
plt.figure(figsize=(17,7))

sns.scatterplot('x','log_price', hue= 'clarity', data = df).set_title('Price vs x-size graph, by clarity')
dfx = df[(df.x<7.3) & (df.x>7.2)]

fig, axs = plt.subplots(ncols=3, figsize=(17,5))

sns.boxplot(y='price', x='clarity', data =  dfx, ax = axs[0]).set_title('For 7.2<x<7.3')

sns.boxplot(y='price', x='color', data =  dfx, ax = axs[1]).set_title('For 7.2<x<7.3')

sns.boxplot(y='price', x='cut', data =  dfx, ax = axs[2]).set_title('For 7.2<x<7.3')
cut_dict = {'Good':1, 'Premium':3, 'Very Good':2, 'Ideal':4, 'Fair':0}

df.cut = df.cut.map(cut_dict)

clarity_dict = {'SI1':2, 'VS1':4, 'VVS1':6, 'SI2':1, 'VS2':3, 'IF':7, 'VVS2':5, 'I1':0}

df.clarity = df.clarity.map(clarity_dict)

color_dict = {'E':5, 'H':2, 'D':6, 'F':4, 'I':1, 'G':3, 'J':0}

df.color = df.color.map(color_dict)
most_freq = data[(data['color']=='G') & (data['cut']=='Ideal') & (data['clarity']=='SI1') 

                 & (data['x']>5.67) & (data['x']<6)]

sns.scatterplot('x', 'price', data = most_freq)

plt.title('Prices of diamonds with most common features by size')
X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)

y = df.price



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe_lr = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree = 4), PCA(n_components=80) , LinearRegression())

pipe_lr.fit(X_train, y_train)



print('Training score:', pipe_lr.score(X_train,y_train))

print('Test score: ',pipe_lr.score(X_test, y_test))

plt.title('Residual graph')

plt.scatter(y_test.values, 100*(y_test.values - pipe_lr.predict(X_test))/y_test.values)

plt.ylim(-100,100)

x = [50 * i for i in range(380) ]

plt.plot(x,[0]*380, color = 'red')

plt.ylabel('Error in % of the price')

plt.xlabel('Real price')    
def evaluate_graph(estimator, X_test, y_test):

    """sketches a graph to evaluate residual errors and prints regression score"""

    

    print('Test score: ',estimator.score(X_test, y_test))

    plt.title('Residual graph')

    plt.scatter(np.expm1(y_test.values), 

                100*(np.expm1(y_test.values) - np.expm1(estimator.predict(X_test)))/np.expm1(y_test.values))

    plt.ylim(-100,100)

    x = [50 * i for i in range(380) ]

    plt.plot(x,[0]*380, color = 'red')

    plt.ylabel('Error in % of the price')

    plt.xlabel('Real price')    
X = df.drop(['price', 'table', 'depth',  'log_price'], axis = 1)

y = df.log_price



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr=LinearRegression()

lr.fit(X_train, y_train)



print('Training score:', lr.score(X_train, y_train))

evaluate_graph(lr, X_test, y_test)
feat_data = df[['table', 'depth', 'cut' ]]

feat_data.head()
Xf = feat_data[['table', 'depth']]

yf = feat_data['cut']

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.33)



#svm = LinearSVR(epsilon=0.0, tol=0.0001, C=10, loss='epsilon_insensitive', 

#                intercept_scaling=1.0, dual=True, verbose=0, max_iter=1000)

clf = KNeighborsRegressor(n_neighbors=150, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)

clf.fit(Xf_train, yf_train)

clf.score(Xf_test, yf_test)
def plot_decision_regions(classifier, resolution = 1):

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:5])

    

    x1_min = 50

    x2_min =50

    x1_max = 70

    x2_max = 70

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                          np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.5, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.xlim(xx2.min(), xx2.max())

    plt.xlabel('table')

    plt.ylabel('depth')

    plt.title('Decision regions for cut type')



    #Credit to Sebastian Raschka, Vahid Mirjalili - Python Machine Learning. 2nd ed-Packt Publishing (2017)
plot_decision_regions(clf)
df['new_cut'] = clf.predict(df[['table', 'depth']])
df.head()
df[['cut', 'new_cut', 'log_price']].corr()
dfpr = df[(df.x>5.67) & (df.x<6) & (df['color']==3) & (df['clarity']==2) ]

print('Correlations for diamonds with same most common features:')

dfpr[['cut', 'new_cut', 'log_price']].corr()
print('Mean continuos cut by cut type:')

df.new_cut.groupby(df.cut).mean()
sns.boxplot(x= 'cut', y = 'new_cut',  data = df)
X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)

y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr= LinearRegression()

lr.fit(X_train, y_train)



print('Training score:', lr.score(X_train, y_train))

evaluate_graph(lr, X_test, y_test)

df['volume'] = df.x*df.y*df.z

df['area'] = df.x*df.y

df['root_carat'] = np.sqrt(df.carat)

X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)

y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr= LinearRegression()

lr.fit(X_train, y_train)



print('Training score:', lr.score(X_train, y_train))

evaluate_graph(lr, X_test, y_test)
residual_error = 100*(np.expm1(y_test.values) - np.expm1(lr.predict(X_test)))/np.expm1(y_test.values)

plt.figure(figsize=(7,6))

plt.title('Residual error distribution')

plt.xlim(-100,100)

plt.xlabel('Residual error in %')

plt.ylabel('Density')

sns.distplot(residual_error, bins = 100, kde = True, fit=norm)

plt.show()
print(residual_error.mean(), residual_error.std())
X_test.loc[:,'residual_error'] = residual_error

X_test.join(df[[ 'table', 'depth', 'price', 'log_price']]).corr()['residual_error'].sort_values()
final_data.cut = final_data.cut.map(cut_dict)

final_data.clarity = final_data.clarity.map(clarity_dict)

final_data.color = final_data.color.map(color_dict)

final_data['new_cut']  = clf.predict(final_data[['table', 'depth']])

final_data['volume'] = final_data.x*final_data.y*final_data.z

final_data['area'] = final_data.x*final_data.y

final_data['root_carat'] = np.sqrt(final_data.carat)

final_data = final_data.drop(['Unnamed: 0', 'table', 'depth'], axis=1)
linreg = LinearRegression()

linreg.fit(X,y)
plt.title('Residual graph')

plt.scatter(final_result.values, 100*(final_result.values - np.expm1(lr.predict(final_data)))/final_result.values)

plt.ylim(-100,100)

plt.ylabel('Error in % of the price')

plt.xlabel('Real price')

x = [50 * i for i in range(380) ]

plt.plot(x,[0]*380, color = 'red')

print('Regression score:', lr.score(final_data, np.log1p(final_result.values)))