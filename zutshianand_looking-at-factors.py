#Libraries to import
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import Imputer , Normalizer , scale, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_selection import RFECV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_absolute_error, average_precision_score, precision_recall_curve, roc_curve, precision_score, make_scorer, accuracy_score, classification_report, confusion_matrix, mean_squared_error, recall_score, f1_score, roc_auc_score, r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import pydot
import matplotlib.patches as mpatches
from pandas import get_dummies
import xgboost as xgb
from xgboost import XGBRegressor
import scipy
import math
import json
import sys
import csv
import os
import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, Embedding, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from tqdm import tqdm_notebook
from nltk.corpus import stopwords
import string
from collections import Counter
from string import punctuation
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from gensim.models import Word2Vec
from scipy.stats import norm
from keras.callbacks import ModelCheckpoint
import time
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from keras.utils.vis_utils import plot_model
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.describe()
def plot_gaussian_distribution(df, var):
    f, (ax1) = plt.subplots(1,1, figsize=(20, 6))
    dist = df[var].values
    sns.distplot(dist,ax=ax1, fit=norm, color='#FB8861')
    ax1.set_title('Distribution for ' + str(var), fontsize=14)
    plt.show()

def plot_boxplot(df, var, target):
    """
    var has to be a categorical variable
    target has to be discrete
    """
    f, axes = plt.subplots(ncols=2, figsize=(20,4))
    sns.boxplot(x=var, y=target, data=df, palette="Blues", ax=axes[0])
    axes[0].set_title('Boxplot for ' + str(var) + ' and ' + str(target))
    plt.show()
    
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + var_name )
    fig.tight_layout()
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 24 , 20 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
                    corr,
                    cmap = cmap,
                    square=True,
                    cbar_kws={ 'shrink' : .9 },
                    ax=ax,
                    annot = True,
                    annot_kws = { 'fontsize' : 12 }
                    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels
def num_missing(x):
    return sum(x.isnull())

print("Missing values per column:")
print(df.apply(num_missing, axis=0))

print("\nMissing values per row:")
print(df.apply(num_missing, axis=1).head())
plot_gaussian_distribution(df, 'math score')
plot_gaussian_distribution(df, 'reading score')
plot_gaussian_distribution(df, 'writing score')
new_df = df[['math score', 'reading score', 'writing score']]
plot_correlation_map(new_df)
plt.boxplot(df['math score'])
plt.boxplot(df['reading score'])
plt.boxplot(df['writing score'])
df = df[df['reading score'] > 30]
df = df[df['math score'] > 30]
df = df[df['writing score'] > 30]
plt.boxplot(df['math score'])
plt.boxplot(df['reading score'])
plt.boxplot(df['writing score'])
plot_gaussian_distribution(df, 'math score')
plot_gaussian_distribution(df, 'reading score')
plot_gaussian_distribution(df, 'writing score')
df.head()
plot_distribution(df, 'math score', 'gender', row = 'race/ethnicity', col = 'lunch')
plot_distribution(df, 'math score', 'parental level of education', row = 'test preparation course', col = 'gender')
new_df = df
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
new_df = MultiColumnLabelEncoder(columns = ['gender','race/ethnicity', 'lunch', 'test preparation course']).fit_transform(new_df)
new_df.head()
plot_distribution(new_df, 'math score', 'parental level of education')
dict = {"master's degree" : 6, "associate's degree": 5, "bachelor's degree": 4, "some college": 3, "some high school": 2, "high school": 1}
new_df['parental level of education'] = new_df['parental level of education'].map(dict).astype(int)
new_df.head()
X = new_df.iloc[:, [0,1,2,3,4,6,7]].values
y = new_df.iloc[:, 5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print(acc_decision_tree)
random_forest = RandomForestRegressor(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest)
sc = StandardScaler()
new_df[['math score', 'reading score', 'writing score']] = sc.fit_transform(new_df[['math score', 'reading score', 'writing score']])
new_df = pd.get_dummies(new_df, columns=['race/ethnicity', 'parental level of education'], drop_first=True)
new_df.head()
y = new_df['math score'].values
new_df.drop(['math score'], axis=1, inplace=True)
X = new_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
def create_polynomial_regression_model(X_train, X_test, y_train, y_test, degree):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_train_predicted = poly_model.predict(X_train_poly)
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)
  
    print("The model performance for the training set")
    print("-------------------------------------------")
    print("RMSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))
    
    print("\n")
    
    print("The model performance for the test set")
    print("-------------------------------------------")
    print("RMSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))
create_polynomial_regression_model(X_train, X_test, y_train, y_test, 2)
X_train.shape
model = Sequential()
model.add(Dense(128, input_dim=14))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
predictions = model.predict(X_test)
res = 0
for i in range(len(y_test)):
    val = (y_test[i] - predictions[i])*(y_test[i] - predictions[i])
    res += val
res /= len(y_test)
print("Mean squared error on the test set: ", res)
XGBModel = XGBRegressor()
XGBModel.fit(X_train, y_train, verbose=False)

XGBpredictions = XGBModel.predict(X_test)
MAE = mean_absolute_error(y_test, XGBpredictions)
print('XGBoost test MAE = ',MAE)
