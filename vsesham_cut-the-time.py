""" importing required packages """

%matplotlib inline



""" packages for data manipulation, munging and merging """

import pandas as pd

import numpy as np



""" packages for visualiztion and exploratory analysis """

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)



""" packages for running machine learning algorithms """

from time import time

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.svm import SVR, LinearSVC

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.cluster import FeatureAgglomeration

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
""" file paths """

train_file = '../input/train.csv'

test_file = '../input/test.csv'



""" function to read training data into pandas dataframe """

def read_training_data(filepath):

    train_df = pd.read_csv(filepath)

    return train_df



""" function to read test data into pandas dataframe """

def read_test_data(filepath):

    test_df = pd.read_csv(filepath)

    return test_df



""" calling function to read training data into a dataframe """

train_df = read_training_data(train_file)

test_df = read_test_data(test_file)
def data_summary(df):

    print(df.shape)

    print(df.info())   

    print(df.head(5))



""" calling functions to print summary statistics of training data """

data_summary(train_df)

data_summary(test_df)
def check_missing_values(df):

    if df.isnull().any().any():

        print("There are missing values in the dataframe")

    else:

        print("There are no missing values in the dataframe")

        

""" calling functions to check missing values on training and test datasets """

check_missing_values(train_df)

check_missing_values(test_df)
def initial_datatype_conversion(df):

    cols = ['X0','X1','X2','X3','X4','X5','X6','X8']

    for col in cols:

        df[col] = df[col].astype('category')

    return df



""" datatype conversion """

ret_train_df = initial_datatype_conversion(train_df)

ret_test_df = initial_datatype_conversion(test_df)



""" combining categorical attributes from training and test datasets """

train_df_cat = ret_train_df.loc[:,['X0','X1','X2','X3','X4','X5','X6','X8']]

test_df_cat = ret_test_df.loc[:,['X0','X1','X2','X3','X4','X5','X6','X8']]

train_df_cat = train_df_cat.add_prefix('train_')

test_df_cat = test_df_cat.add_prefix('test_')

combined = train_df_cat.append(test_df_cat, ignore_index=True)
def visualize_categories(df, **kwargs):

    row = kwargs.get('row',None)

    col = kwargs.get('col',None)

    hue = kwargs.get('hue',None)

    

    df_types = ['train_','test_']

    y_val = ['X0','X1','X2','X3','X4','X5','X6','X8']

    for df_type in df_types:

        for val in y_val:

            yval = df_type + val

            plt.figure()

            sns.countplot(y=yval, data=df, color="c");

            plt.show()



visualize_categories(combined)
""" converting training dataset object categorical values to numerical categorical types """

le = LabelEncoder()

cols = ['X0', 'X1', 'X2','X3','X4','X5','X6','X8']

#for col in cols:

#    ret_train_df[col] = ret_train_df[col].astype('category')

#    ret_test_df[col] = ret_test_df[col].astype('category')

#    

#    ret_train_df[col] = le.fit_transform(ret_train_df[col])

#    ret_test_df[col] = le.fit_transform(ret_test_df[col])

    

ret_train_df = pd.get_dummies(ret_train_df, columns=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'], prefix=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'])

ret_test_df = pd.get_dummies(ret_test_df, columns=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'], prefix=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'])
""" data preparation for modeling and prediction steps """

cols = ret_test_df.filter(like='_bb').columns

train_X = ret_train_df.drop(['ID','y'], axis=1)

train_Y = ret_train_df['y']

train_Y = train_Y.values

test_X = ret_test_df.drop(['ID'],axis=1)

test_X = test_X.drop(cols, axis=1)



#train_X = train_X.loc[:, train_X.var()>0.01]

#test_X = test_X.loc[:, test_X.var()>0.01]



matching_cols = train_X.columns.intersection(test_X.columns)

matching_cols_list = matching_cols.tolist()



test_X = test_X[matching_cols_list]

train_X = train_X[matching_cols_list]



#test_X = test_X.loc[:, 'X0_a':]

#train_X = train_X.loc[:, 'X0_a':]



X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.33, random_state=42)



print(X_train.head())

print(X_test.head())

print(y_train)

print(y_test)
def perform_svc(train_X, train_Y, test_X, test_Y):

    svr_clf = SVR()

    svr_clf.fit(X=train_X, y=train_Y)

    pred_Y = svr_clf.predict(test_X)

    r2_score_svc = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(svr_clf.score(train_X, train_Y) * 100, 2)

    returnval = {'model':'SVR', 'r2_score':r2_score_svc}

    return returnval

#perform_svc(train_X, train_Y, test_X)
def perform_rfc(df_X, df_Y, test_df_X, test_Y):



    rfr_clf = RandomForestRegressor(n_estimators = 100 ,oob_score=True, max_features="auto")

    rfr_clf.fit(df_X, df_Y)

    pred_Y = rfr_clf.predict(test_df_X)

    r2_score_rfc = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(rfr_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'RandomForestRegressor','r2_score':r2_score_rfc}

    return returnval

#perform_rfc(train_X, train_Y, test_X)
def perform_knn(df_X, df_Y, test_df_X, test_Y):

    knn = KNeighborsRegressor(n_neighbors=5)

    knn.fit(df_X, df_Y)

    pred_Y = knn.predict(test_df_X)

    r2_score_knn = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(knn.score(df_X, df_Y) *100,2)

    returnval = {'model':'KNeighborsRegressor','r2_score':r2_score_knn}

    #print (returnval)

    return returnval

#perform_knn(train_X, train_Y, test_X)
def perform_linear_regression(df_X,df_Y, test_X, test_Y):

    regr = LinearRegression()

    regr.fit(df_X, df_Y)

    pred_Y = regr.predict(test_X)

    r2_score_lr = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(regr.score(df_X, df_Y) *100,2)

    returnval = {'model':'LinearRegressor', 'r2_score':r2_score_lr}

    return returnval

#perform_linear_regression(train_X, train_Y, test_X)
def perform_linear_lasso(df_X, df_Y, test_X, test_Y):

    clf = Lasso(alpha=1.0)

    clf.fit(df_X, df_Y)

    pred_Y = clf.predict(test_X)

    r2_score_ll = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(clf.score(df_X, df_Y) *100,2)

    returnval = {'model':'Lasso','r2_score':r2_score_ll}

    return returnval

#perform_linear_lasso(train_X, train_Y, test_X)
def perform_ridge_regression(df_X, df_Y, test_X, test_Y):

    clf = Ridge(alpha=1.0)

    clf.fit(df_X, df_Y)

    pred_Y = clf.predict(test_X)

    r2_score_rr = r2_score(test_Y, pred_Y)

    accuracy = round(clf.score(df_X, df_Y) *100,3)

    returnval = {'model':'RidgeRegression','r2_score':r2_score_rr}

    return returnval

#perform_ridge_regression(train_X, train_Y, test_X)
def perform_elastinet_regression(df_X, df_Y, test_X, test_Y):

    clf = ElasticNet(alpha=0.1, l1_ratio=0.7)

    clf.fit(df_X, df_Y)

    pred_Y = clf.predict(test_X)

    r2_score_rr = round(r2_score(test_Y, pred_Y),3)

    accuracy = round(clf.score(df_X, df_Y) *100,2)

    returnval = {'model':'ElasticNet','r2_score':r2_score_rr}

    return returnval
def perform_pca(train_X, train_Y, test_X, test_Y):

    pca_model_accuracies = pd.DataFrame()

    pca = PCA(n_components = 10,svd_solver='randomized',whiten=True)

    pca.fit(train_X)

    train_X_pca = pca.transform(train_X)

    test_X_pca = pca.transform(test_X)



    svc_acc_val = perform_svc(train_X_pca, train_Y, test_X_pca, test_Y)



    rfc_acc_val = perform_rfc(train_X_pca, train_Y, test_X_pca, test_Y)



    knn_acc_val = perform_knn(train_X_pca, train_Y, test_X_pca, test_Y)

    

    lr_acc_val = perform_linear_regression(train_X_pca, train_Y, test_X_pca, test_Y)

    lc_acc_val = perform_linear_lasso(train_X_pca, train_Y, test_X_pca, test_Y)

    rr_acc_val = perform_ridge_regression(train_X_pca, train_Y, test_X_pca, test_Y)

    enet_acc_val = perform_elastinet_regression(train_X_pca, train_Y, test_X_pca, test_Y)

    

    pca_model_accuracies = pca_model_accuracies.append([svc_acc_val,rfc_acc_val,knn_acc_val,lr_acc_val,

                                                       lc_acc_val,rr_acc_val,enet_acc_val])

    cols = list(pca_model_accuracies.columns.values)

    cols = cols[-1:] + cols[:-1]

    pca_model_accuracies = pca_model_accuracies[cols]

    pca_model_accuracies = pca_model_accuracies.sort_values(by='r2_score')

    return pca_model_accuracies
def perform_feature_agglomeration(train_X, train_Y, test_X, test_Y):

    n_clusters = [32]

    fagg_model_accuracies = pd.DataFrame()

    for n_cluster in n_clusters:

        agglo = FeatureAgglomeration(connectivity=None, n_clusters=n_cluster)

        agglo.fit(train_X)

        train_X_reduced = agglo.transform(train_X)

        test_X_reduced = agglo.transform(test_X)

        

        svc_acc_val = perform_svc(train_X_reduced, train_Y, test_X_reduced, test_Y)

                

        rfc_acc_val = perform_rfc(train_X_reduced, train_Y, test_X_reduced, test_Y)

                

        knn_acc_val = perform_knn(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

        lr_acc_val = perform_linear_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

        lc_acc_val = perform_linear_lasso(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

        rr_acc_val = perform_ridge_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

        enet_acc_val = perform_elastinet_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

        fagg_model_accuracies = fagg_model_accuracies.append([svc_acc_val,rfc_acc_val,knn_acc_val,

                                                              lr_acc_val,lc_acc_val,rr_acc_val,enet_acc_val])

        cols = list(fagg_model_accuracies.columns.values)

        cols = cols[-1:] + cols[:-1]

        fagg_model_accuracies = fagg_model_accuracies[cols]

        fagg_model_accuracies = fagg_model_accuracies.sort_values(by='r2_score')

    return fagg_model_accuracies
def perform_variance_threshold(train_X, train_Y, test_X, test_Y):

    vt_model_accuracies = pd.DataFrame()

    selector = VarianceThreshold(threshold=0.03)

    selector.fit(train_X)

    train_X_reduced = selector.transform(train_X)

    test_X_reduced = selector.transform(test_X)

    

    svc_acc_val = perform_svc(train_X_reduced, train_Y, test_X_reduced, test_Y)

                

    rfc_acc_val = perform_rfc(train_X_reduced, train_Y, test_X_reduced, test_Y)

                

    knn_acc_val = perform_knn(train_X_reduced, train_Y, test_X_reduced, test_Y)

    

    lr_acc_val = perform_linear_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

    lc_acc_val = perform_linear_lasso(train_X_reduced, train_Y, test_X_reduced, test_Y)

        

    rr_acc_val = perform_ridge_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

    

    enet_acc_val = perform_elastinet_regression(train_X_reduced, train_Y, test_X_reduced, test_Y)

                

    vt_model_accuracies = vt_model_accuracies.append([svc_acc_val,rfc_acc_val,knn_acc_val,

                                                     lr_acc_val,lc_acc_val,rr_acc_val,enet_acc_val])

    cols = list(vt_model_accuracies.columns.values)

    cols = cols[-1:] + cols[:-1]

    vt_model_accuracies = vt_model_accuracies[cols]

    vt_model_accuracies = vt_model_accuracies.sort_values(by='r2_score')

    return vt_model_accuracies
""" calling function to perform principal component analysis """

pca_model_accuracies = perform_pca(X_train, y_train, X_test, y_test)

print(pca_model_accuracies)
""" calling function to perform feature agglomeration """

fagg_model_accuracies = perform_feature_agglomeration(X_train, y_train, X_test, y_test)

print(fagg_model_accuracies)
""" calling function to perform variance threshold analysis """

vt_model_accuracies = perform_variance_threshold(X_train, y_train, X_test, y_test)

print(vt_model_accuracies)
def write_to_csv(train_X, train_Y, test_X, test_df):

    

    print(train_X.shape)

    print(test_X.shape)

    

    #pca = PCA(n_components = 10,svd_solver='randomized')

    #pca.fit(train_X)

    #train_X_reduced = pca.transform(train_X)

    #test_X_reduced = pca.transform(test_X)

    

    #agglo = FeatureAgglomeration(connectivity=None, n_clusters=32)

    #agglo.fit(train_X)

    #train_X_reduced = agglo.transform(train_X)

    #test_X_reduced = agglo.transform(test_X)

    

    selector = VarianceThreshold(threshold=0.03)

    selector.fit(train_X)

    train_X_reduced = selector.transform(train_X)

    test_X_reduced = selector.transform(test_X)

    

    #clf = RandomForestRegressor(n_estimators = 100 ,oob_score=True, max_features=None)

    #clf.fit(train_X_reduced, train_Y)

    #pred_Y = rfr_clf.predict(test_X_reduced)

    

    clf = Ridge(alpha=1.0)

    clf.fit(train_X_reduced, train_Y)

    pred_Y = clf.predict(test_X_reduced)

    

    pred_Y_list = pred_Y.tolist()

    test_X['y'] = pred_Y_list

    test_X['ID'] = test_df.loc[:,'ID']

    final_df = test_X.loc[:,['ID','y']]

    final_df.to_csv('submission_1.csv',sep=',',index=False)

write_to_csv(train_X, train_Y, test_X, ret_test_df)