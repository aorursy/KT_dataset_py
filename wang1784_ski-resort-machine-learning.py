import pandas as pd

import re

import numpy as np

import matplotlib.pyplot as plt

from statistics import mode, mean, median

import seaborn as sns
read = pd.read_csv('ski_main')

useless_cols = ['link', 'Uniquenumber', 'Resort name', 'Ski map', 'E-mail', 'Website', 

                'Current piste quality', 'Youtube url', 'Skiresort.info url', 

                'Telephone', 'HP', 'SKIMAP', 'Total lifts open', 'Current upper snow depth', 

                'Current lower snow depth', 'Current fresh snow']

df_analysis = read.drop(useless_cols, axis = 1)

print(read.columns)
#split strings in season column to starting and ending season

noreport = list()

seasonstart = list()

seasonend = list()

for each in df_analysis['Season']:

    eachsplit = each.split(' - ')

    if len(eachsplit) == 2:

        seasonstart.append(eachsplit[0])

        seasonend.append(eachsplit[1])

    elif len(eachsplit) > 2:

        splitagain = eachsplit[1].split(' ')

        seasonstart.append(eachsplit[0])

        seasonend.append(splitagain[0])

    else:

        seasonstart.append(float('nan'))

        seasonend.append(float('nan'))

df_analysis['seasonstart'] = seasonstart

df_analysis['seasonend'] = seasonend

df_analysis = df_analysis.drop('Season', axis = 1)
#drop columns that have only 1 value or if the entire column is null or if each row has a unique value

for each in df_analysis.columns:

    if ((df_analysis[each].isnull()).sum == df_analysis.shape[0]) or (

        len(df_analysis[each].unique()) == 1):

        df_analysis = df_analysis.drop(each, axis = 1)

        print(each)
#check for columns that have nan values

col_nan = list(df_analysis.columns[(df_analysis.isnull()).sum() != 0])

print(col_nan)
#substitude the nan values with mode since all the nan columns have strings

for each in col_nan:

    not_nan = df_analysis.loc[df_analysis[each].notnull(), each]

    col_mode = mode(not_nan)

    df_analysis.loc[df_analysis[each].isnull(), each] = col_mode

print(((df_analysis.isnull()).sum() == 0).all()) #check if every column has zero nans
plt.scatter('Regions', 'Elevation', data = df_analysis)

plt.xticks(rotation = 90)

plt.xlabel('Regions')

plt.ylabel('Elevation')

plt.show()



plt.scatter('Regions', 'Ski pass price adult', data = df_analysis)

plt.xticks(rotation = 90)

plt.xlabel('Regions')

plt.ylabel('Ski Price Adult')

plt.show()



plt.scatter('Beginner slopes', 'Ski pass price adult', data = df_analysis)

plt.scatter('Intermediate slopes', 'Ski pass price adult', data = df_analysis)

plt.scatter('Difficult slopes', 'Ski pass price adult', data = df_analysis)

plt.xlabel('Beginner slopes')

plt.ylabel('Ski pass price adult')

# plt.legend('Beginner slopes', 'Intermediate slopes', 'Difficult slopes')

plt.show()



plt.scatter('User rating', 'Ski pass price adult', data = df_analysis)

plt.xlabel('rating')

plt.ylabel('price')

plt.show()



plt.hist('seasonstart', data = df_analysis)

plt.show()
#from plot above, there seems to be a lot of 0a ratings, which means that the resort does not have rating

if (df_analysis['User rating'] == '0a').sum()/df_analysis.shape[0] > 0.6:

    df_analysis = df_analysis.drop('User rating', axis = 1)

    print('dropped user rating')
# print(df_analysis.dtypes)

test = df_analysis.copy()

df_model = test.loc[:, test.dtypes != object]

for each in test.columns:

#     print(df_analysis[each].dtype)

    if test[each].dtype == object:

        test[each] = test[each].astype('category')

#         print(test[each].cat.codes)

        dummy = test[each].cat.codes

        df_model[each] = dummy
# from mlxtend.plotting import heatmap

# import mlxtend

cols = df_model.columns

cm = np.corrcoef(df_model[cols].values.T)

# hm = heatmap(cm, row_names = cols, column_names = cols)

# plt.show



sns.heatmap(cm)

print(df_model.columns[2:15])

print(df_model.columns[15:])
df_model.head()
adult_price_mean = (df_model['Ski pass price adult']).mean()

kid_price_mean = (df_model['Ski pass price children']).mean()

df_model.loc[df_model['Ski pass price adult'] == 0, 'Ski pass price adult'] = adult_price_mean

df_model.loc[df_model['Ski pass price children'] == 0, 'Ski pass price children'] = kid_price_mean

# adult_price_max = max(df_model['Ski pass price children'])

print(adult_price_mean, kid_price_mean)

print((df_model['Ski pass price adult']).std(), (df_model['Ski pass price children']).std())

# print(ss.fit_transform(df_model))

print('mean and median of adult price: ', mean(df_model['Ski pass price adult']), median(df_model['Ski pass price adult']))
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.model_selection import cross_val_score

import sklearn.metrics as metrics

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures
ycol = ['Ski pass price adult', 'Ski pass price children'] #predict two values

y1col = ['Ski pass price adult']

xcol = list(df_model.columns.values) #all other variables

for each in ycol:

    xcol.remove(each) #remove the columns that are y

    

#categorical df

y_cat = df_model['Ski pass price adult']

x_cat = df_model[xcol]

x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(x_cat, y_cat, test_size = 0.2)

df_cat = df_model



#continuous

ss = StandardScaler()

df_fit = ss.fit(df_model)

ymean = df_fit.mean_[2:4] #column 2, 3 are means for price adult and price children

ystd = df_fit.var_[2:4] ** (1/2) #column 2, 3 are means for price adult and price children

df_model = pd.DataFrame(ss.fit_transform(df_model), columns = cols)

# y = df_model['Ski pass price adult']

y = df_model[ycol]

x = df_model[xcol]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

y1 = df_model['Ski pass price adult']

x1_train, x1_test, y1_train, y1_test = train_test_split(x, y1, test_size = 0.2, random_state = 0)
#linear regression

lr = linear_model.LinearRegression()

scores = cross_val_score(estimator = lr, X = x, y = y, cv=5, scoring = 'neg_mean_squared_error')

print(scores)
# #linear regression

# pipe_lr = make_pipeline(StandardScaler(),

#                        linear_model.LinearRegression())

# scores = cross_val_score(estimator = pipe_lr, X = x, y = y, cv=5, scoring = 'neg_mean_squared_error')

# print(scores)
#linear regression with loop

fold = 5

byrow = df_model.sample(frac = 1) #set the split by row

folds = np.array_split(byrow, 5) #split dataframe into 5 so that 20% is testing and 80% is training

mses = list() #record mse of each fold



for eachfold in range(fold): #for each number of fold

    fold_num = list(np.arange(0, fold)) #create list of indices to refer which dataframe in folds is testing

    fold_num.pop(eachfold) #pop current fold number

#     print(fold_num)

    #form training and testing set

    test = folds[eachfold] #let current fold number be testing

    train_list = list() #list for list of subdataframes to form training

    for eachtrain in fold_num: #for each training fold number 

        train_list.append(folds[eachtrain]) #append the training sub-dataframe

    train = pd.concat(train_list) #combine all the training data

    

    #get x and y

    y_train_lr = train[ycol[0]]

    x_train_lr = train[xcol]

    y_test_lr = test[ycol[0]]

    x_test_lr = test[xcol]

    

    #modeling

    lr = linear_model.LinearRegression()

    lr.fit(x_train_lr, y_train_lr)

    mse = mean_squared_error(y_test_lr, lr.predict(x_test_lr))

    mses.append(mse)

    

print('mean of MSEs of all the folds are: ', mean(mses))

print((mean(mses)) ** (1/2) * ystd[0] + ymean[0])
lr.summary
#poly regression

degrees = np.arange(1, 5)

mean_error = list()

for each in degrees:

    pipe_poly = make_pipeline(PolynomialFeatures(degree = each),

                              linear_model.LinearRegression())

    scores = cross_val_score(estimator = pipe_poly, X = df_model[xcol], y = df_model[ycol], cv=5, scoring = 'neg_mean_squared_error')

    mean_error.append(mean(scores))

plt.scatter(degrees, mean_error)

plt.xlabel('Degrees')

plt.ylabel('mean squared error')

plt.show()
from sklearn.preprocessing import scale

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, MultiTaskLasso, MultiTaskLassoCV
#set up grid for lambda, used in both ridge and lasso

grid = 10**np.linspace(10, -2, 100)
#both adult and children prices

ridgecv = RidgeCV(alphas = grid, normalize = True, cv = 5, scoring = 'neg_mean_squared_error')

ridgecv.fit(x_train, y_train)

print('best alpha:', ridgecv.alpha_)

ridgebest = Ridge(alpha = ridgecv.alpha_, normalize = True)

ridgebest.fit(x_train, y_train)

print('testing mean squared error:', mean_squared_error(y_test, ridgebest.predict(x_test)))

ridge_coef = pd.DataFrame({'adult price':ridgebest.coef_[0], 'children price':ridgebest.coef_[1]}, index = xcol)

print(ridge_coef)
#just adult price

ridgecv = RidgeCV(alphas = grid, normalize = True, cv = 5, scoring = 'neg_mean_squared_error')

ridgecv.fit(x_train, y1_train)

print('best alpha:', ridgecv.alpha_)

ridgebest = Ridge(alpha = ridgecv.alpha_, normalize = True)

ridgebest.fit(x_train, y1_train)

print('testing mean squared error:', mean_squared_error(y1_test, ridgebest.predict(x_test)))

ridge_coef = pd.DataFrame({'adult price':ridgebest.coef_}, index = xcol)

print(ridge_coef)
lassocv = MultiTaskLassoCV(alphas = grid, normalize = True, cv = 5, max_iter = 5000)

lassocv.fit(x_train, y_train)

print(lassocv.alpha_)

lasso = MultiTaskLasso(alpha = lassocv.alpha_, normalize = True, max_iter = 5000)

lasso.fit(x_train, y_train)

print(mean_squared_error(y_test, lasso.predict(x_test)))

coeffs = pd.DataFrame(lasso.coef_, columns = x.columns, index = ['adult price', 'children price']).transpose()

print('do target variable have the same nonzero parameters:', ((coeffs['adult price'] == 0) == (coeffs['children price'] == 0)).all())

print(coeffs[coeffs['adult price']>0])
lassocv = LassoCV(alphas = grid, normalize = True, cv = 5)

lassocv.fit(x1_train, y1_train)

print('best alpha: ', lassocv.alpha_)

lasso = Lasso(alpha = lassocv.alpha_, normalize = True)

lasso.fit(x1_train, y1_train)

mse = mean_squared_error(y1_test, lasso.predict(x_test))

mse_dollar = mse ** (1/2) * ystd[0] + ymean[0]

print('MSE: ', mse, '\nMSE in dollar amount: ', mse_dollar)

coeffs1 = pd.Series(lasso.coef_, index = x.columns)

# coeffs = pd.DataFrame(lasso.coef_, columns = x.columns).transpose()

print(coeffs1[coeffs1!=0])

# print(coeffs1)
lasso.summary
from sklearn.svm import SVR, SVC
#split into train and test

svr_train, svr_test = train_test_split(df_model, test_size = 0.2, random_state = 1)

# svm_train, svm_test = train_test_split(df_svm, test_size = 0.2, random_state = 1)

# x_svm = svm_test[xcol]

# y_svm = svm_test['Ski pass price adult']



#split svr into 5 subdataframes for CV

fold = 10

svr_dfs = svr_train.sample(frac = 1, random_state = 1)

svr_folds = np.array_split(svr_dfs, fold)



# #split svm into 5 subdataframes for cross validation

# byrow = svm_train.sample(frac = 1, random_state = 1) #set the split by row

# folds = np.array_split(byrow, 5) #split dataframe into 5 so that 20% is testing and 80% is training
#rbf

gamma = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5] #range of gamma

# c = [0.00001, 0.0001, 0.001, 0.01, 0.1] #range of c

c = [1, 5, 10, 50, 100, 200, 300, 400, 500] #c for CV

mses = list() #store mse for each gamma



for eachgamma in gamma:

    mse_gamma = list()

    for eachc in c:

        mse_fold = list() #store mse for current gamma, c and each fold

        for eachfold in range(fold): #for each number of fold

            fold_num = list(np.arange(0, fold)) #create list of indices to refer which dataframe in folds is testing

            fold_num.pop(eachfold) #pop current fold number



            #form training and testing set

            test = svr_folds[eachfold] #let current fold number be testing

            train_list = list() #list for list of subdataframes to form training

            for eachtrain in fold_num: #for each training fold number 

                train_list.append(svr_folds[eachtrain]) #append the training sub-dataframe

            train = pd.concat(train_list) #combine all the training data



            #get x and y

            y_train_svr = train[ycol[0]]

            x_train_svr = train[xcol]

            y_test_svr = test[ycol[0]]

            x_test_svr = test[xcol]

            

            #modeling

            svr = SVR(gamma = eachgamma, C = eachc)

            svr.fit(x_train_svr, y_train_svr)

            mse = ((y_train_svr - svr.predict(x_train_svr)) ** 2).sum()/len(y_train_svr)

            mse_fold.append(mse) #store mse of current fold

        mse_gamma.append(mean(mse_fold)) #store the mean mse for current gamma and c

    mses.append(mse_gamma)



#plotting 

min_mse_list = list()

for eachmse in range(len(mses)):

    plt.scatter(c, mses[eachmse], label = gamma[eachmse])

    min_mse_list.append(min(mses[eachmse]))

plt.xlabel('c')

plt.ylabel('mse')

plt.title('MSE vs C per gamma')

plt.legend(title = 'gamma', loc = 'lower right')

plt.show()



#best gamma

min_mse = min(min_mse_list) #overall min value

best_gamma = gamma[min_mse_list.index(min_mse)] #find the gamma that has the min mse

best_gamma_mse = mses[gamma.index(best_gamma)] #get all the mse of best gamma

best_c = c[best_gamma_mse.index(min_mse)] #get best c in best gamma

print('best gamma is: ', best_gamma, '\nbest c is: ', best_c)



#investigate gamma = 2

# c = np.arange(0.001, 0.1, 1e-3) #for new range of c, where mse declines

# mses = list()

# for eachc in c:

#     svr = SVR(gamma = 2, C = eachc)

#     mse = cross_val_score(svr, X = x_train, y = y1_train, cv=5, scoring = 'neg_mean_squared_error')

#     if (mse<0).any():

#         print(mse)

#     mses.append(mean(mse))

# minmse = min(mses)

# minc = c[mses.index(minmse)]

# plt.plot(c, mses)

# plt.scatter(minc, minmse, color = 'red')

# plt.xlabel('c')

# plt.ylabel('mse')

# plt.title('C vs MSE when gamma = 2')

# plt.show()

# print('c that provides the smallest mse is: ', minc, '\nmin training mse is: ', minmse)

# #best c = 0.032



#get testing error for gamma = 2, c = minc = 0.032

x_svr = svr_test[xcol]

y_svr = svr_test['Ski pass price adult']

test_svr = SVR(gamma = best_gamma, C = best_c)

test_svr.fit(x_svr, y_svr)

test_mse = mean_squared_error(y_svr, test_svr.predict(x_svr))

mse_dollar = test_mse ** (1/2) * ystd[0] + ymean[0]

print('testing mse: ', test_mse, '\ntesting mse in dollar', mse_dollar)
from sklearn.metrics import accuracy_score
#try split by median

median_price = median(df_cat['Ski pass price adult'])

df_above_mean = df_cat[df_cat['Ski pass price adult'] > median_price]

print(df_above_mean.shape[0]/df_cat.shape[0] * 100) #will split dataframe by half
#split by median of adult price

df_svm = df_cat.copy()

df_svm = df_svm.drop('Ski pass price children', axis = 1) 

df_svm.loc[df_svm['Ski pass price adult'] <= median_price, 'Ski pass price adult'] = 0

df_svm.loc[df_svm['Ski pass price adult'] > median_price, 'Ski pass price adult'] = 1

df_svm['Ski pass price adult'] = df_svm['Ski pass price adult'].astype('category')



#split into train and test

svm_train, svm_test = train_test_split(df_svm, test_size = 0.2, random_state = 1)

x_svm = svm_test[xcol]

y_svm = svm_test['Ski pass price adult']



#split svm into 5 subdataframes for cross validation

fold = 5

byrow = svm_train.sample(frac = 1, random_state = 1) #set the split by row

folds = np.array_split(byrow, 5) #split dataframe into 5 so that 20% is testing and 80% is training



# #split into subdataframes

# fold = 5

# byrow = svm_train.sample(frac = 1, random_state = 1) #set the split by row

# folds = np.array_split(byrow, 5) #split dataframe into 5 so that 20% is testing and 80% is training

accuracies = list() #record mse of each fold



#modeling procedure

c = [0.001, 0.01, 0.1, 1, 2, 3, 5, 10, 20, 30, 40, 50, 100] #c for CV

for eachc in c:

    accuracy_c = list()

    for eachfold in range(fold): #for each number of fold

        fold_num = list(np.arange(0, fold)) #create list of indices to refer which dataframe in folds is testing

        fold_num.pop(eachfold) #pop current fold number



        #form training and testing set

        test = folds[eachfold] #let current fold number be testing

        train_list = list() #list for list of subdataframes to form training

        for eachtrain in fold_num: #for each training fold number 

            train_list.append(folds[eachtrain]) #append the training sub-dataframe

        train = pd.concat(train_list) #combine all the training data



        #get x and y

        y_train_svm = train[ycol[0]]

        x_train_svm = train[xcol]

        y_test_svm = test[ycol[0]]

        x_test_svm = test[xcol]



        #modeling

#     for eachc in c:

        svm = SVC(C = eachc, kernel='linear')

        svm.fit(x_train_svm, y_train_svm)

        accuracy = accuracy_score(y_test_svm, svm.predict(x_test_svm))

        accuracy_c.append(accuracy)

    accuracies.append(mean(accuracy_c))



#graphing accuracy VS c

maxaccuracy = max(accuracies) #find the highest accuracy

maxc = c[accuracies.index(maxaccuracy)] #find c that has highest accuracy



fig = plt.figure() #figure

ax = fig.add_subplot(111) #add ax to same figure

plt.scatter(c, accuracies)

plt.xlabel('c')

plt.ylabel('Mean Accuracy')

plt.title('Mean Accuracy of Each C value')

plt.scatter(maxc, maxaccuracy)

annotate_text = 'max c: ' + str("%.2f" % maxc) + '\nmax accuracy' + str("%.2f" % maxaccuracy)

ax.annotate(s = annotate_text, xy = (maxc, maxaccuracy))

plt.show()



#investigate maxc = 0.1

x_svm = svm_test[xcol]

y_svm = svm_test['Ski pass price adult']

svm_all = SVC(C = maxc, kernel = 'linear')

svm_all.fit(x_svm, y_svm)

accuracy_all = accuracy_score(y_svm, svm_all.predict(x_svm))

print('accuracy is: ', accuracy_all)
# def svmcv(fold, df, ycol, xcol, random = None):

'''performing CV for svm

    input:

    fold = number of folds wanted in CV

    df = the dataframe to use

    ycol = name of the column in df as response variable, only 1

    xcol = names of columns in df as predicting variables

    random = random_state number for splitting

    

    output:

    accuracy = mean accuracy of this CV

'''

kernel = ['linear', 'poly', 'rbf']

accuracy_kernel = list()

acc_calculated = list()

for eachkernel in kernel:

    accuracy_c = list()

    acc_cal_c = list()

    for eachc in c:

        byrow = svm_train.sample(frac = 1, random_state = 1)

        folds = np.array_split(byrow, 5)

        accuracy_fold = list()

        acc_cal_fold = list()

        for eachfold in range(fold): #for each number of fold

            fold_num = list(np.arange(0, fold)) #create list of indices to refer which dataframe in folds is testing

            fold_num.pop(eachfold) #pop current fold number



            #form training and testing set

            test = folds[eachfold] #let current fold number be testing

            train_list = list() #list for list of subdataframes to form training

            for eachtrain in fold_num: #for each training fold number 

                train_list.append(folds[eachtrain]) #append the training sub-dataframe

            train = pd.concat(train_list) #combine all the training data



            #get x and y

            y_train_svm = train[ycol[0]]

            x_train_svm = train[xcol]

            y_test_svm = test[ycol[0]]

            x_test_svm = test[xcol]

            

            #modeling

            svm = SVC(C = eachc, kernel = eachkernel, gamma = 'auto')

            svm.fit(x_train_svm, y_train_svm)

            accuracy = accuracy_score(y_test_svm, svm.predict(x_test_svm))

            accuracy_fold.append(accuracy)

            correct_count = (y_test_svm == svm.predict(x_test_svm)).sum()

            acc_cal_fold.append(correct_count)

        accuracy_c.append(mean(accuracy_fold))

        acc_cal_c.append(mean(acc_cal_fold))

    accuracy_kernel.append(accuracy_c)

    acc_calculated.append(acc_cal_c)



#plotting accuracy VS c of each kernel

plt.scatter(c, accuracy_kernel[0], label = kernel[0])

plt.scatter(c, accuracy_kernel[1], label = kernel[1])

plt.scatter(c, accuracy_kernel[2], label = kernel[2])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Accuracy VS C by Kernel')

plt.legend()

plt.show()



#summary of graph

accuracies = list() #store the best accuracy of each kernel

for eachkernel in range(len(kernel)):

    accuracies.append(max(accuracy_kernel[eachkernel]))



max_accuracy = max(accuracies) #get the best accuracy among all kernels

max_kernel = kernel[accuracies.index(max_accuracy)] #get the kernel that has the best accuracy

k_accuracy = accuracy_kernel[kernel.index(max_kernel)] #get the accuracies of best kernel

max_c = c[k_accuracy.index(max_accuracy)] #find the c corresponding to the best kernel

print('best kernel: ', max_kernel, '\nmax training accuracy: ', max_accuracy, '\nmax c: ', max_c)



#investigate best c and best kernel 

svm_best = SVC(C = max_c, kernel = max_kernel, gamma = 'auto')

svm_best.fit(x_svm, y_svm)

# df_svm_coef = pd.DataFrame({'col':xcol, 'coef':svm_best.dual_coef_[0]})

# print(len(svm_best.dual_coef_[0]),x_svm.shape)

accuracy_best = accuracy_score(y_svm, svm_best.predict(x_svm))

print('testing accuracy: ', accuracy_best)
print(acc_calculated[1], acc_calculated[2])
from sklearn.ensemble import RandomForestClassifier as RFC
def tr_ts_fold(n_fold, folds, xcol, ycol):

    dfs = list()

    for eachfold in range(n_fold):

        fold_num = list(np.arange(0, n_fold)) #create list of indices to refer which dataframe in folds is testing

        fold_num.pop(eachfold) #pop current fold number



        #form training and testing set

        test = folds[eachfold] #let current fold number be testing

        train_list = list() #list for list of subdataframes to form training

        for eachtrain in fold_num: #for each training fold number 

            train_list.append(folds[eachtrain]) #append the training sub-dataframe

        train = pd.concat(train_list) #combine all the training data



        #get x and y

        y_train = train[ycol]

        x_train = train[xcol]

        y_test = test[ycol]

        x_test = test[xcol]

        

        #append dataframes for this fold

        fold_df = [x_train, y_train, x_test, y_test]

        dfs.append(fold_df)

    return dfs
# m = int(len(xcol)**(1/2)) #number of features considered at each split

n_bs = 0.5 #percentage of observations for bootstrapping

n_trees = np.arange(10, 110, 10)

depth = [2, 5, 10, 20, 30, 40, 50, 100] #depth of trees

accuracy = list()

df_fold = tr_ts_fold(5, folds, xcol, y1col[0])

for eachtree in n_trees:

    tree_accuracy = list()

    for eachdepth in depth:

        depth_accuracy = list()

        for eachfold in range(len(df_fold)): #for each number of fold

            #get train and test data for current fold

            current_fold = df_fold[eachfold]

            x_train_rf = current_fold[0]

            y_train_rf = current_fold[1]

            x_test_rf = current_fold[2]

            y_test_rf = current_fold[3]



            #modeling

            rfc = RFC(n_estimators = eachtree,

                      max_depth = eachdepth, 

                      max_features = 'sqrt', 

                      bootstrap = True, 

    #                   max_samples = n_bs, 

                      random_state = 0)

            rfc.fit(x_train_rf, y_train_rf)

            depth_accuracy.append(rfc.score(x_test_rf, y_test_rf)) #record accuracy of current fold

        tree_accuracy.append(mean(depth_accuracy)) #store mean fold accuracy as accuracy for current depth

    accuracy.append(tree_accuracy)



# #plot accuracy by depth

# max_accuracy_list = list()

# for each in range(len(accuracy)):

#     max_accuracy_list.append(max(accuracy[each]))

#     plt.scatter(depth, accuracy[each], label = n_trees[each])

# plt.legend(title = 'number of trees')

# plt.xlabel('depth of trees')

# plt.ylabel('accuracy')

# plt.title('Accuracy VS Depth of Trees by Number of Trees')

# plt.show()



# #finding the best depth and number of trees

# max_accuracy = max(max_accuracy_list)
#plot accuracy by depth

max_accuracy_list = list()

for each in range(len(accuracy)):

    max_accuracy_list.append(max(accuracy[each]))

    plt.scatter(depth, accuracy[each], label = n_trees[each])

plt.legend(title = 'number of trees')

plt.xlabel('depth of trees')

plt.ylabel('accuracy')

plt.title('Accuracy VS Depth of Trees by Number of Trees')

plt.show()



#finding the best depth and number of trees

max_accuracy = max(max_accuracy_list)

best_tree = n_trees[max_accuracy_list.index(max_accuracy)]

best_tree_accuracy = accuracy[max_accuracy_list.index(max_accuracy)]

best_depth = depth[best_tree_accuracy.index(max_accuracy)]

print('highest accuracy: ', max_accuracy, '\nbest depth: ', best_depth, '\nbest number of trees: ', best_tree)
#find the best depth of best number of trees

depth = np.arange(5, 20)

acc = list()

for eachdepth in depth: #for each new range of depth

    depth_acc = list()

    for eachfold in range(len(df_fold)): #for each number of fold

        #get train and test data for current fold

        current_fold = df_fold[eachfold]

        x_train_rf = current_fold[0]

        y_train_rf = current_fold[1]

        x_test_rf = current_fold[2]

        y_test_rf = current_fold[3]



        #modeling

        rfc = RFC(n_estimators = best_tree,

                  max_depth = eachdepth, 

                  max_features = 'sqrt', 

                  bootstrap = True, 

    #             max_samples = n_bs, 

                  random_state = 0)

        rfc.fit(x_train_rf, y_train_rf)

        depth_acc.append(rfc.score(x_test_rf, y_test_rf)) #record accuracy of current fold

    acc.append(mean(depth_acc))
#plotting

plt.scatter(depth, acc)

plt.xlabel('depth of trees')

plt.ylabel('accuracy')

plt.title('Accuracy VS Depth of Trees for 30 Trees in Forest')

best_acc = max(acc)

best_depth = depth[acc.index(best_acc)]

print('highest accuracy: ', best_acc, '\nbest depth: ', best_depth)



#fit data to model

rfc_test = RFC(n_estimators = best_tree, 

               max_depth = best_depth, 

               max_features = 'sqrt', 

               bootstrap = True, 

               random_state = 0)

rfc_test.fit(x_svm, y_svm)

print('testing accuracy: ', rfc_test.score(x_svm, y_svm))