import pandas as pd

import numpy as np

from statistics import mean

import matplotlib.pyplot as plt
df_read = pd.read_csv('../input/mental-health-in-tech-field-eda-cleaning/main_dummy2.csv')

df_dummies2 = df_read.drop('Unnamed: 0', axis = 1).drop('Have you been diagnosed with a mental health condition by a medical professional?_Yes', axis = 1)

df_read
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# df_dummies2['age'] = ss.fit_transform(np.array(df_dummies2['age']).reshape(-1,1))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
cols = df_dummies2.columns.values

ycol = 'Have you ever sought treatment for a mental health issue from a mental health professional?'

xcol = list()

for each in cols:

    if each != ycol:

        xcol.append(each)

y = df_dummies2[ycol]

x = df_dummies2[xcol]

x_whiten = ss.fit_transform(x)

x = pd.DataFrame(x_whiten, columns = xcol)

x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size = 0.2, random_state = 0)
n_fold = 5

x_folds2 = np.array_split(x_train2, n_fold) #split training data into n_fold proportions

y_folds2 = np.array_split(y_train2, n_fold) #split training data into n_fold proportions



cv_folds = list() #stores dataframes for each fold

for eachfold in range(n_fold): #for each fold

    train_number = list(np.arange(0, n_fold)) #a list of fold numbers

    train_number.pop(eachfold) #pop current fold number

#     print(train_number)

    df_y_ts = y_folds2[eachfold] #use current fold number as testing fold, create testing y

    df_x_ts = x_folds2[eachfold] #use current fold number as testing fold, create testing x

    y_train_list = list() #stores all the df y from folds for training

    x_train_list = list() #stores all the df x from folds for training

    for eachnumber in train_number: #for each training fold number

        x_train_list.append(x_folds2[eachnumber]) #append the df in training fold number for x

        y_train_list.append(y_folds2[eachnumber]) #append the df in training fold number for y

    df_x_tr = pd.concat(x_train_list) #combine all the training dfs for x into 1

    df_y_tr = pd.concat(y_train_list) #combine all the training dfs for y into 1

    cv_folds.append([df_x_tr, df_y_tr, df_x_ts, df_y_ts]) #append training and testing dataframe for current fold
from sklearn.feature_selection import chi2, SelectKBest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

import scipy.stats
from sklearn.decomposition import PCA
#calculate variance explained

var_exp_list = list() #store variance explained for n features/eigenvalues

for ncomp in range(1, x.shape[1]+1): #for each number of features/eigenvalues

    pca_comp = PCA(n_components = ncomp) #set the number of components wanted for pca

    pca_comp.fit(x)

    var_exp = sum(pca_comp.explained_variance_ratio_) #sum the variance explained of each component to get total variance explained

    var_exp_list.append(var_exp) #append total variance explained in list

    

#plot variance explained

plt.scatter(range(x.shape[1]), var_exp_list) #plot variance explained against number of features

plt.xlabel('number of eigenvalues/features')

plt.ylabel('% of variance explained')

plt.title('Variance explained')



#find the good number of eigenvalues

for eachvar in var_exp_list:

    if eachvar>0.95:

        good_var = eachvar

        break

good_n_eigen = var_exp_list.index(good_var)

plt.scatter(good_n_eigen, good_var)

plt.show()

print('number of eigenvalues above 0.95: ', good_n_eigen, good_var)
#get newly generated components

pca_best = PCA(n_components = good_n_eigen)

pca_best.fit(x)

pca_features = pca_best.fit_transform(x)

single_best = pca_best.singular_values_

df_pca = pd.DataFrame(pca_features, columns = list(range(good_n_eigen)))

print(df_pca.shape)
#use random forest to select features

n_trees = np.arange(10, 110, 10)

depth = [2, 5, 10, 20, 30, 40, 50, 100]

accuracy = list()



for eachn in n_trees:

    acc_n = list()

    for eachd in depth:

        acc_d = list()

        for eachfold in cv_folds:

            x_tr_fold = eachfold[0]

            y_tr_fold = eachfold[1]

            x_ts_fold = eachfold[2]

            y_ts_fold = eachfold[3]

            rf = RandomForestClassifier(n_estimators = eachn,

                                       max_depth = eachd, 

                                       bootstrap = True, 

                                       random_state = 0)

            rf.fit(x_tr_fold, y_tr_fold)

            acc_d.append(rf.score(x_ts_fold, y_ts_fold))

        acc_n.append(mean(acc_d))

    accuracy.append(acc_n)
#plot results

tree_acc_list = list()

for eachacc in range(len(accuracy)):

    plt.scatter(depth, accuracy[eachacc], label = n_trees[eachacc])

    tree_acc_list.append(max(accuracy[eachacc]))

plt.legend(title = 'number of trees')

plt.xlabel('depth of forest')

plt.ylabel('accuracy')

plt.show()



#find the best number of trees and depth

best_acc = max(tree_acc_list)

best_n = n_trees[tree_acc_list.index(best_acc)]

best_n_list = accuracy[tree_acc_list.index(best_acc)]

best_d = depth[best_n_list.index(best_acc)]

print('the highest accuracy is: ', best_acc, '\nbest number of trees is: ', best_n, '\nbest forest depth is: ', best_d)
# acc_n = accuracy[6:9]

# count = 6

# for eachn in acc_n:

#     plt.scatter(depth[3:6], eachn[3:6], label = n_trees[count])

#     count += 1

# plt.legend(title = 'number of trees')

# plt.xlabel('depth of forest')

# plt.ylabel('accuracy')

# plt.show()

#investigate new range of ntree and depth

n_trees_s = np.arange(45, 56)

depth_s = np.arange(1, 10)

accuracy_s = list()



for eachns in n_trees_s:

    acc_ns = list()

    print(eachns)

    for eachds in depth_s:

        acc_ds = list()

        for eachfold in cv_folds:

            x_tr_fold = eachfold[0]

            y_tr_fold = eachfold[1]

            x_ts_fold = eachfold[2]

            y_ts_fold = eachfold[3]

            rf = RandomForestClassifier(n_estimators = eachns,

                                       max_depth = eachds, 

                                       bootstrap = True, 

                                       random_state = 0)

            rf.fit(x_tr_fold, y_tr_fold)

            acc_ds.append(rf.score(x_ts_fold, y_ts_fold))

        acc_ns.append(mean(acc_ds))

    accuracy_s.append(acc_ns)
#plot results

best_acc_d = list()

for eachacc in range(len(accuracy_s)):

    plt.scatter(depth_s, accuracy_s[eachacc], label = n_trees_s[eachacc])

    best_acc_d.append(max(accuracy_s[eachacc]))

plt.legend(title = 'number of trees')

plt.xlabel('depth of forest')

plt.ylabel('accuracy')

plt.show()



#get the best accuracy, number of trees and depth

best_acc = max(best_acc_d)

best_n = n_trees_s[best_acc_d.index(best_acc)]

best_n_list = accuracy_s[best_acc_d.index(best_acc)]

best_d = depth_s[best_n_list.index(best_acc)]

print('the highest accuracy is: ', best_acc, '\nbest number of trees is: ', best_n, '\nbest forest depth is: ', best_d)
best_rf = RandomForestClassifier(n_estimators = best_n, 

                                 max_depth = best_d, 

                                 random_state = 0)

best_rf.fit(x_train2, y_train2)

select_feature = SelectFromModel(best_rf, prefit=True)

x_new = select_feature.transform(x)

print('original number of features: ', x.shape[1], '\nnumber of selected features: ', x_new.shape[1])

feature_index = select_feature.get_support()

feature_name = list(x.columns[feature_index])

print(feature_name)

# importances = best_rf.feature_importances_

# indices = np.argsort(importances)[::-1]



# # Print the feature ranking

# print("Feature ranking:")



# for f in range(x.shape[1]):

#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#test accuracy

best_rf.fit(x_test2[feature_name], y_test2)

print(best_rf.score(x_test2[feature_name], y_test2))
from sklearn.svm import SVC as svc
# kernel = ['linear', 'poly', 'rbf']

c = [0.01, 0.1, 1, 5, 10, 50, 100, 200] #c for all kernels

gamma = [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5] #for rbf and poly

degree = [2, 3, 4, 5] #for poly kernel

# kernel = ['linear', 'poly', 'rbf']
def svc_cv(cv_folds, c, kernel, gamma = [], degree = [], print_para = False):

    accuracy = list()

    n_fold = len(cv_folds)

    if kernel == 'linear':

        for eachc in c:

            linear_c = list()

            for eachfold in cv_folds:

                x_tr_fold = eachfold[0]

                y_tr_fold = eachfold[1]

                x_ts_fold = eachfold[2]

                y_ts_fold = eachfold[3]

                linear = svc(C = eachc, kernel = 'linear')

                linear.fit(x_tr_fold, y_tr_fold)

                linear_c.append(linear.score(x_ts_fold, y_ts_fold))

            accuracy.append(mean(linear_c))

    elif kernel == 'poly':

        for eachd in degree:

            poly_acc_g = list()

            if print_para:

                print('working on degree: ', eachd)

            for eachg in gamma:

                if print_para:

                    print('working on gamma: ', eachg)

                poly_acc_c = list()

                for eachc in c:

                    if print_para:

                        print('working on c: ', eachc)

                    poly_acc_fold = list()

                    for eachfold in cv_folds:

                        x_tr_fold = eachfold[0]

                        y_tr_fold = eachfold[1]

                        x_ts_fold = eachfold[2]

                        y_ts_fold = eachfold[3]

                        poly = svc(C = eachc, gamma = eachg, degree = eachd, kernel = 'poly')

                        poly.fit(x_tr_fold, y_tr_fold)

                        poly_acc_fold.append(poly.score(x_ts_fold, y_ts_fold))

                    poly_acc_c.append(mean(poly_acc_fold))

                poly_acc_g.append(poly_acc_c)

            accuracy.append(poly_acc_g)

    elif kernel == 'rbf':

        for eachg in gamma:

            rbf_acc_c = list()

            for eachc in c:

                rbf_acc_fold = list()

                for eachfold in cv_folds:

                    x_tr_fold = eachfold[0]

                    y_tr_fold = eachfold[1]

                    x_ts_fold = eachfold[2]

                    y_ts_fold = eachfold[3]

                    rbf = svc(C = eachc, gamma = eachg, kernel = 'rbf')

                    rbf.fit(x_tr_fold, y_tr_fold)

                    rbf_acc_fold.append(rbf.score(x_ts_fold, y_ts_fold))

                rbf_acc_c.append(mean(rbf_acc_fold))

            accuracy.append(rbf_acc_c)

    return accuracy
#train linear kernel

linear_acc = svc_cv(cv_folds, c, 'linear')

plt.scatter(c, linear_acc)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')

print(max(linear_acc), c[linear_acc.index(max(linear_acc))])
#focus on new range of c for linear

c_linear = np.arange(0.001, 0.02, 0.001)

linear_acc = svc_cv(cv_folds, c_linear, 'linear')

#plot

plt.scatter(c_linear, linear_acc)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')

plt.show()



#test

best_c_linear = c_linear[linear_acc.index(max(linear_acc))]

linear_test = svc(C = best_c_linear, kernel = 'linear')

linear_test.fit(x_test2, y_test2)

linear_test_acc = linear_test.score(x_test2, y_test2)

print('testing accuracy for linear kernel is: ', linear_test_acc, '\nbest c is: ', best_c_linear)
#train poly model

poly_acc = svc_cv(cv_folds, c, 'poly', gamma, degree)

# print(len(poly_acc))

for eachdegree in range(len(poly_acc)):

    degree_acc = poly_acc[eachdegree]

    for eachacc in range(len(poly_acc[eachdegree])):

        plt.scatter(c, degree_acc[eachacc], label = gamma[eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma')

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()
#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd in poly_acc: #for each degree

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma[each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c[best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
#new c and gamma

# c_new1 = np.arange(5, 15)

# gamma_new1 = np.arange(0.005, 0.015, 0.001)

# c_new2 = np.arange(45, 56)

# gamma_new2 = np.arange(0.005, 0.015, 0.001)

# c_new3 = np.arange(95, 106)

# gamma_new3 = np.arange(0.005, 0.015, 0.001)

# c_new4 = np.arange(0.05, 0.15, 0.01)

# gamma_new4 = np.arange(0.05, 0.15, 0.01)

c_new1 = np.arange(95, 106)

gamma_new1 = np.arange(0.0005, 0.0015, 0.0001)

c_new2 = np.arange(0.1, 2, 0.1)

gamma_new2 = np.arange(0.005, 0.015, 0.001)

c_new3 = np.arange(0.1, 2, 0.1)

gamma_new3 = np.arange(0.005, 0.015, 0.001)

c_new4 = np.arange(0.1, 2, 0.1)

gamma_new4 = np.arange(0.005, 0.015, 0.001)



c_new_all = [c_new1, c_new2, c_new3, c_new4]

gamma_new_all = [gamma_new1, gamma_new2, gamma_new3, gamma_new4]



#train with new hyperparameters

poly_all = list()

for eachnew in range(len(c_new_all)):

    pca_degree = int(eachnew+2)

    pca_degree = [pca_degree]

    new = svc_cv(cv_folds, c_new_all[eachnew], 'poly', gamma_new_all[eachnew], pca_degree) 

    poly_all.append(new)
# for eachdegree in range(len(poly_acc)):

#     degree_acc = poly_acc[eachdegree]

#     for eachacc in range(len(poly_acc[eachdegree])):

#         plt.scatter(c_poly, degree_acc[eachacc], label = gamma_poly[eachacc])

#     plt.xlabel('c')

#     plt.ylabel('accuracy')

#     plt.legend(title = 'gamma', loc = (1, 0))

#     plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

#     plt.show()

#plot new acc

for eachdegree in range(len(poly_all)):

    degree_acc = poly_all[eachdegree][0]

    for eachacc in range(len(degree_acc)):

#         print(len(c_new_pca[eachdegree]), c_new_pca[eachdegree], np.arange(0.005, 0.025, 0.005))

#         print(len(degree_acc), len(degree_acc[eachacc]))

#         print(len(gamma_new_pca[eachdegree]))

        plt.scatter(c_new_all[eachdegree], degree_acc[eachacc], label = gamma_new_all[eachdegree][eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma', loc = (1, 0))

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()

    

#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd_list in range(len(poly_all)): #for each degree

    eachd = poly_all[eachd_list][0]

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma_new_all[eachd_list][each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c_new_all[eachd_list][best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
# #find the best c and gamma

# degree_max_acc = list() #store the max accuracy for each degree

# best_g_c = list() #store the g and c values of the highest accuracy for each degree

# for eachd in poly_acc: #for each degree

#     each_g_acc = list() #store the best accuracy for each degree

#     for eachg in eachd: #for accuracy of each g value 

#         each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

#     degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

#     best_g = gamma_poly[each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

#     best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

#     best_c = c_poly[best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

#     best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

# best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

# print('max accuracy for each degree is: ', degree_max_acc)

# print(best_degree)

# print(best_g_c)



#get testing accuracy for the best 

poly_test_accuracies = list()

for eachgc in range(len(best_g_c)):

    current_g_c = best_g_c[eachgc]

    poly = svc(C = current_g_c[1], gamma = current_g_c[0], degree = degree[eachgc], kernel = 'poly')

    poly.fit(x_test2, y_test2)

    poly_test_acc = poly.score(x_test2, y_test2)

    poly_test_accuracies.append(poly_test_acc)

#     print(current_g_c)

# print(best_g_c)

print(poly_test_accuracies)

print('the best degree is: ', degree[poly_test_accuracies.index(max(poly_test_accuracies))], 

      '\nthe highest testing accuracy is: ', max(poly_test_accuracies))
# #get testing accuracy for the best 

# poly_test_accuracies = list()

# for eachgc in range(len(best_g_c)):

#     current_g_c = best_g_c[eachgc]

#     poly = svc(C = current_g_c[1], gamma = current_g_c[0], degree = degree[eachgc], kernel = 'poly')

#     poly.fit(x_test2, y_test2)

#     poly_test_acc = poly.score(x_test2, y_test2)

#     poly_test_accuracies.append(poly_test_acc)

# print(poly_test_accuracies)

# print('the best degree is: ', degree[poly_test_accuracies.index(max(poly_test_accuracies))], 

#       '\nthe highest testing accuracy is: ', max(poly_test_accuracies))
#train poly model

rbf_acc = svc_cv(cv_folds, c, 'rbf', gamma)
#plot accuracy

gamma_acc_list = list()

for eachacc in range(len(rbf_acc)):

    plt.scatter(c, rbf_acc[eachacc], label = gamma[eachacc])

    gamma_acc_list.append(max(rbf_acc[eachacc]))

max_acc = max(gamma_acc_list)

best_gamma = gamma[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_acc[gamma_acc_list.index(max_acc)]

best_c = c[best_gamma_acc.index(max_acc)]

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma')

plt.title('Accuracy for rbf kernel')

plt.show()

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#focus on new c and gamma

rbf_c = np.arange(0.5, 1.5, 0.1)

rbf_gamma = np.arange(0.005, 0.015, 0.001)

rbf_new = svc_cv(cv_folds, rbf_c, 'rbf', rbf_gamma)
#plot new rbf

gamma_acc_list = list()

for eachacc in range(len(rbf_new)):

    plt.scatter(rbf_c, rbf_new[eachacc], label = rbf_gamma[eachacc])

    gamma_acc_list.append(max(rbf_new[eachacc]))

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma', loc = (1.1, 0))

plt.title('Accuracy for rbf kernel')

plt.show()

max_acc = max(gamma_acc_list)

best_gamma = rbf_gamma[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_new[gamma_acc_list.index(max_acc)]

best_c = rbf_c[best_gamma_acc.index(max_acc)]

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#get testing accuracy for rbf

rbf_test = svc(C = best_c, gamma = best_gamma, kernel = 'rbf')

rbf_test.fit(x_test2, y_test2)

rbf_test_acc = rbf_test.score(x_test2, y_test2)

print('testing accuracy for rbf kernel is: ', rbf_test_acc)
#change from all features to selected features

selected_cv_folds = list()

for eachfold in cv_folds:

    selected_x_tr = eachfold[0][feature_name]

    selected_x_ts = eachfold[2][feature_name]

    selected_cv_folds.append([selected_x_tr, eachfold[1], selected_x_ts, eachfold[3]])

selectedx_test = x_test2[feature_name]
linear_acc = svc_cv(selected_cv_folds, c, 'linear')

plt.scatter(c, linear_acc)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')

print('highest accuracy is: ', max(linear_acc), '\nbest c is: ', c[linear_acc.index(max(linear_acc))])
#focus on new range of c

c_linear = np.arange(0.005, 0.015, 0.001)

linear_acc = svc_cv(selected_cv_folds, c_linear, 'linear')

plt.scatter(c_linear, linear_acc)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')



#best c

best_c_linear = c_linear[linear_acc.index(max(linear_acc))]

print(max(linear_acc), 'best c: ', best_c_linear)

# print(linear_acc)



#test

linear_test = svc(C = best_c_linear, kernel = 'linear')

linear_test.fit(selectedx_test, y_test2)

linear_test_acc = linear_test.score(selectedx_test, y_test2)

print('best testing accuracy: ', linear_test_acc)
# c_selected = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100]

# # c_selected = [0.001, 0.01, 0.1, 1]



# gamma_selected = [0.001, 0.01, 0.1, 1, 5, 10]
#train poly model

poly_acc = svc_cv(selected_cv_folds, c, 'poly', gamma, degree)

# print(len(poly_acc))

for eachdegree in range(len(poly_acc)):

    degree_acc = poly_acc[eachdegree]

    for eachacc in range(len(poly_acc[eachdegree])):

        plt.scatter(c, degree_acc[eachacc], label = gamma[eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma', loc = (1, 0))

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()
#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd in poly_acc: #for each degree

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma[each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c[best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
#new c and gamma

c_new1 = np.arange(45, 61)

gamma_new1 = np.arange(0.005, 0.015, 0.001)

c_new2 = np.arange(150, 260, 10)

gamma_new2 = np.arange(0.005, 0.015, 0.001)

c_new3 = np.arange(0.05, 0.15, 0.01)

gamma_new3 = np.arange(0.05, 0.15, 0.01)

c_new4 = np.arange(0.05, 0.15, 0.01)

gamma_new4 = np.arange(0.05, 0.15, 0.01)



c_new_selected = [c_new1, c_new2, c_new3, c_new4]

gamma_new_selected = [gamma_new1, gamma_new2, gamma_new3, gamma_new4]



#train with new hyperparameters

poly_selected = list()

for eachnew in range(len(c_new_selected)):

    pca_degree = int(eachnew+2)

    pca_degree = [pca_degree]

    new = svc_cv(selected_cv_folds, c_new_selected[eachnew], 'poly', gamma_new_selected[eachnew], pca_degree) 

    poly_selected.append(new)
#plot new acc

for eachdegree in range(len(poly_selected)):

    degree_acc = poly_selected[eachdegree][0]

    for eachacc in range(len(degree_acc)):

        plt.scatter(c_new_selected[eachdegree], degree_acc[eachacc], label = gamma_new_selected[eachdegree][eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma', loc = (1, 0))

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()

    

#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd_list in range(len(poly_selected)): #for each degree

    eachd = poly_selected[eachd_list][0]

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma_new_selected[eachd_list][each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c_new_selected[eachd_list][best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
#get testing accuracy for the best 

poly_test_accuracies = list()

for eachgc in range(len(best_g_c)):

    current_g_c = best_g_c[eachgc]

    poly = svc(C = current_g_c[1], gamma = current_g_c[0], degree = degree[eachgc], kernel = 'poly')

    poly.fit(selectedx_test, y_test2)

    poly_test_acc = poly.score(selectedx_test, y_test2)

    poly_test_accuracies.append(poly_test_acc)

#     print(current_g_c)

# print(best_g_c)

print(poly_test_accuracies)

print('the best degree is: ', degree[poly_test_accuracies.index(max(poly_test_accuracies))], 

      '\nthe highest testing accuracy is: ', max(poly_test_accuracies))
# #get testing accuracy for the best 

# poly_test_accuracies = list()

# for eachgc in range(len(best_g_c)):

#     current_g_c = best_g_c[eachgc]

#     poly = svc(C = current_g_c[1], gamma = current_g_c[0], degree = degree[eachgc], kernel = 'poly')

#     poly.fit(x_test2, y_test2)

#     poly_test_acc = poly.score(x_test2, y_test2)

#     poly_test_accuracies.append(poly_test_acc)

# print(poly_test_accuracies)

# print('the best degree is: ', degree[poly_test_accuracies.index(max(poly_test_accuracies))], 

#       '\nthe highest testing accuracy is: ', max(poly_test_accuracies))
rbf_acc = svc_cv(selected_cv_folds, c, 'rbf', gamma)
#plot accuracy

gamma_acc_list = list()

for eachacc in range(len(rbf_acc)):

    plt.scatter(c, rbf_acc[eachacc], label = gamma[eachacc])

    gamma_acc_list.append(max(rbf_acc[eachacc]))

max_acc = max(gamma_acc_list)

best_gamma = gamma[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_acc[gamma_acc_list.index(max_acc)]

best_c = c[best_gamma_acc.index(max_acc)]

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma')

plt.title('Accuracy for rbf kernel')

plt.show()

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#new c and gamma

c_rbf = np.arange(0.1, 2, 0.1)

gamma_rbf = np.arange(0.005, 0.015, 0.001)

rbf_acc = svc_cv(selected_cv_folds, c_rbf, 'rbf', gamma_rbf)
#plot accuracy

gamma_acc_list = list()

for eachacc in range(len(rbf_acc)):

    plt.scatter(c_rbf, rbf_acc[eachacc], label = gamma_rbf[eachacc])

    gamma_acc_list.append(max(rbf_acc[eachacc]))

max_acc = max(gamma_acc_list)

best_gamma = gamma_rbf[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_acc[gamma_acc_list.index(max_acc)]

best_c = c_rbf[best_gamma_acc.index(max_acc)]

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma', loc = (1, 0))

plt.title('Accuracy for rbf kernel')

plt.show()

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#testing rbf

rbf_test = svc(C = best_c, gamma = best_gamma, kernel = 'rbf')

rbf_test.fit(selectedx_test, y_test2)

rbf_test_acc = rbf_test.score(selectedx_test, y_test2)

print('the test accuracy for rbf kernel is: ', rbf_test_acc)
#split into train and test set

x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(df_pca, y, test_size = 0.2, random_state = 0)



#split into cv folds

x_folds_pca = np.array_split(x_train_pca, n_fold) #split training data into n_fold proportions

y_folds_pca = np.array_split(y_train_pca, n_fold) #split training data into n_fold proportions



cv_folds_pca = list() #stores dataframes for each fold

for eachfold in range(n_fold): #for each fold

    train_number = list(np.arange(0, n_fold)) #a list of fold numbers

    train_number.pop(eachfold) #pop current fold number

    df_y_ts = y_folds_pca[eachfold] #use current fold number as testing fold, create testing y

    df_x_ts = x_folds_pca[eachfold] #use current fold number as testing fold, create testing x

    

    y_train_list = list() #stores all the df y from folds for training

    x_train_list = list() #stores all the df x from folds for training

    for eachnumber in train_number: #for each training fold number

        x_train_list.append(x_folds_pca[eachnumber]) #append the df in training fold number for x

        y_train_list.append(y_folds_pca[eachnumber]) #append the df in training fold number for y

    df_x_tr = pd.concat(x_train_list) #combine all the training dfs for x into 1

    df_y_tr = pd.concat(y_train_list) #combine all the training dfs for y into 1

    cv_folds_pca.append([df_x_tr, df_y_tr, df_x_ts, df_y_ts]) #append training and testing dataframe for current fold
linear_pca = svc_cv(cv_folds_pca, c, 'linear')

plt.scatter(c, linear_pca)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')



linear_acc_pca = max(linear_pca)

best_c_pca = c[linear_pca.index(max(linear_pca))]

print('highest accuracy is: ', linear_acc_pca, '\nbest c is: ', best_c_pca)
#new c

c_pca = np.arange(0.005, 0.015, 0.001)

linear_pca = svc_cv(cv_folds_pca, c_pca, 'linear')

plt.scatter(c_pca, linear_pca)

plt.xlabel('c')

plt.ylabel('accuracy')

plt.title('Accuracy VS C for Linear Kernel')



linear_acc_pca = max(linear_pca)

best_c_pca = c_pca[linear_pca.index(max(linear_pca))]

print('highest accuracy is: ', linear_acc_pca, '\nbest c is: ', best_c_pca)
#test accuracy

linear_test_pca = svc(C = best_c_pca, kernel = 'linear')

linear_test_pca.fit(x_test_pca, y_test_pca)

linear_tsacc_pca = linear_test_pca.score(x_test_pca, y_test_pca)

print('the test accuracy for rbf kernel is: ', linear_tsacc_pca)
poly_pca = svc_cv(cv_folds_pca, c, 'poly', gamma, degree)
#plot accuracy

for eachdegree in range(len(poly_pca)):

    degree_acc = poly_pca[eachdegree]

    for eachacc in range(len(poly_pca[eachdegree])):

        plt.scatter(c, degree_acc[eachacc], label = gamma[eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma', loc = (1, 0))

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()

    

#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd in poly_pca: #for each degree

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma[each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c[best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
#new c and gamma

c_new1 = np.arange(95, 106)

gamma_new1 = np.arange(0.0005, 0.0015, 0.0001)

c_new2 = np.arange(1, 10)

gamma_new2 = np.arange(0.005, 0.015, 0.001)

c_new3 = np.arange(0.5, 1.5, 0.1)

gamma_new3 = np.arange(0.005, 0.015, 0.001)

c_new4 = np.arange(0.5, 1.5, 0.1)

gamma_new4 = np.arange(0.005, 0.015, 0.001)



c_new_pca = [c_new1, c_new2, c_new3, c_new4]

gamma_new_pca = [gamma_new1, gamma_new2, gamma_new3, gamma_new4]



#train with new hyperparameters

poly_pca = list()

for eachnew in range(len(c_new_pca)):

    pca_degree = int(eachnew+2)

    pca_degree = [pca_degree]

    new = svc_cv(cv_folds_pca, c_new_pca[eachnew], 'poly', gamma_new_pca[eachnew], pca_degree) 

    poly_pca.append(new)
#plot new acc

for eachdegree in range(len(poly_pca)):

    degree_acc = poly_pca[eachdegree][0]

    for eachacc in range(len(degree_acc)):

#         print(len(c_new_pca[eachdegree]), c_new_pca[eachdegree], np.arange(0.005, 0.025, 0.005))

#         print(len(degree_acc), len(degree_acc[eachacc]))

#         print(len(gamma_new_pca[eachdegree]))

        plt.scatter(c_new_pca[eachdegree], degree_acc[eachacc], label = gamma_new_pca[eachdegree][eachacc])

    plt.xlabel('c')

    plt.ylabel('accuracy')

    plt.legend(title = 'gamma', loc = (1, 0))

    plt.title('Accuracy for '+ str(degree[eachdegree]) + ' degree polynomial')

    plt.show()

    

#find the degree that has the highest accuracy

degree_max_acc = list() #store the max accuracy for each degree

best_g_c = list() #store the g and c values of the highest accuracy for each degree

for eachd_list in range(len(poly_pca)): #for each degree

    eachd = poly_pca[eachd_list][0]

    each_g_acc = list() #store the best accuracy for each degree

    for eachg in eachd: #for accuracy of each g value 

        each_g_acc.append(max(eachg)) #append the highest accuracy among c for current g and degree

    degree_max_acc.append(max(each_g_acc)) #append the highest accuracy for current degree

    best_g = gamma_new_pca[eachd_list][each_g_acc.index(max(each_g_acc))] #find which gamma has the highest accuracy

    best_g_list = eachd[each_g_acc.index(max(each_g_acc))] #get accuracies of each c of current gamma

    best_c = c_new_pca[eachd_list][best_g_list.index(max(each_g_acc))] #get which c has the highest accuracy

    best_g_c.append([best_g, best_c]) #append the best gamma and c values for this degree in a list

    

best_degree = degree[degree_max_acc.index(max(degree_max_acc))] #get the degree that has the best accuracy

print('max accuracy for each degree is: ', degree_max_acc)

print(best_degree)

print(best_g_c)
#get testing accuracy for the best 

poly_test_accuracies = list()

for eachgc in range(len(best_g_c)):

    current_g_c = best_g_c[eachgc]

    poly = svc(C = current_g_c[1], gamma = current_g_c[0], degree = degree[eachgc], kernel = 'poly')

    poly.fit(x_test_pca, y_test_pca)

    poly_test_acc = poly.score(x_test_pca, y_test_pca)

    poly_test_accuracies.append(poly_test_acc)

print(poly_test_accuracies)

print('the best degree is: ', degree[poly_test_accuracies.index(max(poly_test_accuracies))], 

      '\nthe highest testing accuracy is: ', max(poly_test_accuracies))
rbf_pca = svc_cv(cv_folds_pca, c, 'rbf', gamma)
#plot accuracy

gamma_acc_list = list()

for eachacc in range(len(rbf_pca)):

    plt.scatter(c, rbf_pca[eachacc], label = gamma[eachacc])

    gamma_acc_list.append(max(rbf_pca[eachacc]))

max_acc = max(gamma_acc_list)

best_gamma = gamma[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_pca[gamma_acc_list.index(max_acc)]

best_c = c[best_gamma_acc.index(max_acc)]

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma', loc = (1,0))

plt.title('Accuracy for rbf kernel')

plt.show()

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#new c and gamma

c_rbf = np.arange(80, 116)

gamma_rbf = np.arange(0.00005, 0.00015, 0.00001)

rbf_pca = svc_cv(cv_folds_pca, c_rbf, 'rbf', gamma_rbf)
#plot accuracy

gamma_acc_list = list()

for eachacc in range(len(rbf_pca)):

    plt.scatter(c_rbf, rbf_pca[eachacc], label = gamma_rbf[eachacc])

    gamma_acc_list.append(max(rbf_pca[eachacc]))

max_acc = max(gamma_acc_list)

best_gamma = gamma_rbf[gamma_acc_list.index(max_acc)]

best_gamma_acc = rbf_pca[gamma_acc_list.index(max_acc)]

best_c = c_rbf[best_gamma_acc.index(max_acc)]

plt.xlabel('c')

plt.ylabel('accuracy')

plt.legend(title = 'gamma', loc = (1, 0))

plt.title('Accuracy for rbf kernel')

plt.show()

print('the highest accuracy is: ', max_acc, '\nbest gamma is: ', best_gamma, '\nbest c is: ', best_c)
#testing rbf

rbf_test = svc(C = best_c, gamma = best_gamma, kernel = 'rbf')

rbf_test.fit(x_test_pca, y_test_pca)

rbf_test_acc = rbf_test.score(x_test_pca, y_test_pca)

print('the test accuracy for rbf kernel is: ', rbf_test_acc)