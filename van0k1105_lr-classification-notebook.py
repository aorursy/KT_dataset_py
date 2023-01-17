import pandas as pd



from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.cross_validation import KFold

from sklearn.metrics import log_loss,r2_score

from sklearn.preprocessing import normalize



import numpy as np

np.set_printoptions(precision=4,suppress=True,linewidth=100)



from IPython.display import display



from scipy import stats



import math



from matplotlib import pyplot as plt

%matplotlib inline
#reading data

df = pd.read_csv('../input/data.csv')

#changing symbolic class tags into numeric ones (M = 1, B = 0)

#feature selection and transformation - will be expanded in the future

df['diagnosis'] = df['diagnosis'].apply(lambda x:1 if x == 'M' else 0)

df['concavity_mean'] = df['concavity_mean'].apply(lambda x: np.mean(df['concavity_mean']) if x<=0 else x)

df['log_concavity'] = df['concavity_mean'].apply(lambda x: math.log(x))

df['log_fd'] = df['fractal_dimension_mean'].apply(lambda x: math.log(x))

df = df[['diagnosis','radius_mean','texture_mean','log_concavity','smoothness_mean','symmetry_mean','log_fd']]

display(df)
#assigning independent and dependent variables

X = df[['radius_mean','texture_mean','log_concavity','smoothness_mean','symmetry_mean','log_fd']]

y = df['diagnosis']
#calculating coefficients of determination for linear regression of each predictor from other ones

r_squared = []

lin_regr = LinearRegression()

for column in X.columns:

    regr_var = X.ix[:, X.columns != column]

    regr_target = X[column]

    lin_regr.fit(regr_var,regr_target)

    r_squared.append(r2_score(regr_target,lin_regr.predict(regr_var)))

#calculating VIFs

vif = [1/(1-r2) for r2 in r_squared]

#output

r2_data = pd.DataFrame(index = X.columns)

r2_data['R^2'] = r_squared

r2_data['VIF'] = vif

display(r2_data)
#normalizing the matrix of independent variables before applying BKW

X_bkw = normalize(X,norm='l2',axis=0)

#calculating SVD of the matrix

U, s, V = np.linalg.svd(X_bkw)

#calculating condition indexes

cond_indexes = np.max(s)/s

#calculating variance-decomposition proportions

var_frac = np.matrix([[V[k,j]**2/s[j]**2 for k in range(len(s))] for j in range(len(s))])

var_frac = [var_frac[:,j]/np.sum(var_frac[:,j]) for j in range(np.shape(var_frac)[1])]

var_frac = np.transpose(np.reshape(a=var_frac,newshape=(6,6)))

#output

df_bkw = pd.DataFrame()

df_bkw['Condition Index'] = cond_indexes

df_bkw['radius_mean'] = var_frac[:,0]

df_bkw['texture_mean'] = var_frac[:,1]

df_bkw['log_concavity'] = var_frac[:,2]

df_bkw['smoothness_mean'] = var_frac[:,3]

df_bkw['symmetry_mean'] = var_frac[:,4]

df_bkw['log_fd'] = var_frac[:,5]

display(df_bkw)
#С is inverse regularization coefficient, we take a large value to suppress regularization for now

log_regr = LogisticRegression(C=100000)

#Fitting the model

log_regr.fit(X,y)

#Displaying coefficients

print('Coefficients:')

print('Constant: {} X1: {} X2: {} X3: {} X4: {} X5: {} X6: {}'.format(log_regr.intercept_[0],log_regr.coef_[0,0],log_regr.coef_[0,1],log_regr.coef_[0,2],log_regr.coef_[0,3],log_regr.coef_[0,4],log_regr.coef_[0,5]))

coefs = [log_regr.intercept_[0],log_regr.coef_[0,0],log_regr.coef_[0,1],log_regr.coef_[0,2],log_regr.coef_[0,3],log_regr.coef_[0,4],log_regr.coef_[0,5]]
#getting the probability scores

probs = log_regr.predict_proba(X)[:,1]

#calculating the covariance matrix

X_const = pd.DataFrame()

X_const['Constant_term'] = [1]*len(X)

X_const = pd.concat([X_const,X],axis=1)

V = np.diag([x*(1-x) for x in probs])

covariance_matrix = np.linalg.inv(np.transpose(X_const).dot(V).dot(X_const))

covar_dataframe = pd.DataFrame(data=covariance_matrix,index=X_const.columns,columns=X_const.columns)

print('Covariance matrix:')

display(covar_dataframe)
#getting the variances

variances = np.diagonal(covariance_matrix)

print('Variances:')

print('Constant: {} X1: {} X2: {} X3: {} X4: {} X5: {} X6: {}'.format(*variances))
#calculating Wald statistics

wald_stats = [coefs[i]**2 / variances[i] for i in range(len(coefs))]

print('Wald statistics:')

print('Constant: {} X1: {} X2: {} X3: {} X4: {} X5: {} X6: {}'.format(*wald_stats))

#calculating p-values

p_values = [1 - stats.chi2.cdf(stat,1) for stat in wald_stats]

print('P-value:')

print('Constant: {} X1: {} X2: {} X3: {} X4: {} X5: {} X6: {}'.format(*p_values))
#building the output table

base_model_df = pd.DataFrame(index=X_const.columns)

base_model_df['Coefficient'] = coefs

base_model_df['Variance'] = variances

base_model_df['Wald-stat'] = wald_stats

base_model_df['P-value'] = p_values

pd.set_option('display.float_format', lambda x: '%.5f' % x)

display(base_model_df)
#calculating log-likelihood for the basic model and the "null" model

L_1 = -log_loss(y,probs)

L_0 = -log_loss(y,[0]*len(y))

print('L1: {} L0: {}'.format(L_1,L_0))

#calculating McFadden's R^2

r2_macfadden = 1 - (L_1/L_0)

print('R^2 McFadden: {}'.format(r2_macfadden))
#getting the values for the "ground truth/predicted" table

#cross-validating over 10 folds

total_TP = 0

total_FP = 0

total_TN = 0

total_FN = 0

total_accuracy = []

total_precision = []

total_recall = []

total_f_score = []

kf = KFold(len(X),n_folds=10)

for train_index,test_index in kf:

    x_train, x_test = X.iloc[train_index],X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    log_regr.fit(x_train,y_train)

    pred = log_regr.predict(x_test)

    TP, FP, TN, FN = 0, 0, 0, 0

    y_test = y_test.values

    for i in range(len(pred)):

        if pred[i] == 1 and y_test[i] == 1:

            TP += 1

            total_TP += 1

        if pred[i] == 1 and y_test[i] == 0:

            FP += 1

            total_FP += 1

        if pred[i] == 0 and y_test[i] == 0:

            TN += 1

            total_TN += 1

        if pred[i] == 0 and y_test[i] == 1:

            FN += 1

            total_FN += 1

    #calculating metrics for the iteration

    accuracy = float(TP + TN)/len(pred)

    total_accuracy.append(accuracy)

    precision = float(TP)/(TP + FP)

    total_precision.append(precision)

    recall = float(TP)/(TP + FN)

    total_recall.append(recall)

    f_score = 2*precision*recall/(precision+recall)

    total_f_score.append(f_score)

#displayig the "true/predicted" table

true_pred_table = pd.DataFrame(index=['M','B','Total'],columns=['M','B','Total'])

true_pred_table.columns.name = 'True\Pred'

true_pred_table['M']['M'] = total_TP

true_pred_table['M']['B'] = total_FP

true_pred_table['B']['M'] = total_FN

true_pred_table['B']['B'] = total_TN

true_pred_table['M']['Total'] = total_TP + total_FP

true_pred_table['B']['Total'] = total_FN + total_TN

true_pred_table['Total']['M'] = total_TP + total_FN

true_pred_table['Total']['B'] = total_FP + total_TN

true_pred_table['Total']['Total'] = total_TP + total_FP + total_FN + total_TN

display(true_pred_table)

#displaying metrics

metric_df = pd.DataFrame(index = ['Accuracy','Precision','Recall','F Score'],columns=['Value'])

metric_df.columns.name = 'Quality metric'

metric_df['Value'] = [np.mean(total_accuracy),np.mean(total_precision),np.mean(total_recall),np.mean(total_f_score)]

display(metric_df)
#building the variable matrix with new test variables

X_reset = X.copy()

X_reset['probs_squared'] = probs**2

X_reset['probs_cubed'] = probs**3

#Fitting the model and calculating variances

log_regr.fit(X_reset,y)

coefs_reset = [log_regr.coef_[0,6],log_regr.coef_[0,7]]

probs_reset = log_regr.predict_proba(X_reset)[:,1]

X_const_reset = pd.DataFrame()

X_const_reset['Constant_term'] = [1]*len(X)

X_const_reset = pd.concat([X_const_reset,X_reset],axis=1)

V_reset = np.diag([x*(1-x) for x in probs_reset])

covariance_matrix_reset = np.linalg.inv(np.transpose(X_const_reset).dot(V_reset).dot(X_const_reset))

covar_dataframe_reset = pd.DataFrame(data=covariance_matrix_reset,index=X_const_reset.columns,columns=X_const_reset.columns)

print('Covariance matrix:')

display(covar_dataframe_reset)
#calculating Wald statistics

variances_reset = np.diagonal(a=covariance_matrix_reset)[7:]

wald_stats_reset = [coefs_reset[i]**2/variances_reset[i] for i in range(len(coefs_reset))]

p_values_reset = [1 - stats.chi2.cdf(stat,1) for stat in wald_stats_reset]

#the final output

reset_model_df = pd.DataFrame(index=['probs_squared','probs_cubed'])

reset_model_df['Coefficient'] = coefs_reset

reset_model_df['Variance'] = variances_reset

reset_model_df['Wald-stat'] = wald_stats_reset

reset_model_df['P-value'] = p_values_reset

display(reset_model_df)
reg_strength = [1.1**i for i in range(-20,20)]

crossval_results = pd.DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F Score'], index = reg_strength)

crossval_results.columns.name = 'Inverse reg. strength'

#iterating over coefficient's values

for c in reg_strength:

    log_regr = LogisticRegression(C=c)

    total_accuracy = []

    total_precision = []

    total_recall = []

    total_f_score = []

    #cross-validating over 10 folds

    kf = KFold(len(df),n_folds=10)

    for train_index,test_index in kf:

        x_train, x_test = X.loc[train_index],X.loc[test_index]

        y_train, y_test = y.loc[train_index], y.loc[test_index]

        log_regr.fit(x_train,y_train)

        pred = [1 if x>0.5 else 0 for x in log_regr.predict_proba(x_test)[:,1]]

        TP, FP, TN, FN = 0, 0, 0, 0

        y_test = y_test.values

        for i in range(len(pred)):

            if pred[i] == 1 and y_test[i] == 1:

                TP += 1

            if pred[i] == 1 and y_test[i] == 0:

                FP += 1

            if pred[i] == 0 and y_test[i] == 0:

                TN += 1

            if pred[i] == 0 and y_test[i] == 1:

                FN += 1

        #calculating metrics for the iteration

        accuracy = float(TP + TN)/len(pred)

        total_accuracy.append(accuracy)

        precision = float(TP)/(TP + FP)

        total_precision.append(precision)

        recall = float(TP)/(TP + FN)

        total_recall.append(recall)

        f_score = 2*precision*recall/(precision+recall)

        total_f_score.append(f_score)

    crossval_results['Accuracy'][c]=np.mean(total_accuracy)

    crossval_results['Precision'][c]=np.mean(total_precision)

    crossval_results['Recall'][c]=np.mean(total_recall)

    crossval_results['F Score'][c]=np.mean(total_f_score)

display(crossval_results)
#displaying the best iteration

best_result = crossval_results.loc[crossval_results['Recall'].idxmax()]

print(best_result)
#fitting the regularized model

log_regr = LogisticRegression(C=best_result.name)

log_regr.fit(X,y)

coefs = [log_regr.intercept_[0],log_regr.coef_[0,0],log_regr.coef_[0,1],log_regr.coef_[0,2],log_regr.coef_[0,3],log_regr.coef_[0,4],log_regr.coef_[0,5]]

probs_reg = log_regr.predict_proba(X)[:,1]

X_const = pd.DataFrame()

X_const['Constant_term'] = [1]*len(X)

X_const = pd.concat([X_const,X],axis=1)

V = np.diag([x*(1-x) for x in probs_reg])

covariance_matrix = np.linalg.inv(np.transpose(X_const).dot(V).dot(X_const))

variances = np.diagonal(covariance_matrix)

wald_stats = [coefs[i]**2 / variances[i] for i in range(len(coefs))]

p_values = [1 - stats.chi2.cdf(stat,1) for stat in wald_stats]



#displaying the regularized model's coefficients

reg_model_df = pd.DataFrame(index=X_const.columns)

reg_model_df['Coefficient'] = coefs

reg_model_df['Variance'] = variances

reg_model_df['Wald-stat'] = wald_stats

reg_model_df['P-value'] = p_values

display(reg_model_df)
#building the new variable matrix without insignificant variables

X = X[['radius_mean','texture_mean','log_concavity']]

#building the new model

log_regr.fit(X,y)

coefs = [log_regr.intercept_[0],log_regr.coef_[0,0],log_regr.coef_[0,1],log_regr.coef_[0,2]]

probs = log_regr.predict_proba(X)[:,1]

X_const = pd.DataFrame()

X_const['Constant_term'] = [1]*len(X)

X_const = pd.concat([X_const,X],axis=1)

V = np.diag([x*(1-x) for x in probs])

covariance_matrix = np.linalg.inv(np.transpose(X_const).dot(V).dot(X_const))

variances = np.diagonal(covariance_matrix)

wald_stats = [coefs[i]**2 / variances[i] for i in range(len(coefs))]

p_values = [1 - stats.chi2.cdf(stat,1) for stat in wald_stats]



#displaying the coefficients

reg_model_df = pd.DataFrame(index=X_const.columns)

reg_model_df['Coefficient'] = coefs

reg_model_df['Variance'] = variances

reg_model_df['Wald-stat'] = wald_stats

reg_model_df['P-value'] = p_values

display(reg_model_df)
#displaying the new model's metrics

#getting the values for the "ground truth/predicted" table

#cross-validating over 10 folds

total_TP = 0

total_FP = 0

total_TN = 0

total_FN = 0

total_accuracy = []

total_precision = []

total_recall = []

total_f_score = []

kf = KFold(len(X),n_folds=10)

for train_index,test_index in kf:

    x_train, x_test = X.iloc[train_index],X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    log_regr.fit(x_train,y_train)

    pred = log_regr.predict(x_test)

    TP, FP, TN, FN = 0, 0, 0, 0

    y_test = y_test.values

    for i in range(len(pred)):

        if pred[i] == 1 and y_test[i] == 1:

            TP += 1

            total_TP += 1

        if pred[i] == 1 and y_test[i] == 0:

            FP += 1

            total_FP += 1

        if pred[i] == 0 and y_test[i] == 0:

            TN += 1

            total_TN += 1

        if pred[i] == 0 and y_test[i] == 1:

            FN += 1

            total_FN += 1

    #calculating metrics for the iteration

    accuracy = float(TP + TN)/len(pred)

    total_accuracy.append(accuracy)

    precision = float(TP)/(TP + FP)

    total_precision.append(precision)

    recall = float(TP)/(TP + FN)

    total_recall.append(recall)

    f_score = 2*precision*recall/(precision+recall)

    total_f_score.append(f_score)

#displayig the "true/predicted" table

true_pred_table = pd.DataFrame(index=['M','B','Total'],columns=['M','B','Total'])

true_pred_table.columns.name = 'True\Pred'

true_pred_table['M']['M'] = total_TP

true_pred_table['M']['B'] = total_FP

true_pred_table['B']['M'] = total_FN

true_pred_table['B']['B'] = total_TN

true_pred_table['M']['Total'] = total_TP + total_FP

true_pred_table['B']['Total'] = total_FN + total_TN

true_pred_table['Total']['M'] = total_TP + total_FN

true_pred_table['Total']['B'] = total_FP + total_TN

true_pred_table['Total']['Total'] = total_TP + total_FP + total_FN + total_TN

display(true_pred_table)

#displaying metrics

metric_df = pd.DataFrame(index = ['Accuracy','Precision','Recall','F Score'],columns=['Value'])

metric_df.columns.name = 'Quality metric'

metric_df['Value'] = [np.mean(total_accuracy),np.mean(total_precision),np.mean(total_recall),np.mean(total_f_score)]

display(metric_df)
#cross-validation for an optimal classification threshold

classification_thresholds = np.arange(0.05,0.5,0.05)

crossval_results = pd.DataFrame(columns = ['Accuracy', 'Precision', 'Recall', 'F Score'], index = classification_thresholds)

crossval_results.columns.name = 'Threshold'

#iterating over threshold values

log_regr = LogisticRegression(C=best_result.name)

for threshold in classification_thresholds:

    total_accuracy = []

    total_precision = []

    total_recall = []

    total_f_score = []

    #cross-validating over 10 folds

    kf = KFold(len(df),n_folds=10)

    for train_index,test_index in kf:

        x_train, x_test = X.loc[train_index],X.loc[test_index]

        y_train, y_test = y.loc[train_index], y.loc[test_index]

        log_regr.fit(x_train,y_train)

        pred = [1 if x>threshold else 0 for x in log_regr.predict_proba(x_test)[:,1]]

        TP, FP, TN, FN = 0, 0, 0, 0

        y_test = y_test.values

        for i in range(len(pred)):

            if pred[i] == 1 and y_test[i] == 1:

                TP += 1

            if pred[i] == 1 and y_test[i] == 0:

                FP += 1

            if pred[i] == 0 and y_test[i] == 0:

                TN += 1

            if pred[i] == 0 and y_test[i] == 1:

                FN += 1

        #calculating metrics for the iteration

        accuracy = float(TP + TN)/len(pred)

        total_accuracy.append(accuracy)

        precision = float(TP)/(TP + FP)

        total_precision.append(precision)

        recall = float(TP)/(TP + FN)

        total_recall.append(recall)

        f_score = 2*precision*recall/(precision+recall)

        total_f_score.append(f_score)

    crossval_results['Accuracy'][threshold]=np.mean(total_accuracy)

    crossval_results['Precision'][threshold]=np.mean(total_precision)

    crossval_results['Recall'][threshold]=np.mean(total_recall)

    crossval_results['F Score'][threshold]=np.mean(total_f_score)

display(crossval_results)
#calculating log-likelihood

probs_optimized = log_regr.predict_proba(X)[:,1]

L_1 = -log_loss(y,probs_optimized)

L_0 = -log_loss(y,[0]*len(y))

print('L1: {} L0: {}'.format(L_1,L_0))

#calculating McFadden's R2

r2_macfadden = 1 - (L_1/L_0)

print('R^2 McFadden: {}'.format(r2_macfadden))
#calculating log-likelihood

L_basic = -log_loss(y,probs)

L_optimized = -log_loss(y,probs_optimized)

#Calculating AICs

aic_basic = 16 - 2*L_basic

aic_optimized = 10 - 2*L_optimized

print('Basic: {} Optimized: {}'.format(aic_basic,aic_optimized))

information_loss_proba = math.exp((aic_optimized - aic_basic)/2)

print('Probability that basic model minimizes information loss: {}'.format(information_loss_proba))
bic_basic = 8*math.log(len(df)) - 2*L_basic

bic_optimized = 5*math.log(len(df)) - 2*L_optimized

print('Basic: {} Optimized: {}'.format(bic_basic,bic_optimized))
#calculating confidence intervals for odds ratio

CI_df = reg_model_df[['Coefficient','Variance']].loc[['radius_mean','texture_mean','log_concavity']]

CI_df['Odds Ratio'] = CI_df.apply(lambda x: math.exp(x['Coefficient']),axis=1)

CI_df['Lower CL'] = CI_df.apply(lambda x: x['Odds Ratio']+stats.norm.interval(0.95)[0]*math.sqrt(x['Variance'])*x['Odds Ratio'],axis=1)

CI_df['Upper CL'] = CI_df.apply(lambda x: x['Odds Ratio']+stats.norm.interval(0.95)[1]*math.sqrt(x['Variance'])*x['Odds Ratio'],axis=1)

display(CI_df)
residuals_df = pd.DataFrame()

#calculating absolute deviation

residuals_df['Y'] = df['diagnosis']

residuals_df['Probability'] = probs_optimized

residuals_df['Error'] = residuals_df['Y'] - residuals_df['Probability']

#calculating Pearson-normalized residuals

residuals_df['Pearson residual'] = residuals_df['Error']/(residuals_df['Probability']*(1 - residuals_df['Probability']))

#calculating studentized Pearson-normalized residuals

#calculating the projection matrix to get Pregibon leverages

W = np.diag([prob*(1-prob) for prob in probs_optimized])

H = (W**(1/2)).dot(X).dot(np.linalg.matrix_power(np.transpose(X).dot(W).dot(X),-1)).dot(np.transpose(X)).dot(W**(1/2))

h = np.diagonal(H)

residuals_df['SPR'] = [residuals_df['Pearson residual'].iloc[i]/math.sqrt(1-h[i]) for i in range(len(df))]

#calculating deviance residual

residuals_df['Deviance residual'] = [np.sign(residuals_df['Y'].iloc[i] - residuals_df['Probability'].iloc[i])*math.sqrt(-2*(residuals_df['Y'].iloc[i]*math.log(residuals_df['Probability'].iloc[i])+(1-residuals_df['Y'].iloc[i])*math.log(1-residuals_df['Probability'].iloc[i]))) for i in range(len(df))]

#calculating Delta Chi2

residuals_df['Delta Chi2'] = residuals_df['SPR']**2

#calculating deltas of deviation

residuals_df['Delta D'] = [(residuals_df['Deviance residual'].iloc[i])**2/(1-h[i]) for i in range(len(df))]

#calculating deltas of regression coefficients

residuals_df['Delta Coefficients'] = [h[i]*(residuals_df['SPR'].iloc[i])**2/(1-h[i]) for i in range(len(df))]

#displaying residual plots

f,ax = plt.subplots(2,3,figsize=(15,10))

plt.tight_layout()

ax[0,0].scatter(x=residuals_df['Probability'],y=residuals_df['Pearson residual'])

ax[0,0].set_title('PR')

ax[0,1].scatter(x=residuals_df['Probability'],y=residuals_df['SPR'])

ax[0,1].set_title('SPR')

ax[0,2].scatter(x=residuals_df['Probability'],y=residuals_df['Deviance residual'])

ax[0,2].set_title('DR')

ax[1,0].scatter(x=residuals_df['Probability'],y=residuals_df['Delta Chi2'])

ax[1,0].set_title('dChi2')

ax[1,1].scatter(x=residuals_df['Probability'],y=residuals_df['Delta D'])

ax[1,1].set_title('dD')

ax[1,2].scatter(x=residuals_df['Probability'],y=residuals_df['Delta Coefficients'])

ax[1,2].set_title('dCoef')

plt.show()

#displaying the table

display(residuals_df)
#finding objects where Delta Chi2 and Delta D lie outside the 95-th percentile of Chi2 distribution with 1 df

#and Delta Coefficients is more than 0.1

chisq_thresh = stats.chi2.ppf(0.95,1)

outliers1 = residuals_df.loc[(residuals_df['Delta Chi2'] > chisq_thresh) | (residuals_df['Delta D'] > chisq_thresh)]

display(outliers1)

print('Number of outliers by the Chi2 Test: {}'.format(len(outliers1)))

outliers2 = outliers1[outliers1['Delta Coefficients'] > 0.1]

display(outliers2)

print('Number of outliers with big coefficient influence: {}'.format(len(outliers2)))
#displaying the outliers

display(df[df.index.isin(outliers2.index.values)])
X = X[X.index.isin(outliers2.index.values) == False]

y = y[y.index.isin(outliers2.index.values) == False]

log_regr.fit(X,y)

#cross-validating over 10 folds

total_TP = 0

total_FP = 0

total_TN = 0

total_FN = 0

total_accuracy = []

total_precision = []

total_recall = []

total_f_score = []

kf = KFold(len(X),n_folds=10)

for train_index,test_index in kf:

    x_train, x_test = X.iloc[train_index],X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    log_regr.fit(x_train,y_train)

    pred = [1 if x>0.35 else 0 for x in log_regr.predict_proba(x_test)[:,1]]

    TP, FP, TN, FN = 0, 0, 0, 0

    y_test = y_test.values

    for i in range(len(pred)):

        if pred[i] == 1 and y_test[i] == 1:

            TP += 1

            total_TP += 1

        if pred[i] == 1 and y_test[i] == 0:

            FP += 1

            total_FP += 1

        if pred[i] == 0 and y_test[i] == 0:

            TN += 1

            total_TN += 1

        if pred[i] == 0 and y_test[i] == 1:

            FN += 1

            total_FN += 1

    #calculating metrics for the iteration

    accuracy = float(TP + TN)/len(pred)

    total_accuracy.append(accuracy)

    precision = float(TP)/(TP + FP)

    total_precision.append(precision)

    recall = float(TP)/(TP + FN)

    total_recall.append(recall)

    f_score = 2*precision*recall/(precision+recall)

    total_f_score.append(f_score)

#displayig the "true/predicted" table

true_pred_table = pd.DataFrame(index=['M','B','Total'],columns=['M','B','Total'])

true_pred_table.columns.name = 'True\Pred'

true_pred_table['M']['M'] = total_TP

true_pred_table['M']['B'] = total_FP

true_pred_table['B']['M'] = total_FN

true_pred_table['B']['B'] = total_TN

true_pred_table['M']['Total'] = total_TP + total_FP

true_pred_table['B']['Total'] = total_FN + total_TN

true_pred_table['Total']['M'] = total_TP + total_FN

true_pred_table['Total']['B'] = total_FP + total_TN

true_pred_table['Total']['Total'] = total_TP + total_FP + total_FN + total_TN

display(true_pred_table)

#displaying metrics

metric_df = pd.DataFrame(index = ['Accuracy','Precision','Recall','F Score'],columns=['Value'])

metric_df.columns.name = 'Quality metric'

metric_df['Value'] = [np.mean(total_accuracy),np.mean(total_precision),np.mean(total_recall),np.mean(total_f_score)]

display(metric_df)