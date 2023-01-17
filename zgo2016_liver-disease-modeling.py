# Import Library



import os

import numpy as np 

import pandas as pd 

import seaborn as sns

import xgboost as xgb

import matplotlib.pyplot as plt



from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold, cross_val_score

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.metrics import precision_recall_fscore_support as score



import warnings 



warnings.filterwarnings("ignore")
# Load Indian liver patient data 

data=pd.read_csv('../input/indian_liver_patient.csv')
data.info()
data.head().T
data.describe().T
# Create new DataFrame that includes Male, Female patient information



disease, no_disease = data['Dataset'].value_counts()

male, female = data['Gender'].value_counts()



info=['Diognised with Liver Disease', 'Not Diognised with Liver Disease', 'Male', 'Female']

count=[disease, no_disease, male, female]



df_patient=pd.DataFrame({'Patient Info': info, 'Count': count})
df_patient
data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].median(), inplace=True)
target=data['Dataset']

sex=pd.get_dummies(data['Gender'])

#data = data.join(sex)

data.insert(loc=0, column='Male', value=sex['Male'])

data.insert(loc=0, column='Female', value=sex['Female'])

data.drop(['Gender'], axis=1, inplace=True)

#data.drop(['Dataset'], axis=1, inplace=True)
cols = data.columns

cols = list(set(cols))

del cols[cols.index('Dataset')]

#data.hist(column=cols, bins=10, figsize=(20,20), xlabelsize = 7, color='green', log=True)

del cols[cols.index('Male')]

del cols[cols.index('Female')]
def plot_data(cols, data, plot_type):



    fig = plt.figure(figsize = (25,25))

    

    sns.set(font_scale=1.5) 

    

    for idx, val in enumerate(cols):

            

        plt.subplot(3, 3, idx+1)



        if plot_type == 'hist':

            disease = 'sns.distplot(data[data["Dataset"] == 1].' + val + ', color="blue", label="Liver disease")'

            healthy = 'sns.distplot(data[data["Dataset"] == 2].' + val + ', color="orange", label="Healthy liver")'

            exec (disease)

            exec (healthy)

            plt.legend()

            plt.xlabel(val)

            plt.ylabel("Frequency")

          

        if plot_type == 'cdf':

            a='plt.hist(data[data["Dataset"] == 1].' + val + ',bins=50,fc=(0,1,0,0.5),label="Bening",normed = True,cumulative = True)'

            exec (a)

            sorted_data = exec('np.sort(data[data["Dataset"] == 1].' + val + ')')

            #sorted_data = exec (sorted_d)

            y = np.arange(len(sorted_data))/float(len(sorted_data)-1)

            plt.plot(sorted_data,y,color='red')

            plt.title('CDF of liver dicease bilirubin')

            

        if plot_type == 'swarm':

            condition = 'sns.swarmplot(x=' +  "'" + 'Dataset' + "'" + ',y=' + "'" + val + "'" + ',data=data)'

            print (condition)

            exec (condition)

              

        if plot_type == 'box':

            condition = 'sns.boxplot(x=' +  "'" + 'Dataset' + "'" + ',y=' + "'" + val + "'" + ',data=data)'

            print (condition)

            exec (condition)

            

        if plot_type == 'violin':

            condition = 'sns.violinplot(x=' +  "'" + 'Dataset' + "'" + ',y=' + "'" + val + "'" + ',data=data)'

            print (condition)

            exec (condition)

        

    return 0
plot_data(cols, data, 'hist')
plot_data(cols, data, 'swarm')
plot_data(cols, data, 'box')
# Define X and y for train/test split



X = data.drop(['Dataset'], axis=1)

y = data.Dataset



cols = data.columns

cols = list(set(cols))

del cols[cols.index('Dataset')]
def XGB(X_train, y_train, X_test, y_test):

   

    import xgboost as xgb



    xgb_clf = xgb.XGBClassifier()



    params={'max_depth': [2,3,4], 'subsample': [0.6, 1.0],'colsample_bytree': [0.5, 0.6],

    'n_estimators': [500, 1000], 'reg_alpha': [0.03, 0.05]}



    xgb = GridSearchCV(xgb_clf,param_grid=params, n_jobs=-1, cv=3, scoring='f1')

    xgb.fit(X_train, y_train)



    y_pred = xgb.predict(X_test)

    predictions = [round(value) for value in y_pred]



    # evaluate predictions

    print('Accuracy of xgb classifier on test set: {:.2f}'.format(accuracy_score(y_test, predictions)))

    print ("Classification report:\n{}".format(classification_report(y_test,predictions)))

    

    precision,recall,fscore,support=score(y_test,predictions)

    

    return fscore, accuracy_score(y_test, predictions)
def LR(X_train, y_train, X_test, y_test):

    

    clf = LogisticRegression()

    grid_values = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}

    

    lr = GridSearchCV(clf, param_grid=grid_values, cv=3, n_jobs=-1, scoring="f1")

    lr.fit(X_train, y_train)



    # make predictions on test data

    y_pred = lr.predict(X_test)



    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(X_test, y_test)))

    print ("Classification report:\n{}".format(classification_report(y_test,y_pred)))

    

    precision,recall,fscore,support=score(y_test, y_pred)

    

    return fscore, accuracy_score(y_test, y_pred)
def KNN(X_train, y_train, X_test, y_test):

    

    reg=KNeighborsClassifier(n_neighbors=8)



    #lr = GridSearchCV(clf, param_grid=grid_values, scoring="f1")



    k_range = list(range(1, 31))

    param_grid = dict(n_neighbors=k_range)



    grid = GridSearchCV(reg, param_grid, cv=3, n_jobs=-1, scoring='f1')

    grid.fit(X_train, y_train)

       

    print('Accuracy of KNeighbors classifier on test set: {:.2f}'.format(grid.score(X_test, y_test)))

    print ("Classification report:\n{}".format(classification_report(y_test,grid.predict(X_test))))

    

    precision,recall,fscore,support=score(y_test, grid.predict(X_test))

    return fscore, accuracy_score(y_test, grid.predict(X_test))
def RF(X_train, y_train, X_test, y_test, flag=0):

    

    from sklearn.metrics import precision_recall_fscore_support as score

    from sklearn.metrics import confusion_matrix

    from sklearn import metrics



    rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)

    parameters = {'n_estimators':[200, 400, 600, 800, 1000], 'min_samples_leaf':[4, 8, 16], 

                  'max_features': ['auto', 'sqrt']}



    scoring = make_scorer(accuracy_score, greater_is_better=True)



    cl_rand_fr = GridSearchCV(rfc, param_grid=parameters, cv=3, n_jobs=-1, scoring='f1')

    cl_rand_fr.fit(X_train, y_train)

    cl_rand_fr = cl_rand_fr.best_estimator_



    # Show prediction accuracy score

    print ('Accuracy of random forest classifier on test set: {:.2f}'.format(accuracy_score(y_test, cl_rand_fr.predict(X_test))))

    print ("Classification report:\n{}".format(classification_report(y_test,cl_rand_fr.predict(X_test))))

    

    if flag == 1:

    

        print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, cl_rand_fr.predict(X_test))))

    

        from yellowbrick.classifier import ROCAUC

        fig, ax=plt.subplots(1,1,figsize=(12,8))



        auc=ROCAUC(cl_rand_fr, macro=False, micro=False)

        auc.fit(X_train, y_train)

        auc.score(X_test, y_test)

        auc.poof()

        

        return 0

    

    else:

        

        precision,recall,fscore,support=score(y_test, cl_rand_fr.predict(X_test))

        return fscore, accuracy_score(y_test, cl_rand_fr.predict(X_test)), cl_rand_fr.feature_importances_
model = ["Xgboost", "KN Neighbors", "Logistic Regression", "Random Forest"]

fsc1=[]

fsc2=[]

acc=[]

f1acc=[]

result = ["Regular", "MinMaxScaled", "Quantile", "SMOTE", "Max"]
X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size = 0.3, random_state=42, stratify=y)



fscore, accuracy = XGB(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = KNN(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = LR(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy, features = RF(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



coef = pd.Series(features, index = X_train.columns).sort_values(ascending=False)



plt.figure(figsize=(10, 10))

coef.head(11).plot(kind='bar')

plt.title('Feature Significance')
scaler=MinMaxScaler()



X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])



fscore, accuracy = XGB(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = KNN(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = LR(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy, features = RF(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)
# Introduce quantile features and add them to dataset

scaler=MinMaxScaler()



qcut_cols=['Alkaline_Phosphotase', 'Direct_Bilirubin', 'Alamine_Aminotransferase', 'Total_Bilirubin', 'Aspartate_Aminotransferase', 'Age', 'Albumin_and_Globulin_Ratio']

important_fs=data[qcut_cols]



for i in range(len(qcut_cols)):

    new_q_f1=pd.get_dummies(pd.qcut(data[qcut_cols[i]], 4, labels=[qcut_cols[i]+"_Q0", qcut_cols[i]+"_Q1", qcut_cols[i]+"_Q2", qcut_cols[i]+"_Q3"]))

    important_fs=pd.concat([important_fs, new_q_f1], axis=1, sort=False)



new_cols=important_fs.columns.values



X_train, X_test, y_train, y_test = train_test_split(important_fs[new_cols], y, test_size = 0.3, random_state=0, stratify=y)



X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])



fscore, accuracy = XGB(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = KNN(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = LR(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy, features = RF(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)
# SMOTE oversampling 

from sklearn.metrics import f1_score

from sklearn.metrics import make_scorer



f1_scorer = make_scorer(f1_score)



X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size = 0.3, random_state=0, stratify=y)



old_X_test = X_test

old_y_test = y_test



parameters = {'n_estimators':[200, 400, 600, 800, 1000], 'min_samples_leaf':[4, 8, 16], 'max_features': ['auto', 'sqrt']}

rfc = RandomForestClassifier(random_state=42, min_samples_split=5, oob_score=True)



sm = SMOTE(random_state=0, sampling_strategy=1.0, k_neighbors=7, n_jobs=-1)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

X_test_res, y_test_res = sm.fit_sample(X_test, y_test.ravel())



fscore, accuracy = XGB(X_train_res, y_train_res, X_test.values, y_test.values)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = KNN(X_train_res, y_train_res, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = LR(X_train_res, y_train_res, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy, features = RF(X_train_res, y_train_res, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)
# Setting sample value to 1 for the new feature, if healthy patient’s value from the original 

# feature is above the whisker extend of the corresponding un-healthy patient’s value.

# Otherwise, we will set it to 0.



important_max=data[qcut_cols]

val = data[qcut_cols][data['Dataset'] == 2].max()

for i in range(len(qcut_cols)):

    new_max_f1=qcut_cols[i]

    important_max[new_max_f1+'_max']=np.where(important_max[new_max_f1]>val[new_max_f1], 1, 0)



new_cols_max=important_max.columns.values



X_train, X_test, y_train, y_test = train_test_split(important_max[new_cols_max], y, test_size = 0.3, random_state=0, stratify=y)



fscore, accuracy = XGB(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = KNN(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy = LR(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)



fscore, accuracy, features = RF(X_train, y_train, X_test, y_test)



fsc1.append(fscore[0])

fsc2.append(fscore[1])

acc.append(accuracy)


table=pd.DataFrame({'Applied Method':result, 'f1 score for unhealthy patients from Xgboost':[fsc1[0], fsc1[4], fsc1[8], fsc1[12], fsc1[16]], 'f1 score for healthy patients from Xgboost':[fsc2[0], fsc2[4], fsc2[8], fsc2[12], fsc2[16]], 'f1 accuracy score for Xgboost':[acc[0], acc[4], acc[8], acc[12], acc[16]], 'f1 score for unhealthy patients from KNN':[fsc1[1], fsc1[5], fsc1[9], fsc1[13], fsc1[17]], 'f1 score for healthy patients from KNN':[fsc2[1], fsc2[5], fsc2[9], fsc2[13], fsc2[17]], 'f1 accuracy score for KNN':[acc[1], acc[5], acc[9], acc[13], acc[17]], 'f1 score for unhealthy patients from Logistic Regression':[fsc1[2], fsc1[6], fsc1[10], fsc1[14], fsc1[18]], 'f1 score for healthy patients from Logistic Regression':[fsc2[2], fsc2[6], fsc2[10], fsc2[14], fsc2[18]], 'f1 accuracy score for Logistic Regression':[acc[2], acc[6], acc[10], acc[14], acc[18]], 'f1 score for unhealthy patients from Random Forest':[fsc1[3], fsc1[7], fsc1[11], fsc1[15], fsc1[19]], 'f1 score for healthy patients from Random Forest':[fsc2[3], fsc2[7], fsc2[11], fsc2[15], fsc2[19]], 'f1 accuracy score for Random Forest':[acc[3], acc[7], acc[11], acc[15], acc[19]]})

table=table.set_index('Applied Method').T

table
print ("The best f1 accuracy score for Random Forest is with SMOTE oversampled train/test split:", table['SMOTE'].loc["f1 accuracy score for Random Forest"])

print ("with f1 score for unhealthy patients:", table['SMOTE'].loc["f1 score for unhealthy patients from Random Forest"], "and f1 score for healthy patients:", table['SMOTE'].loc["f1 score for healthy patients from Random Forest"])
RF(X_train_res, y_train_res, old_X_test, old_y_test, 1)
num_cols = data._get_numeric_data().columns

cor = data[num_cols].corr()



threshold = 0.7



corlist = []



for i in range(0,len(num_cols)):

    for j in range(i+1,len(num_cols)):

        if (j != i and cor.iloc[i,j] <= 1 ) or (j != i and cor.iloc[i,j] >= -1):

            corlist.append([cor.iloc[i,j],i,j]) 



#Sort higher correlations first            

sort_corlist = sorted(corlist,key=lambda x: -abs(x[0]))
fig, ax = plt.subplots(figsize=(17,17))



corr_mat=data.corr()

sns.heatmap(corr_mat,annot=True,linewidths=1, ax=ax)
x_plot=[]

y_plot=[]

for x,i,j in sort_corlist:

    if num_cols[i] != 'Dataset' and num_cols[j] != 'Dataset':

#        print (num_cols[i],num_cols[j],x)

        x_plot.append(num_cols[i])

        y_plot.append(num_cols[j])
# Strongly correlated pairs

x_max_plot=[]

y_max_plot=[]

for x,i,j in sort_corlist:

    if num_cols[i] != 'Dataset' and num_cols[j] != 'Dataset':

        if x >= 0.60:

            x_max_plot.append(num_cols[i])

            y_max_plot.append(num_cols[j])

            print (num_cols[i],num_cols[j],x)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

axes=axes.flatten()

for i in range(len(x_max_plot)):

    sns.scatterplot(data=data, x=x_max_plot[i], y=y_max_plot[i], ax=axes[i])