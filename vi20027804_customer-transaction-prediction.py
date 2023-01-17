# system

import os

import warnings



# data manipulation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import gc



# setting warning to ignored

warnings.filterwarnings("ignore")

print(os.listdir("../input"))



# importing classifiers

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



# preprocessing/ cross-validation

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

from sklearn.model_selection import GridSearchCV



# evaluation metrics

from sklearn.metrics import make_scorer, roc_auc_score, auc, precision_score, recall_score, classification_report, roc_curve, accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve



# pipeline builder

from sklearn.pipeline import Pipeline



# decompostion 

from sklearn.decomposition import PCA



# ensemble models

import lightgbm as lgb



# over sampling model

from imblearn.over_sampling import SMOTE
# loading the data files

train = pd.read_csv('../input/train.csv', sep=',')

test = pd.read_csv('../input/test.csv', sep=',')


# taking sneak peak to datasets

print(f'Dimension of our Train data {train.shape} \n Data feature informations')

print(train.info())

print(f'Dimension of our Test data {test.shape} \n Data feature informations')

print(test.info())



print(train.head(), test.head())

print(f'Train columns: {train.columns}\nTest columns: {test.columns}')
### EDA for understanding datasets and getting clues for feature selections.





# Datatypes in dataset

print('Train target column datatype:',train.target.dtype)

print('Train var_0 column datatype:',train.var_0.dtype)





print('Train Describe:\n',train.describe(),'\nTest Describe:\n', test.describe())

print('Different values in target:\n',train.target.unique())

# Looking Variance

print('Train Variance:\n',train.var(),'\nTest Variance:\n', test.var())

# Looking Skewness

print('Train skewness:\n',train.skew(),'\nTest skewness:\n', test.skew())

# Missing value analysis

print('Train missing values:',train.isnull().sum().sum())

print('Test missing values:',test.isnull().sum().sum())

# Digging target variable

target = train['target']

print('Different values in target:\n',target.value_counts())

print('')

print("There are {}% target values with 1".format(100 *target.value_counts()[1]/(target.value_counts()[1] + target.value_counts()[0])))



sns.countplot(train['target'], palette='Set1')



plt.figure(figsize=(10,6))

train['target'].value_counts().plot.pie(autopct='%1.1f%%', explode=([0,0.1]))

plt.show()

# Boxplot Analysis

# Plot  features.

train.iloc[:, 2:50].plot(kind='box', figsize=[16,8])

train.iloc[:, 50:101].plot(kind='box', figsize=[16,8])

train.iloc[:, 101:151].plot(kind='box', figsize=[16,8])

train.iloc[:, 151:].plot(kind='box', figsize=[16,8])

# Distribution plot Analysis



# Function for quick plot of distribution

def plot_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(10,10,figsize=(18,22))



    for feature in features:

        i += 1

        plt.subplot(10,10,i)

        sns.distplot(df1[feature], hist=False,label=label1)

        sns.distplot(df2[feature], hist=False,label=label2)

        plt.xlabel(feature, fontsize=9)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=6)

        plt.tick_params(axis='y', which='major', labelsize=6)

    plt.show() 

    

t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]





# First 100 features dustribution

features = train.columns.values[2:102]

plot_feature_distribution(t0, t1, '0', '1', features)



# Rest 100 features dustribution

features = train.columns.values[102:]

plot_feature_distribution(t0, t1, '0', '1', features)





# 1. STD distribution of target variable



t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]



plt.figure(figsize=(16,6))

plt.title("Distribution of std values per row in the train set")

sns.distplot(t0[features].std(axis=1),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].std(axis=1),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of std values per column in the train set")

sns.distplot(t0[features].std(axis=0),color="green", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].std(axis=0),color="red", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()





# 2. Mean distribution of target variable

plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per row in the train and test set")

sns.distplot(t0[features].mean(axis=1),color="green", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].mean(axis=1),color="red", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of mean values per column in the train and test set")

sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].mean(axis=0),color="red", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# 3. Min distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of min values per row in the train set")

sns.distplot(t0[features].min(axis=1),color="orange", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].min(axis=1),color="darkblue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of min values per column in the train set")

sns.distplot(t0[features].min(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].min(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# 4. Max distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of max values per row in the train set")

sns.distplot(t0[features].max(axis=1),color="gold", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].max(axis=1),color="darkblue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of max values per column in the train set")

sns.distplot(t0[features].max(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].max(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# 5. Skew distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per row in the train set")

sns.distplot(t0[features].skew(axis=1),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].skew(axis=1),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of skew values per column in the train set")

sns.distplot(t0[features].skew(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].skew(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()





# 6. Kurtosis distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per row in the train set")

sns.distplot(t0[features].kurtosis(axis=1),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].kurtosis(axis=1),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of kurtosis values per column in the train set")

sns.distplot(t0[features].kurtosis(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].kurtosis(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# 7. Median distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of median values per row in the train set")

sns.distplot(t0[features].median(axis=1),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].median(axis=1),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of median values per column in the train set")

sns.distplot(t0[features].median(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].median(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# 8. Sum distribution

plt.figure(figsize=(16,6))

plt.title("Distribution of sum values per row in the train set")

sns.distplot(t0[features].sum(axis=1),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].sum(axis=1),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



plt.figure(figsize=(16,6))

plt.title("Distribution of sum values per column in the train set")

sns.distplot(t0[features].sum(axis=0),color="red", kde=True,bins=150, label='target  0')

sns.distplot(t1[features].sum(axis=0),color="blue", kde=True,bins=150, label='target  1')

plt.legend()

plt.show()



# Correlation Analysis

data_corr=train.drop(['target','ID_code'], axis=1).corr()

print('Maximum corr within all variables correlations :', np.sort(train.drop(['target','ID_code'], axis=1).corr())[:,-2:-1].max())



# Correlation Heatmap

plt.figure(figsize=(20,20))

sns.heatmap(data_corr, square=True)

plt.title('Feature Correlation')

plt.show()

# Data Preprocessiing



# Remove outliers

train_x = train.iloc[:, 1:]

IQR = train_x.quantile(.75) - train_x.quantile(.25)

print("Train.shape:",train.shape)

df_in = train[~((train_x < (train_x.quantile(.25) - 1.5 * IQR)) |(train_x > (train_x.quantile(.75) + 1.5 * IQR))).any(axis=1)]

df_out = train[((train_x < (train_x.quantile(.25) - 1.5 * IQR)) |(train_x > (train_x.quantile(.75) + 1.5 * IQR))).any(axis=1)]

print("df_in.shape:",df_in.shape)

print("df_out.shape:",df_out.shape)
print("df_in.target:\n", df_in['target'].value_counts())

print("df_out.target:\n", df_out['target'].value_counts())
# PCA Analysis





# feature extraction

pca = PCA().fit(train.drop(['target','ID_code'], axis=1))



plt.figure(figsize=(10,6))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.plot(pca.explained_variance_ratio_)

plt.title('Scree Plot')

plt.xlabel('Principal Component')

plt.ylabel('Eigenvalue')

leg = plt.legend(['Eigenvalues from PCA'], loc='best', borderpad=0.3,shadow=False,markerscale=0.4)

plt.grid(True)

plt.show()
# Using Stratified sampling

X_train, X_test, y_train, y_test = train_test_split(train.drop(['target', 'ID_code'], axis=1), train['target'], test_size=0.3, random_state=147, stratify=train.target)

print('Shape:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)




parameters = {'min_samples_leaf': [10,25]}

forest = RandomForestClassifier(max_depth=15, n_estimators=15)

grid_rfc = GridSearchCV(forest, parameters, cv=3, n_jobs=1, verbose=3, scoring=make_scorer(roc_auc_score))

grid_rfc.fit(X_train, y_train)

imp = grid_rfc.best_estimator_.feature_importances_

idx = np.argsort(imp)[::-1][-26:]



remove_features_RFC = train.columns[2:]

#train.drop(remove_features_RFC[idx],axis=1, inplace=True)

#test.drop(remove_features_RFC[idx],axis=1, inplace=True)

remove_col = remove_features_RFC[idx]

print('Removing features:', remove_col)

print('Train shape:',train.shape)
remove_col = ['var_187', 'var_113', 'var_7', 'var_126', 'var_189', 'var_62',

       'var_117', 'var_45', 'var_182', 'var_96', 'var_199', 'var_19', 'var_68',

       'var_77', 'var_3', 'var_25', 'var_14', 'var_41', 'var_73', 'var_30',

       'var_64', 'var_185', 'var_29', 'var_129', 'var_171', 'var_140']



trainFE = train.drop(remove_col,axis=1)

testFE = test.drop(remove_col,axis=1)

print('Removing features:', remove_col)

print('Columns left in Train :',trainFE.shape)

print('Columns left in Test :',testFE.shape)


print('Featuring Engineering raw data: Adding aggregates :')

idx = features = train.columns[2:]

for df in [test, train]:

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)



print('Train:', train.shape)

print('Test:' , test.shape)
def plot_new_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,4,figsize=(16,6))



    for feature in features:

        i += 1

        plt.subplot(2,4,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=11)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();



t0 = train.loc[train['target'] == 0]

t1 = train.loc[train['target'] == 1]

features = train.columns.values[202:]

plot_new_feature_distribution(t0, t1, 'target: 0', 'target: 1', features)


X_train, X_test, y_train, y_test = train_test_split(train.drop(['target','ID_code'], axis=1), train.target, test_size=0.3, random_state=147, stratify=train.target)



print('Shape:',X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Spot-Check Algorithms

models = []

models.append(( 'LR' , LogisticRegression(solver='liblinear')))

models.append(( 'CART' , DecisionTreeClassifier()))

models.append(( 'NB' , GaussianNB()))

models.append(('RFC', RandomForestClassifier()))

def cv_auc_score(models,scoring, num_folds=3):

    seed = 147 

    results = []

    names = []

    

    print('-> 3-Fold cross-validation ',scoring.__name__,'score for the training data for 4 classifiers.')

    for name, model in models:

        kfold = KFold( n_splits=num_folds, random_state=seed)

        cv_results = cross_val_score(model, X_train, y_train, cv=kfold,verbose=3 ,scoring=make_scorer(scoring))

        results.append(cv_results)

        names.append(name)

        print("Algo: ", name,'::',np.mean(cv_results))

    

    # Compare Algorithms

    fig = plt.figure()



    fig.suptitle( 'Algorithm Comparison: {}'.format(scoring.__name__ ))

    ax = fig.add_subplot(111)

    plt.boxplot(results)

    ax.set_xticklabels(names)

    plt.show()





# AUC score

num_folds = 3

scoring=roc_auc_score

print("Scores without StandardScale")

cv_auc_score(models, scoring=scoring, num_folds=num_folds)
# Accuracy score

scoring =  accuracy_score

print("Scores without StandardScale")

cv_auc_score(models, scoring=scoring, num_folds=num_folds)
def aur_prob_value_precision_recall_curve(models, X_train,  X_test,y_train, y_test):

    for name, model in models:

        model.fit(X_train, y_train)

        

        y_pred = model.predict_proba(X_test)

        y_pred2 = model.predict(X_test)

        

        print(name,' AUC prob: ',roc_auc_score(y_test, y_pred[:,1]))

        print(name,' AUC value: ',roc_auc_score(y_test, y_pred2))

        print(name,' f1 score: ',f1_score(y_test, y_pred2))

        

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:,1])

        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1])

        fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred2)

        

        fig, ax = plt.subplots(1,1, figsize=(6,6))  

        ax.plot(precision, recall)

        ax.plot(fpr, tpr, color='red')

        ax.plot(fpr2, tpr2, color='green')

        ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6)) 

        ax.legend([f'Precision-recall: {auc(recall, precision)}',f'AUC Prob: {auc(fpr, tpr)}',f'AUC Value: {auc(fpr2, tpr2)}'])

       # ax.legned()

        ax.set_xlabel('False Positive Rate')

        ax.set_ylabel('True Positive Rate')

        ax.set_title('Receiver operating characteristic {}'.format(name))

def classification_report_models(models,X_train, X_test, y_train, y_test ):

    for name, model in models:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(name, ':\n', confusion_matrix(y_test, y_pred))

        print(name,':\n',classification_report(y_test, y_pred))
print("AUC curve (Prob and Value) without StandardScale")

aur_prob_value_precision_recall_curve(models=models, X_train = X_train, X_test= X_test, y_train=y_train, y_test=y_test)

classification_report_models(models=models, X_train = X_train, X_test= X_test, y_train=y_train, y_test=y_test)


tr_X = train.drop([ 'ID_code'], axis=1)

test_X = test.drop(['ID_code'], axis=1)

for col in tr_X.drop(['target'], axis=1).columns:

    tr_X[col] = ((tr_X[col] - tr_X[col].mean()) / tr_X[col].std()).astype('float32')

for col in test_X.columns:

    test_X[col] = ((test_X[col] - test_X[col].mean()) / test_X[col].std()).astype('float32')

  



#Training data

X=tr_X.drop(['target'],axis=1)

Y=train['target']

#StratifiedKFold cross validator

cv=StratifiedKFold(n_splits=5,random_state=147,shuffle=True)

for train_index,valid_index in cv.split(X,Y):

    X_train1, X_valid=X.iloc[train_index], X.iloc[valid_index]

    y_train1, y_valid=Y.iloc[train_index], Y.iloc[valid_index]



print('Shape of X_train :',X_train1.shape)

print('Shape of X_valid :',X_valid.shape)

print('Shape of y_train :',y_train1.shape)

print('Shape of y_valid :',y_valid.shape)
from imblearn.over_sampling import SMOTE

#Synthetic Minority Oversampling Technique

sm = SMOTE(random_state=147, ratio=1.0)

#Generating synthetic data points

X_smote,y_smote=sm.fit_sample(X_train1,y_train1)

X_smote_v,y_smote_v=sm.fit_sample(X_valid,y_valid)
print("AUC curve (Prob and Value) with Standardization and SMOTE oversampling")

aur_prob_value_precision_recall_curve(models=models, X_train = X_smote, X_test= X_smote_v,  y_train=y_smote, y_test=y_smote_v)

classification_report_models(models=models, X_train = X_smote, X_test= X_smote_v, y_train=y_smote, y_test=y_smote_v)
#Training the model with simple train_test_split stratified data

#training data

lgb_train=lgb.Dataset(X_train,label=y_train)

#validation data

lgb_valid=lgb.Dataset(X_test,label=y_test)
params={'boosting_type': 'gbdt', 

          'max_depth' : -1, #no limit for max_depth if <0

          'objective': 'binary',

          'boost_from_average':False, 

          'nthread': 20,

          'metric':'auc',

          'num_leaves': 50,

          'learning_rate': 0.01,

          'max_bin': 100,      #default 255

          'subsample_for_bin': 100,

          'subsample': 1,

          'subsample_freq': 1,

          'colsample_bytree': 0.8,

          'bagging_fraction':0.5,

          'bagging_freq':5,

          'feature_fraction':0.08,

          'min_split_gain': 0.45, #>0

          'min_child_weight': 1,

          'min_child_samples': 5,

          'is_unbalance':True,

          }
# f1_score calculator function



def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities

    return 'f1', f1_score(y_true, y_hat), True



evals_result = {}



num_rounds=1000

lgbm1= lgb.train(params,lgb_train,num_rounds,valid_sets=[lgb_train,lgb_valid],feval=lgb_f1_score,verbose_eval=100,early_stopping_rounds = 500)

# confusion matrix

print('Confusion matrix: Simple Lightgbm')

confusion_matrix(y_test, lgbm1.predict(X_test).round())

def plot_roc(y_test, y_pred, name):

    fig, ax = plt.subplots(1,1, figsize=(6,6))  

    fpr2, tpr2, thresholds2 = roc_curve(y_test,  y_pred)

    ax.legend([f' {auc(fpr2, tpr2)}'])

           # ax.legned()

    ax.set_xlabel('False Positive Rate')

    ax.set_ylabel('True Positive Rate')

    ax.set_title('Receiver operating characteristic {}'.format(name))

    ax.plot(fpr2, tpr2)

plot_roc(y_test,  lgbm1.predict(X_test).round(), name='Simple LightGBM')
#Training the model with StratifiedKFold()+SMOTE() data

#training data

lgb_train2=lgb.Dataset(X_smote,label=y_smote)

#validation data

lgb_valid2=lgb.Dataset(X_smote_v,label=y_smote_v)
num_rounds=10000

lgbm3= lgb.train(params,lgb_train2,num_rounds,valid_sets=[lgb_train2,lgb_valid2],feval=lgb_f1_score,verbose_eval=1000,early_stopping_rounds = 5000)

# confusion matrix

print('Confusion matrix: SMOTE Lightgbm')

confusion_matrix(y_test, lgbm3.predict(X_test).round())

plot_roc(y_test,  lgbm3.predict(X_test).round(), name='SMOTE LightGBM')

#final submission



X_test=test.drop(['ID_code'],axis=1)

#predict the model, probability predictions

lightgbm_predict_prob3=lgbm3.predict(X_test,random_state=42,num_iteration=lgbm3.best_iteration)

lightgbm_predict_prob1=lgbm1.predict(X_test,random_state=42,num_iteration=lgbm1.best_iteration)



#Convert to binary output 1 or 0

lightgbm_predict3=lightgbm_predict_prob3.round()

lightgbm_predict1=lightgbm_predict_prob1.round()

submit=pd.DataFrame({'ID_code':test['ID_code'].values})

submit1=pd.DataFrame({'ID_code':test['ID_code'].values})



#submit['lightgbm_predict_prob']=lightgbm_predict_prob3

submit['target']=lightgbm_predict3.astype(int)

submit1['target']=lightgbm_predict1.astype(int)



submit.to_csv('submission.csv',index=False)

submit1.to_csv('submission1.csv',index=False)

submit1.head()

submit.head()
submit.shape