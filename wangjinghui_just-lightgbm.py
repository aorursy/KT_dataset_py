import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.dependency = train.dependency.replace({"yes": 1, "no": 0}).astype(np.float64)
train.edjefa = train.edjefa.replace({"yes": 1, "no": 0}).astype(np.float64)
train.edjefe = train.edjefe.replace({"yes": 1, "no": 0}).astype(np.float64)
test.dependency = test.dependency.replace({"yes": 1, "no": 0}).astype(np.float64)
test.edjefa = test.edjefa.replace({"yes": 1, "no": 0}).astype(np.float64)
test.edjefe = test.edjefe.replace({"yes": 1, "no": 0}).astype(np.float64)
train.rez_esc = train.rez_esc.replace({99.0 : 5.0}).astype(np.float64)
test.rez_esc = test.rez_esc.replace({99.0 : 5.0}).astype(np.float64)
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone']

ind_non_bool = ['rez_esc', 'escolari', 'age','SQBescolari','SQBage','agesq']

hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']

hh_non_bool = ['v2a1', 'v18q1', 'meaneduc', 'SQBovercrowding', 'SQBdependency',
               'SQBmeaned', 'overcrowding', 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1',
               'r4m2', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'hhsize',
               'hogar_nin', 'hogar_adul', 'hogar_mayor', 'hogar_total',  'bedrooms',
               'qmobilephone', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin']

hh_cont = [ 'dependency', 'edjefe', 'edjefa']


ids = ['Id', 'idhogar', 'Target']
test['Target'] = np.nan
data = train.append(test)
data.info()
train.idhogar.nunique(), test.idhogar.nunique(), data.idhogar.nunique()
# data miss value count > 0
miss_count = data.isnull().sum() > 0
# data miss value count
misvalue_counts = data.isnull().sum()[miss_count]
# miss value percent
misvalue_percent = misvalue_counts/data.shape[0]*100

misvalue_percent
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
hh_train = train.loc[train.parentesco1 == 1, ids+hh_bool+hh_non_bool+hh_cont].reset_index()

target = hh_train[['Target']]
hh_train_ids = hh_train[['idhogar']]
# before filling miss value
hh_train = hh_train.drop(['Id','idhogar','Target','index'], axis=1)
# after filling miss value
hh_train_df = pd.DataFrame(imputer.fit_transform(hh_train),columns=list(hh_train.columns))

# add idhogar and Target columns
hh_train['idhogar'] = hh_train_ids
hh_train_df['idhogar'] = hh_train_ids
hh_train['Target'] = target
hh_train_df['Target'] = target
# indiviual level data on train set
ind_train = train.loc[ :, ids+ind_bool+ind_non_bool].reset_index()

ind_train_ids = ind_train[['idhogar']]
ind_target = ind_train[['Target']]

# before filling miss value, drop old index
ind_train = ind_train.drop(['Id','idhogar','Target','index'], axis=1)

# after filling miss value
ind_train_df=pd.DataFrame(imputer.fit_transform(ind_train),columns=list(ind_train.columns))

# add idhogar, Target
ind_train['idhogar'] = ind_train_ids
ind_train['Target'] = ind_target
ind_train_df['idhogar'] = ind_train_ids
ind_train_df['Target'] = ind_target
from collections import OrderedDict

mis_cols = ['v2a1','v2a1','v18q1','v18q1','meaneduc','meaneduc','SQBmeaned','SQBmeaned']


# Color mapping
colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})
label_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 
                               4: 'non vulnerable'})
#----------------------------------------------------------------------------

plt.figure(figsize = (12, 7))
for i, col in enumerate(mis_cols):
    ax = plt.subplot(4, 2, i + 1)
    # Iterate through the poverty levels
    for poverty_level, color in colors.items():
        # 核密度估计
        if (i%2 == 0):
            sns.kdeplot(hh_train_df.loc[hh_train_df.Target == poverty_level,col].dropna(), 
                        ax = ax, color = color, label = label_mapping[poverty_level])
            plt.title('%s before filling KDE'%(col.capitalize()))
            plt.xlabel('%s'%col)
            plt.ylabel('Density')
        else :
            sns.kdeplot(hh_train.loc[hh_train.Target == poverty_level, col].dropna(),
                        ax = ax, color = color, label = label_mapping[poverty_level])
            plt.title('%s after filling KDE'%(col.capitalize()))
            plt.xlabel('%s'%col)
            plt.ylabel('Density')
plt.subplots_adjust(top = 2.5)
cols = ['rez_esc','rez_esc']
plt.figure(figsize=(14, 2.5))
for i, col in enumerate(cols):
    ax = plt.subplot(1, 2, i + 1)
    for poverty_level, color in colors.items():
        if (i%2 == 0):
            sns.kdeplot(ind_train_df.loc[ind_train_df.Target == poverty_level,col].dropna(), 
                        ax = ax, color = color, label = label_mapping[poverty_level])
            plt.title('%s KDE'%(col.capitalize()))
            plt.xlabel('%s'%col)
            plt.ylabel('Density')
        else :
            sns.kdeplot(ind_train.loc[ind_train.Target == poverty_level, col].dropna(),
                        ax = ax, color = color, label = label_mapping[poverty_level])
            plt.title('%s filled miss KDE'%(col.capitalize()))
            plt.xlabel('%s'%col)
            plt.ylabel('Density')
plt.subplots_adjust(top = 2)
# test set
mis_hh = test.groupby(by='idhogar').parentesco1.agg('sum')==0

# idhogar,miss head of the household
mis_idhogar = test.groupby(by='idhogar').parentesco1.agg('sum')[mis_hh].index
pd.options.display.max_columns = 10
test.loc[test.idhogar.isin(mis_idhogar),:][['Id','idhogar','parentesco1']].sort_values(by='idhogar')
test.loc[test.Id == 'ID_99d27ab2f','parentesco1'] = 1
test.loc[test.Id == 'ID_49d05f9e6','parentesco1'] = 1
test.loc[test.Id == 'ID_b0874f522','parentesco1'] = 1
test.loc[test.Id == 'ID_ceeb5dfe2','parentesco1'] = 1
test.loc[test.Id == 'ID_aa8f26c06','parentesco1'] = 1
test.loc[test.Id == 'ID_e42c1dde2','parentesco1'] = 1
test.loc[test.Id == 'ID_9c12f6ebc','parentesco1'] = 1
test.loc[test.Id == 'ID_26d95edff','parentesco1'] = 1
test.loc[test.Id == 'ID_93fa2f7cc','parentesco1'] = 1
test.loc[test.Id == 'ID_bca8a1dde','parentesco1'] = 1
test.loc[test.Id == 'ID_4036d87e3','parentesco1'] = 1
test.loc[test.Id == 'ID_9f025fde6','parentesco1'] = 1
test.loc[test.Id == 'ID_6094ce990','parentesco1'] = 1
test.loc[test.Id == 'ID_00e8a868f','parentesco1'] = 1
test.loc[test.Id == 'ID_d0beee31f','parentesco1'] = 1
test.loc[test.Id == 'ID_894de66bc','parentesco1'] = 1
test.loc[test.Id == 'ID_aa650fb4a','parentesco1'] = 1
test.loc[test.Id == 'ID_139a474f3','parentesco1'] = 1
# household level test data
hh_test = test.loc[test.parentesco1 == 1, ids+hh_bool+hh_non_bool+hh_cont].reset_index()

hh_test_ids = hh_test[['idhogar']]

hh_test = hh_test.drop(['Id','idhogar','Target','index'], axis = 1)

# filling miss values
hh_test_df = pd.DataFrame(imputer.fit_transform(hh_test),columns=list(hh_test.columns))

# add idhogar columns
hh_test_df['idhogar'] = hh_test_ids
hh_test['idhogar'] = hh_test_ids
# indiviual level test data
ind_test = test.loc[:, ids+ind_bool+ind_non_bool].reset_index()

ind_test_ids = ind_test[['idhogar']]
ind_test = ind_test.drop(['Id','idhogar','Target','index'], axis = 1)
ind_test_df = pd.DataFrame(imputer.fit_transform(ind_test),columns=list(ind_test.columns))

# add idhogar columns
ind_test['idhogar'] = ind_test_ids
ind_test_df['idhogar'] = ind_test_ids
ind_train_groupobj = ind_train_df.groupby(by='idhogar')

ind_train_data = pd.DataFrame({'idhogar':ind_train_df.idhogar.unique()})
def AddFeatures(feature_df, cols, funcs, groupobj):
    for func in funcs:
        for col in cols:
            group_object = groupobj[col].agg(func).reset_index()
            group_object.rename(index=str, columns={col:col+'_'+func}, inplace=True)
            feature_df = feature_df.merge(group_object, on='idhogar', how='left')
    return feature_df
# indiviual bool features
ind_train_data = AddFeatures(ind_train_data, ind_bool, ['mean','sum'], ind_train_groupobj)

# indiviual non bool features
funcs = ['mean','min','max','median','sum','nunique']
ind_train_data = AddFeatures(ind_train_data, ind_non_bool, funcs, ind_train_groupobj)
ind_test_groupobj = ind_test_df.groupby(by='idhogar')
ind_test_data = pd.DataFrame({'idhogar':ind_test_df.idhogar.unique()})

ind_test_data = AddFeatures(ind_test_data, ind_bool, ['mean','sum'], ind_test_groupobj)

ind_test_data = AddFeatures(ind_test_data, ind_non_bool, funcs, ind_test_groupobj)
train_data = hh_train_df.merge(ind_train_data, on = 'idhogar', how='left')
test_data = hh_test_df.merge(ind_test_data, on = 'idhogar', how='left')
import gc
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
def model(train_data, test_data, n_folds = 10):
    # household id
    train_ids = train_data[['idhogar']]
    test_ids = test_data[['idhogar']]
    # Target/label
    labels = train_data[['Target']].astype(int)
    # drop idhogar, Target
    train_data = train_data.drop(['idhogar','Target'],axis = 1)
    test_data = test_data.drop(['idhogar'], axis = 1)
    # feature columns name
    feature_names = list(train_data.columns)
    # 10 folds cross validation
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 2018)
    # test predictions
    test_predictions = list()
    # validation predictions
    out_of_fold = np.zeros(train_data.shape[0])
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    # record scores : means f1_macro
    Valid_F1 = []
    Train_F1 = []
    # lightgbm not support f1_macro, so map
    Valid_Score = []
    Train_Score = []
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(train_data):
        # Training data for the fold
        train_features = train_data.loc[train_indices, :]
        train_labels = labels.loc[train_indices, :]
        # Validation data for the fold
        valid_features = train_data.loc[valid_indices, :]
        valid_labels = labels.loc[valid_indices, :]
        # Create the model
        model = lgb.LGBMClassifier(boosting_type='gbdt',n_estimators=2000, 
                                   objective = 'multiclass', class_weight = 'balanced',
                                   learning_rate = 0.03,  num_leaves = 31,
                                   reg_alpha = 0.1, reg_lambda = 0.3, num_class = 4,
                                   subsample = 0.8, n_jobs = -1, random_state = 2018)

        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'multi_error',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = 'auto',
                  early_stopping_rounds = 100, verbose = 200)
        # Record the best iteration
        best_iteration = model.best_iteration_
        # 
        test_predictions.append(model.predict(test_data, num_iteration = best_iteration))
        # feature importance
        feature_importance_values += model.feature_importances_ /n_folds
        # Record the best multi error
        valid_score = model.best_score_['valid']['multi_error']
        train_score = model.best_score_['train']['multi_error']
        Valid_Score.append(valid_score)
        Train_Score.append(train_score)
        # Record F1_macro score
        pred_valid = model.predict(valid_features, num_iteration = best_iteration)
        pred_train = model.predict(train_features, num_iteration = best_iteration)
        valid_f1 = f1_score(valid_labels, pred_valid, average='macro')
        train_f1 = f1_score(train_labels, pred_train, average='macro')
        Valid_F1.append(valid_f1)
        Train_F1.append(train_f1)

        # validation set result
        out_of_fold[valid_indices] = pred_valid
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        print('................................................')
        
    # feature importance
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    # overall valida
    Valid_F1.append(f1_score(labels, out_of_fold, average='macro'))
    Train_F1.append(np.mean(Train_F1))
    Valid_Score.append(np.mean(Valid_Score))
    Train_Score.append(np.mean(Train_Score))
    # dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train error': Train_Score,
                            'valid error': Valid_Score,
                            'train f1' : Train_F1,
                            'valid f1' : Valid_F1}) 

    # make submission.csv
    predict_df = pd.DataFrame(np.array(test_predictions).T)
    voting_result = [predict_df.iloc[x,:].value_counts().argmax() for x in range(predict_df.shape[0])]
    submission = test_ids.copy()
    submission['Target'] = voting_result
    # metric, fetaure importance , househodl target
    return metrics, feature_importances,submission
metric, feature_importance, submission = model(train_data, test_data, 10)
# filling mean, round 100
metric 
submit = test[['Id','idhogar']]

submit = submit.merge(submission, on = 'idhogar')

submit = submit.drop(['idhogar'],axis = 1)

submit.to_csv('submit.csv',index = False)
feature_importance = feature_importance.sort_values(by = 'importance')

feature_importance.set_index('feature').plot(kind='barh', figsize=(10, 40))
plt.title('Feature Importances')