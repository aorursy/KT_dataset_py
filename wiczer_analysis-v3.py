%matplotlib inline  
import pandas as pd
import numpy as np
import matplotlib.pyplot as py



py.rcParams['figure.figsize'] = [10, 8]
py.style.use('ggplot')
#początki
import os
print(os.listdir("../input"))
data_train = pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")
data_test = pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")
#data_train = pd.read_csv("../input/train.csv")
#data_test = pd.read_csv("../input/test.csv")
#funkcja ex_desc wyświetla dla danej kolumny jej opis wyciągnięty ze strony kaggla, przydaje się tych zmiennych jest od cholery
desc = pd.read_csv("../input/descriptions/descriptions.csv", sep = ",", header = None)
desc.columns = ['variable', 'description']

def ex_desc(var_name):
    output = str(desc.loc[desc['variable'] == var_name]['description']).split('\n')[0][3:]
    return output

labels = {1 : "extreme poverty", 2 : "moderate poverty", 3 : "vulnerable households", 4 : "non vulnerable households"}
id_cols = ['Id', 'idhogar', 'parentesco1', 'Target']
id_colst = ['Id', 'idhogar', 'parentesco1']

print("rozmiar tabeli train {} wierszy {} i kolumny".format(data_train.shape[0], data_train.shape[1]))
print("rozmiar tabeli test {} wierszy {} i kolumny".format(data_test.shape[0], data_test.shape[1]))
to_correct = data_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
to_correct = to_correct[to_correct == False]
houses_no_head = 0
houses_wrong_labels = 0
for idx in to_correct.index:
    all_household_members = data_train[data_train.idhogar == idx]
    head_label = all_household_members[all_household_members['parentesco1'] == 1][['Target']]
    if head_label is not None:
        data_train[data_train.idhogar == idx]['Target'] == head_label
        houses_wrong_labels += 1
    else:
        houses_no_head += 1
        
print("Domostwa ze złą etykietą {} domostwa z brakiem głowy {} - to nie jest błąd ".format(houses_wrong_labels, houses_no_head))
data_train.groupby('parentesco1').size()
data_train['tamviv'].plot.hist(normed = 1, bins = list(range(0, 20)))
data_object = data_train.select_dtypes(include = 'object')
for col in data_object.columns:
    print(col)
    print(np.unique(data_object[col]))
mapping={"yes" : 1, "no": 0}
data_train['dependency'] = data_train['dependency'].replace(mapping).astype(np.float64)
data_train['edjefa'] = data_train['edjefa'].replace(mapping).astype(np.float64)
data_train['edjefe'] = data_train['edjefe'].replace(mapping).astype(np.float64)
data_test['dependency'] = data_test['dependency'].replace(mapping).astype(np.float64)
data_test['edjefa'] = data_test['edjefa'].replace(mapping).astype(np.float64)
data_test['edjefe'] = data_test['edjefe'].replace(mapping).astype(np.float64)

missings = data_train.isnull().sum()
missings = missings.loc[missings != 0]
for i in range(0, len(missings.index)):
    print(missings.index[i], missings[i], ex_desc(missings.index[i]))
# If individual is over 19 or younger than 7 and missing years behind, set it to 0
data_train.loc[((data_train['age'] > 19) | (data_train['age'] < 7)) & (data_train['rez_esc'].isnull()), 'rez_esc'] = 0
data_test.loc[((data_test['age'] > 19) | (data_test['age'] < 7)) & (data_test['rez_esc'].isnull()), 'rez_esc'] = 0

# Add a flag for those between 7 and 19 with a missing value
data_train['rez_esc-missing'] = data_train['rez_esc'].isnull()
data_test['rez_esc-missing'] = data_test['rez_esc'].isnull()

data_train.loc[data_train['rez_esc'] > 5, 'rez_esc'] = 5
data_test.loc[data_test['rez_esc'] > 5, 'rez_esc'] = 5

data_train['rez_esc'] = data_train['rez_esc'].fillna(0)
data_test['rez_esc'] = data_test['rez_esc'].fillna(0)
# Fill in households that own the house with 0 rent payment
data_train.loc[(data_train['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
data_train['v2a1-missing'] = data_train['v2a1'].isnull()
data_train['v2a1-missing'].value_counts()

data_test['v2a1-missing'] = data_test['v2a1'].isnull()
data_test['v2a1-missing'].value_counts()

# Fill in households that own the house with 0 rent payment
data_test.loc[(data_test['tipovivi1'] == 1), 'v2a1'] = 0


# Create missing rent payment column
data_test['v2a1-missing'] = data_test['v2a1'].isnull()
data_test['v2a1-missing'].value_counts()
data_train['v2a1-missing'] = data_train['v2a1'].isnull()
data_train['v2a1-missing'].value_counts()
# Fill rest of them with zeros
data_train['v2a1'] = data_train['v2a1'].fillna(0)
data_test['v2a1'] = data_test['v2a1'].fillna(0)



data_train['v18q1'] = data_train['v18q1'].fillna(0)
data_test['v18q1'] = data_test['v18q1'].fillna(0)
# blind shot
data_train['meaneduc'] = data_train['meaneduc'].fillna(0)
data_test['meaneduc'] = data_test['meaneduc'].fillna(0)
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
print(sqr_)
data_train = data_train.drop(columns = sqr_)
data_test = data_test.drop(columns = sqr_)
data_train['sex'] = data_train['female']
data_train = data_train.drop(columns = ['female', 'male'])
data_test['sex'] = data_test['female']
data_test = data_test.drop(columns = ['female', 'male'])
np.unique(data_test['sex'])
data_train['walls'] = np.argmax(np.array(data_train[['epared1', 'epared2', 'epared3']]),axis = 1)
data_train['roof'] = np.argmax(np.array(data_train[['etecho1', 'etecho2', 'etecho3']]),axis = 1)
data_train['floor'] = np.argmax(np.array(data_train[['eviv1', 'eviv2', 'eviv3']]),axis = 1)

data_test['walls'] = np.argmax(np.array(data_test[['epared1', 'epared2', 'epared3']]),axis = 1)
data_test['roof'] = np.argmax(np.array(data_test[['etecho1', 'etecho2', 'etecho3']]),axis = 1)
data_test['floor'] = np.argmax(np.array(data_test[['eviv1', 'eviv2', 'eviv3']]),axis = 1)
to_drop = ['epared1', 'epared2', 'epared3', 'etecho1','etecho2', 'etecho3',  'eviv1', 'eviv2', 'eviv3']
data_train['walls+roof+floor'] = data_train['walls'] + data_train['roof'] + data_train['floor']
data_test['walls+roof+floor'] = data_test['walls'] + data_test['roof'] + data_test['floor']
data_train = data_train.drop(columns=to_drop)
data_test = data_test.drop(columns=to_drop)
data_train['hhsize_diff'] = (data_train['tamviv'] - data_train['hhsize'])/data_train['tamviv']
data_test['hhsize_diff'] = (data_test['tamviv'] - data_test['hhsize'])/data_test['tamviv']
data_train['has_dis'] = 0
l = 0
for ll in np.unique(data_train[['idhogar']]):
    vals = data_train.loc[data_train.idhogar == ll,'dis'].values
    if [1] in vals:
        l += 1
        data_train.loc[data_train['idhogar'] == ll,'has_dis'] = np.sum(vals)/len(vals)
        #dump_data['idhogar'] = np.where(dump_data.idhogar.values == ll, 1, dump_data.has_dis.values)
        
print("%d hh has dis" % l)

data_test['has_dis'] = 0
l = 0
for ll in np.unique(data_test[['idhogar']]):
    vals = data_test.loc[data_test.idhogar == ll,'dis'].values
    if [1] in vals:
        l += 1
        data_test.loc[data_test['idhogar'] == ll,'has_dis'] = np.sum(vals)/len(vals)
        #dump_data['idhogar'] = np.where(dump_data.idhogar.values == ll, 1, dump_data.has_dis.values)
        
print("%d hh has dis" % l)
#columns_to_drop = ['pca_%d'%d for d in range(1, 6)] + ['ica_%d'%d for d in range(1,6)]
#data_train = data_train.drop(columns = columns_to_drop)
#data_test = data_test.drop(columns = columns_to_drop)
'''
from sklearn.decomposition import FastICA, PCA

nb_pca = 6
ica = FastICA(nb_pca)
pca = PCA()
pca.fit(data_train.drop(columns=id_cols))
ica_train = ica.fit_transform(data_train.drop(columns=id_cols))
pca_train = pca.fit_transform(data_train.drop(columns=id_cols))

for i in range(0, nb_pca):
    ica_lab = 'ica_%d' % int(i+1)
    pca_lab = 'pca_%d' % int(i+1)
    data_train[ica_lab] = ica_train[:,i]
    data_train[pca_lab] = pca_train[:,i]
 
pca = PCA()
pca.fit(data_test.drop(columns=id_colst))
pca_test = pca.fit_transform(data_test.drop(columns=id_colst))
ica_test = ica.fit_transform(data_test.drop(columns=id_colst))

for i in range(0, nb_pca):
    ica_lab = 'ica_%d' % int(i+1)
    pca_lab = 'pca_%d' % int(i+1)
    data_test[ica_lab] = ica_test[:,i]
    data_test[pca_lab] = pca_test[:,i]
    '''
data_train['sex_diff'] = 0
data_train['adults_mean_age'] = 0
data_train['mean_kid_age'] = 0
data_train['lone_old_man'] = 0
data_train['non_family_members'] = 0

for ll in np.unique(data_train[['idhogar']]):
    vals = data_train.loc[data_train.idhogar == ll,'age'].values
    vals_sex = data_train.loc[data_train.idhogar == ll,'sex'].values
    vals_nf = data_train.loc[data_train.idhogar == ll,'parentesco12'].values
    nf_sum = np.sum(vals_nf)
    nf_len = len(vals_nf)
    if len(vals[vals>18]) !=0:
        ad_age_adult = vals[vals>18].mean(dtype='int')
    if len(vals[vals<18]) !=0:
        ad_age_kids = vals[vals<18].mean(dtype='int')
    
    if len(vals[vals >= 60]) == 1:
        data_train.loc[data_train['idhogar'] == ll,'lone_old_man'] = 1
        
    if nf_sum != 0 :
        data_train.loc[data_train['idhogar'] == ll,'non_family_members'] = nf_sum / nf_len
        
    data_train.loc[data_train['idhogar'] == ll,'adults_mean_age'] = ad_age_adult
    data_train.loc[data_train['idhogar'] == ll,'mean_kids_age'] = ad_age_kids
    
    
    data_train.loc[data_train['idhogar'] == ll,'sex_diff'] = np.sum(vals_sex)/len(vals_sex) 
        
data_train['electrinics'] = (data_train['v18q1'] + 2*data_train['computer'] + 2*data_train['television'] + data_train['qmobilephone'])/data_train['tamviv']
data_train['elec_cap'] = data_train['v18q'] + data_train['computer'] + data_train['television'] + data_train['mobilephone']

data_test['sex_diff'] = 0
data_test['adults_mean_age'] = 0
data_test['mean_kid_age'] = 0
data_test['lone_old_man'] = 0
data_test['non_family_members'] = 0

for ll in np.unique(data_test[['idhogar']]):
    vals = data_test.loc[data_test.idhogar == ll,'age'].values
    vals_sex = data_test.loc[data_test.idhogar == ll,'sex'].values
    vals_nf = data_test.loc[data_test.idhogar == ll,'parentesco12'].values
    nf_sum = np.sum(vals_nf)
    nf_len = len(vals_nf)
    if len(vals[vals>18]) !=0:
        ad_age_adult = vals[vals>18].mean(dtype='int')
    if len(vals[vals<18]) !=0:
        ad_age_kids = vals[vals<18].mean(dtype='int')
    
    if len(vals[vals >= 60]) == 1:
        data_test.loc[data_test['idhogar'] == ll,'lone_old_man'] = 1
        
    if nf_sum != 0 :
        data_test.loc[data_test['idhogar'] == ll,'non_family_members'] = nf_sum / nf_len
        
    data_test.loc[data_test['idhogar'] == ll,'adults_mean_age'] = ad_age_adult
    data_test.loc[data_test['idhogar'] == ll,'mean_kids_age'] = ad_age_kids
    
    
    data_test.loc[data_test['idhogar'] == ll,'sex_diff'] = np.sum(vals_sex)/len(vals_sex) 
        
data_test['electrinics'] = (data_test['v18q1'] + 2*data_test['computer'] + 2*data_test['television'] + data_test['qmobilephone'])/data_test['tamviv']
data_test['elec_cap'] = data_test['v18q'] + data_test['computer'] + data_test['television'] + data_test['mobilephone']


# male female diff
#oldest persons
#mean adult age
#mean kid age

#max education of head
# electrinic per capita v18q1 computer television qmobilephone
#
def create_agg_features(list_of_features, DF):
    list_of_agg = ['min', 'max', 'mean', 'std']
    
    for feature in list_of_features:
        for li in list_of_agg:
            
            DF['%s_%s'%(feature, li)] = 0
            
    for feature in list_of_features:
    
        
        for  ll in np.unique(DF[['idhogar']]):
            vals = DF.loc[DF.idhogar == ll,feature].values
            min_1, max_1, mean_1, std_1 = np.min(vals), np.max(vals), np.mean(vals), np.std(vals)
            
            DF.loc[DF['idhogar'] == ll ,'%s_min'%feature ] = min_1
            DF.loc[DF['idhogar'] == ll ,'%s_max'%feature ] = max_1
            DF.loc[DF['idhogar'] == ll ,'%s_mean'%feature ] = mean_1
            DF.loc[DF['idhogar'] == ll ,'%s_std'%feature ] = std_1
            
    return DF

def create_agg_features_sum(list_of_features, DF):
    for feature in list_of_features:
            DF['%s_%s'%(feature, 'sum')] = 0
            
    for feature in list_of_features:
    
        for  ll in np.unique(DF[['idhogar']]):
            vals = DF.loc[DF.idhogar == ll,feature].values
            
            DF.loc[DF['idhogar'] == ll ,'%s_sum'%feature ] = np.sum(vals, dtype = int)            
    return DF
def flg_to_categorical(feature, data_frame):
    
    list_of_features = [c  for c in data_frame.columns if c.startswith(feature)==True ]
    new_feature = feature + '_cat'
    print(list_of_features, "-->", new_feature)
    
    data_frame[new_feature] = np.argmax(data_frame[list_of_features].values, axis = 1)
    data_frame = data_frame.drop(columns = list_of_features)
    return data_frame

def scaled_features(list_of_features, DF):
    
    for feature in list_of_features:
        DF['%s_%s' % (feature, 'scaled')] = DF[feature] / DF['hogar_total']
    
    return DF
    
def flg_to_cat_v2(list_of_features, new_feature, data_frame):

    #list_of_features = ['public', 'planpri', 'noelec', 'coopele']
    #new_feature = 'elect_source' + '_cat'
    print(list_of_features, "-->", new_feature)
    
    data_frame[new_feature] = np.argmax(data_frame[list_of_features].values, axis = 1)
    data_frame = data_frame.drop(columns = list_of_features)
    return data_frame
parentesco_like = ['parentesco' +str(i) for i in range(1, 13)]

data_train = create_agg_features_sum(parentesco_like, data_train)
data_test = create_agg_features_sum(parentesco_like, data_test)
estadocivil = ['estadocivil' + str(i) for i in range(1, 7)]

data_train = create_agg_features_sum(estadocivil, data_train)
data_test = create_agg_features_sum(estadocivil, data_test)
data_train = create_agg_features(['age'], data_train)
data_test = create_agg_features(['age'], data_test)

data_train = create_agg_features(['escolari'], data_train)
data_test = create_agg_features(['escolari'], data_test)
data_train = scaled_features(['hogar_nin', 'hogar_adul', 'hogar_mayor','v2a1','v18q1','qmobilephone'], data_train)
data_test = scaled_features(['hogar_nin', 'hogar_adul', 'hogar_mayor','v2a1','v18q1','qmobilephone'], data_test)
data_train = flg_to_cat_v2(['public', 'planpri', 'noelec', 'coopele'], 'elect_source',data_train)
data_test = flg_to_cat_v2(['public', 'planpri', 'noelec', 'coopele'], 'elect_source',data_test)

data_train = flg_to_cat_v2([ 'abastaguano',  'abastaguafuera','abastaguadentro'], 'water_source', data_train)
data_test = flg_to_cat_v2([ 'abastaguano',  'abastaguafuera','abastaguadentro'], 'water_source', data_test)


data_train = flg_to_categorical('tipovivi', data_train)
data_test = flg_to_categorical('tipovivi', data_test)
data_train = flg_to_categorical('sanitario', data_train)
data_test = flg_to_categorical('sanitario', data_test)
data_train = flg_to_categorical('energcocinar', data_train)
data_test = flg_to_categorical('energcocinar', data_test)
data_train = flg_to_categorical('instlevel', data_train)
data_train = flg_to_categorical('elimbasu', data_train)
data_test = flg_to_categorical('instlevel', data_test)
data_test = flg_to_categorical('elimbasu', data_test)
data_train = create_agg_features(['instlevel_cat'], data_train)
data_test = create_agg_features(['instlevel_cat'], data_test)

data_train = flg_to_categorical('estadocivil', data_train)
data_test = flg_to_categorical('estadocivil', data_test)

parentes_co_like = ['parentesco%d'%i for i in range(2, 13)] + ['parentesco1_sum', 'parentesco2_sum']

data_train = data_train.drop(columns = parentes_co_like)
data_test = data_test.drop(columns = parentes_co_like)
for i in  data_train.columns:
    print(i)
corr_all  = data_train.drop(columns=['Target', 'idhogar', 'parentesco1']).select_dtypes(exclude='object')

corr_all = corr_all.corr()

# trzeba usunąć diagonalę macierzy korelacji
np.fill_diagonal(corr_all.values, 0)

corr_all.head()

to_drop = [col for col in corr_all.columns if any(abs(corr_all[col]) > 0.95) ]

to_drop = np.unique(to_drop)
for td in to_drop:
    print("%s %s" % (td, ex_desc(td)))
hh_like = ['r4t3', 'hhsize', 'tamviv', 'tamhog', 'hogar_total' ]
data_train[hh_like].corr()
to_drop_22 = ['hhsize', 'tamhog', 'hogar_total', 'rooms']
data_train = data_train.drop(columns = to_drop_22)
data_test = data_test.drop(columns = to_drop_22)
to_drop = list(set(to_drop) - set(to_drop_22))
size_like_var = data_train[to_drop]
print(size_like_var.corr().iloc[:5,:5])
to_drop_v1 = [ 'area1','age', 'pca_1', 'pca_2','pca_5','ica_1','ica_4','ica_5', 'ica_6', 'hhsize_diff', 'tamhog']
to_drop = list(set(to_drop) - set(to_drop_v1))
size_like_var = data_train[to_drop]
print(size_like_var.corr())
#print(to_drop)
data_train = data_train.drop(columns = to_drop)
data_test = data_test.drop(columns = to_drop)
from sklearn.ensemble import RandomForestClassifier


n_estimators = 150
max_depth = 6
RFC =RandomForestClassifier(n_estimators = n_estimators, max_depth=max_depth)

X = data_train.drop(columns=id_cols)
Y = np.ravel(data_train[['Target']])
print(X.shape)
RFC.fit(X,Y)
importances = RFC.feature_importances_
indices = np.argsort(importances)[::-1]
for i in range(0, 100):
    idx = indices[i]
    print("{0:2} {1:20} {2:5} {3:10}" \
          .format(i+1, X.columns[idx], np.round(importances[idx], 3), ex_desc(X.columns[idx])))
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
train_data = data_train.loc[data_train.parentesco1 == 1,]
test_data = data_test.loc[data_test.parentesco1 == 1,]
print(data_train.shape, data_test.shape)
print(train_data.shape, test_data.shape)
import xgboost as xgb
#import lightgbm as xgb
from hyperopt import hp, tpe
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


import random
import itertools
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
N_HYPEROPT_PROBES = 5
EARLY_STOPPING = 70 #change to 80
HOLDOUT_SEED = 123456
HOLDOUT_SIZE = 0.10
HYPEROPT_ALGO =  tpe.rand.suggest
SEED0 = random.randint(1,1000000000)
NB_CV_FOLDS = 4 #chagne to 5
NB_ROUNDS = 400

def f1_macro(preds, dtrain):
    labels = dtrain.get_label()
    return 'f1-macro', f1_score(labels, preds, average = 'macro')
space ={
    'booster '    : 'dart',
    'silent'      : 1,
    'objective'   : 'multi:softmax',
    'num_class'   : 4,
    'class_weights': 'balanced',
    
    'drop_rate'     : hp.uniform('drop_rate', 0.05, 0.5),
    'subsample': hp.uniform('dart_subsample', 0.5, 1),
    'subsample_freq': hp.quniform('dart_subsample_freq', 1, 10, 1),
    'num_boost_rounds' : NB_ROUNDS,
    #'num_boost_rounds' :hp.choice('num_boost_rounds', np.arange(100, 500, 10, dtype=int)),
     
    'max_depth'   : hp.choice("max_depth", np.arange(2, 30, 1, dtype='int')),
    'num_leaves': hp.quniform('num_leaves', 3, 50, 1),
   
    'lambda_l1'       : hp.uniform('lambda_l1', 1e-4, 1e-6 ),
    'lambda_l2'      : hp.uniform('lambda_l2', 1e-4, 1e-6 ),
    
    'max_delta_step'   : hp.choice('max_delta_step', np.arange(0,5,   dtype=int)),
    'min_child_weight ': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
    'learning_rate'    : hp.loguniform('learning_rate', -6.9, -2.3),
    
    'seed'             : hp.randint('seed',2000000)
   }
data_to_fit = train_data.drop(columns = ['Id', 'idhogar', 'parentesco1'])
X, y =  data_to_fit.drop(columns = ['Target']).values, data_to_fit[['Target']].values
y = np.ravel(y - 1).astype(int)
y
print("{0: .3} {1: .4}".format(1.11225352355345, .235346345345345346456456))
def objective(space):
    
    global X, y, curr_best_score, best_params

    
    params = sample(space)
    #model = xgb.XGBClassifier(**params, n_jobs = -1, n_estimators = NB_ROUNDS)
    
    cv_scores = []
    nb_rounds = []
       
    for cv_ in range(0, NB_CV_FOLDS):
        Xtrain,  Xtest, ytrain, ytest = train_test_split(X, y, random_state=np.random.randint(0, 1e+8), test_size=0.3, stratify = y )
        
        Xtrain = xgb.DMatrix(Xtrain, ytrain)
        Xtest = xgb.DMatrix(Xtest, ytest)
        
        model = xgb.train(params, Xtrain, 
                          #early_stopping_rounds = EARLY_STOPPING,
                          evals = [(Xtest, 'test')],
                          feval = f1_macro,
                          verbose_eval = False)
        
        preds = model.predict(Xtest, ntree_limit=params['num_boost_rounds'])
        cv_scores.append(f1_score(ytest, preds, average = 'macro'))
        #nb_rounds.append(model.best_iteration)
        

    score, score_std = np.mean(cv_scores), np.std(cv_scores)

    print( 'cv_score={0: .3f} +- {1: .3f} BEST_SCORE: {2: .3f}'.format( score, score_std, curr_best_score ) )
    
    if score > curr_best_score:
        best_params = params
        curr_best_score  = score
        
        
#         do_submit = True

#     if do_submit:
#         submit_guid = uuid4()

#         print('Compute submissions guid={}'.format(submit_guid))

#         y_submission = gbm_model.predict(xgb_test, ntree_limit = n_rounds)
#         submission_filename = 'xgboost_score={:13.11f}_submission_guid={}.csv'.format(score,submit_guid)
#         pd.DataFrame(
#         {'id':test_id, 'target':y_submission}
#         ).to_csv(submission_filename, index=False)
       
    loss = 1 - score
    return {'loss': loss, 'status': STATUS_OK}

curr_best_score = 0
trials = Trials()
best_params = None
fit_report = None
best = fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=0)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')
print(curr_best_score)
print('params:\n',best_params)



X, y =  data_train.drop(columns = id_cols), np.ravel(data_train[['Target']]-1)
Xtest = test_data.drop(columns = id_colst)
xgb_fulldata = xgb.DMatrix(X, y)
xgb_test = xgb.DMatrix(Xtest)
best_model = xgb.train(best_params, xgb_fulldata, verbose_eval=False)

xgb.plot_importance(best_model, height = 0.4, max_num_features=40)
#make submission
ytestpreds = best_model.predict(xgb_test).astype('int') + 1
rest_of_ids = data_test.Id

my_submission = pd.DataFrame({'Id': data_test.Id, 'Target': 4})

for idd,val in zip(test_data.Id, ytestpreds):
        my_submission.loc[my_submission.Id == idd,'Target'] = val


my_submission.to_csv('submission.csv', index=False)


for x,y in zip(data_train.drop(columns = ['Target']).columns, data_test.columns):
    if x != y:
        print(x,y)
