#Setup
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Any results you write to the current directory are saved as output.

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
test.info()
train.info()
train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue')
plt.xlabel('Number of Unique Values')
plt.ylabel('Count')
plt.title('Count of Unique Values in Columns of Type Int64')
# Quality of life changes:
test['Target'] = np.nan
data = train.append(test, ignore_index = False)

heads = data.loc[data['parentesco1'] == 1, :]
consistent = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
inconsistent = consistent[consistent != True]
for household in inconsistent.index:
    actual = int(train[(train['idhogar'] == household) & (train['parentesco1'] == 1)]['Target'])
    train.loc[train['idhogar'] == household, 'Target'] = actual
    
# let's check again for inconsistencies
consistent = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
inconsistent = consistent[consistent != True]
print('There are {} households with inconsistent Target labels.'.format(len(inconsistent)))
missing = train.isnull().sum().sort_values(ascending = False)
missing = missing[missing > 0]
missing = (missing/len(train)) # express missing counts as ratio of whole
missing.round(3).plot.bar()
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
# This counts the number of null entries in v18q1, grouped by v18q. 

# Attribute error raised for .isnull().sum(). the above seems to be the right way to do that. Why is this?
data = data.fillna({'v18q1':0})
# and now we want to graph
df = heads.loc[:, ['idhogar', 'v18q1', 'Target']]
relative = df.groupby('Target')['v18q1'].value_counts()
relative = relative/relative.groupby('Target').sum()
relative = relative.rename('counts').reset_index()
g = sns.catplot(data = relative, col = 'Target', kind = 'bar', y= 'counts', x = 'v18q1')
house_vars = ['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5']
nulls = train.loc[train.v2a1.isnull(), house_vars].sum()
notnulls = train.loc[train.v2a1.notnull(), house_vars].sum()
pd.DataFrame(data = {'isnull': nulls, 'notnull': notnulls})
data.loc[data.tipovivi1 == 1, 'v2a1'] = 0 # if the family owns the house set rent to 0. 

data['v2a1-missing'] = data['v2a1'].isnull()
data.loc[((data['age'] < 7) | (data['age'] > 19)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0

data['rez_esc-missing'] = data['rez_esc'].isnull()
heads = data.loc[data.parentesco1 == 1, :]
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc-missing']

ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['v18q', 'hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
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
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']

hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
heads = heads[id_ + hh_bool + hh_cont + hh_ordered]
maps = {'yes': 1, 'no': 0}

heads['dependency'] = heads.dependency.replace(maps).astype(np.float64)
heads['edjef'] = heads.edjefe.replace(maps).astype(np.float64) + heads.edjefa.replace(maps).astype(np.float64)
heads['leader_male'] = heads.edjefe.map(lambda x: x != 0)
heads = heads.drop(columns = ['edjefe', 'edjefa'])


sns.heatmap(heads[['r4t3', 'tamhog', 'tamviv', 'hhsize', 'hogar_total']].corr(), annot = True, fmt = '.3f')
heads.plot.scatter(x = 'tamviv', y = 'hhsize')

sns.jointplot(x = 'tamviv', y = 'hhsize', data = heads, kind = 'hex', gridsize = 10)
heads['hhsize-diff'] = heads.hhsize - heads.tamviv
heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
heads = heads.drop(columns = ['paredother', 'pisoother', 'techootro', 'abastaguafuera', 
                      'coopele', 'sanitario6', 'energcocinar4', 'elimbasu6', 
                      'tipovivi5', 'lugar6', 'area2'])
# delete the dummy variable trap columns

heads['wall'] = np.argmax(np.array(heads[['epared1', 'epared2', 'epared3']]), axis = 1)
heads['floor'] = np.argmax(np.array(heads[['eviv1', 'eviv2', 'eviv3']]), axis = 1)
heads['roof'] = np.argmax(np.array(heads[['etecho1', 'etecho2', 'etecho3']]), axis = 1)
heads = heads.drop(columns = ['epared1', 'epared2', 'epared3', 
                      'eviv1', 'eviv2', 'eviv3', 
                      'etecho1', 'etecho2', 'etecho3'])
heads['phones-per-capita'] = heads['qmobilephone']/heads['tamviv']
heads['tablets-per-capita'] = heads['v18q1']/heads['tamviv']
heads['rooms-per-capita'] = heads['rooms']/heads['tamviv']
heads['rent-per-capita'] = heads['v2a1']/heads['tamviv']
ind = data[id_ + ind_bool + ind_ordered]
ind['inst'] = np.argmax(np.array(ind[['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                                      'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']]), axis = 1)
ind = ind.drop(columns = [c for c in ind if c.startswith('instlevel')])
ind = ind.drop(columns = 'female')
ind = ind.drop(columns = 'Target') # We don't need Target data from the individual level
ind_agg_ordered = ind[['age', 'escolari', 'rez_esc', 'inst', 'idhogar']].groupby('idhogar').agg(['min', 'max'])
# rename the columns
new_col = []
for c in ind_agg_ordered.columns.levels[0]:
    for stat in ind_agg_ordered.columns.levels[1]:
        new_col.append(f'{c}-{stat}')
ind_agg_ordered.columns = new_col

ind_agg = ind.groupby('idhogar').agg('mean')
# rename the columns
new_col = []
for c in ind_agg:
    new_col.append(f'{c}-mean')
ind_agg.columns = new_col

#concatenate the dataframes
ind_agg = pd.concat([ind_agg, ind_agg_ordered], axis = 1)
# Create correlation matrix
corr_matrix = ind_agg.corr()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]
to_drop
ind_agg = ind_agg.drop(columns = to_drop)
final = heads.merge(ind_agg, on = 'idhogar', how = 'left')
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline

# custom scorer with macro f1
scorer = make_scorer(f1_score, greater_is_better = True, average = 'macro')

# this is where we will make our predictions when submitting
submissions_base = test[['Id', 'idhogar']].copy()

train_set = final[final.Target.notnull()].drop(columns = ['Id', 'idhogar', 'Target'])
test_set = final[final.Target.isnull()].drop(columns = ['Id', 'idhogar', 'Target'])
train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))
test_ids = list(final.loc[final.Target.isnull(), 'idhogar'])
def split_model(model, train_set, train_labels, test_set, test_ids):
    """ 
    Main learning module, two phases
    data is household level DataFrame
    """
    
    # Phase 1: Distinguish class 4 from classes 1-3
    train0_labels = np.array([x < 4 for x in train_labels])
    model.fit(train_set, train0_labels)
    test0_labels = model.predict(test_set)
    
    # Filter out the non-vulnerable households
    test1_set = test_set[test0_labels]
    test1_ids = [test_ids[i] for i in range(len(test_ids)) if test0_labels[i]]
    train1_set = train_set[train0_labels]
    train1_labels = train_labels[train0_labels]
    
    # Phase 2: Distinguish between classes 1-3
    model.fit(train1_set, train1_labels)
    labels = model.predict(test1_set)
    labels = pd.DataFrame({'idhogar': test1_ids, 'Target': labels})
    
    #Everything that hasnt been given a label by Phase 2 either has no parentesco1==1 or belongs to class 4
    submission = submissions_base.merge(labels, how = 'left', on = 'idhogar').drop(columns = 'idhogar')
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    return submission

#features = list(train_set.columns)
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), ('scaler', MinMaxScaler())])

# Impute missing values as well as scale data
train_set = pipeline.fit_transform(train_set)
test_set = pipeline.transform(test_set)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)
cv_score.mean(), cv_score.std()
RF_submission = split_model(model, train_set, train_labels, test_set, test_ids)
RF_submission.to_csv('RF_submission.csv', index = False)
from xgboost import XGBClassifier

model = XGBClassifier(objective = 'multi:softprob', num_class = 4, learning_rate = 0.006)
#cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)
#cv_score.mean(), cv_score.std()
# Phase 1: Distinguish class 4 from classes 1-3
model = XGBClassifier(learning_rate = 0.05)
train0_labels = np.array([x < 4 for x in train_labels])
model.fit(train_set, train0_labels)
test0_labels = model.predict(test_set)

# Filter out the non-vulnerable households
test1_set = test_set[test0_labels]
test1_ids = [test_ids[i] for i in range(len(test_ids)) if test0_labels[i]]
train1_set = train_set[train0_labels]
train1_labels = train_labels[train0_labels]

# Phase 2: Distinguish between classes 1-3
model = XGBClassifier(objective = 'multi:softprob', num_class = 4, learning_rate = 0.05)
model.fit(train1_set, train1_labels)
labels = model.predict(test1_set)
labels = pd.DataFrame({'idhogar': test1_ids, 'Target': labels})

#Everything that hasnt been given a label by Phase 2 either has no parentesco1==1 or belongs to class 4
submission = submissions_base.merge(labels, how = 'left', on = 'idhogar').drop(columns = 'idhogar')
submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
submission.to_csv('xgb_submission.csv', index = False)
import lightgbm as lgb

model = lgb.LGBMClassifier(boosting_type = 'dart', colsample_bytree = 0.88, learning_rate = 0.028,
                          min_child_samples = 10, num_leaves = 36, reg_alpha = 0.76, reg_lambda = 0.43,
                          subsample_for_bin = 40000, subsample = 0.54, class_weight = 'balanced',
                          objective = 'multiclass', n_estimators = 100, random_state = 10)
cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)
cv_score.mean(), cv_score.std()

model = lgb.LGBMClassifier(boosting_type = 'dart', colsample_bytree = 0.88, learning_rate = 0.028,
                          min_child_samples = 10, num_leaves = 36, reg_alpha = 0.76, reg_lambda = 0.43,
                          subsample_for_bin = 40000, subsample = 0.54, class_weight = 'balanced',
                          objective = 'binary', n_estimators = 100, random_state = 10)
train0_labels = np.array([x < 4 for x in train_labels])
model.fit(train_set, train0_labels)
test0_labels = model.predict(test_set)

# Filter out the non-vulnerable households
test1_set = test_set[test0_labels]
test1_ids = [test_ids[i] for i in range(len(test_ids)) if test0_labels[i]]
train1_set = train_set[train0_labels]
train1_labels = train_labels[train0_labels]

# Phase 2: Distinguish between classes 1-3
model = lgb.LGBMClassifier(boosting_type = 'dart', colsample_bytree = 0.88, learning_rate = 0.028,
                          min_child_samples = 10, num_leaves = 36, reg_alpha = 0.76, reg_lambda = 0.43,
                          subsample_for_bin = 40000, subsample = 0.54, class_weight = 'balanced',
                          objective = 'multiclass', n_estimators = 100, random_state = 10)
model.fit(train1_set, train1_labels)
labels = model.predict(test1_set)
labels = pd.DataFrame({'idhogar': test1_ids, 'Target': labels})

#Everything that hasnt been given a label by Phase 2 either has no parentesco1==1 or belongs to class 4
submission = submissions_base.merge(labels, how = 'left', on = 'idhogar').drop(columns = 'idhogar')
submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
submission.to_csv('lgb_submission.csv', index = False)













