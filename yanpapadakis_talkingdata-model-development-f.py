import pandas as pd
import numpy as np
import gc
import pickle
import lightgbm as gbm
from random import random, seed


# Model Parameters
agg_period = '30min'
# estimation window limit
est_window_start = '2017-11-08 0:00:00'
est_window_end   = '2017-11-09 15:59:59'
train = pd.read_csv('../input/train.csv', index_col = 'click_time', 
                    usecols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'], 
                    dtype = {'is_attributed':bool,'app':np.uint16,'device':np.uint16,'os':np.uint16,'channel':np.uint16,'ip':np.uint32}, 
                    parse_dates=['click_time'])
train = train['2017-11-06 16:00:00':'2017-11-09 15:59:59']
train.reset_index(inplace=True)
train['click_time'] = train['click_time'].dt.floor(agg_period)
gc.collect()
train.info()
supp = pd.read_csv('../input/test_supplement.csv',
                   index_col = 'click_time',
                   usecols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'],
                   dtype = {'click_id':np.uint32,'app':np.uint16,'device':np.uint16,
                            'os':np.uint16,'channel':np.uint16,'ip':np.uint32},
                   parse_dates=['click_time'])

supp = supp.loc['2017-11-10 04:00:00':'2017-11-10 15:59:59']
supp.reset_index(inplace=True)
supp['click_time'] = supp['click_time'].dt.floor(agg_period)
gc.collect()
supp.info()
# Categorical Variables
cats = ['app','ip','channel','device','os']
freq = dict()
freq_total = len(train) + len(supp)
for p_ in [('ip',),('channel',),('app',),('os',),('device',),('click_time',),
          ('ip','click_time'),('channel','click_time'),('app','click_time'),('os','click_time'),('device','click_time'),
          ('channel','app'),('ip','device'),('ip','os'),('channel','app','click_time'),
          ('channel','ip'),('app','ip'),('app','os'),('app','device')]:
    p = tuple(sorted(p_))
    freq[p] = pd.concat([train.groupby(p).is_attributed.size(),
                         supp.groupby(p).click_id.size()],axis=1).sum(axis=1).rename('_'.join(p)+'_freq')
    print(p,'{:,d}'.format(len(freq[p])),end=' ')
    freq[p] = freq[p][freq[p] > 1]
    print(' reduced to ','{:,d}'.format(len(freq[p])))
    gc.collect()
top_k = dict()
k_of = {'app':20,'ip':5}
z = pd.concat([train[['ip','app']].groupby(['ip','app']).size(),
               supp[['ip','app']].groupby(['ip','app']).size()],axis=1).sum(axis=1).rename('tot')
for p in ['ip','app']:
    x = z.reset_index(p).sort_values([p,'tot'], ascending=[True,False]).groupby(p,sort=False).head(k_of[p])
    top_k[p] = x.groupby(p,sort=False).tot.sum().rename(p+'_top_k')
del z
del x
gc.collect()
g_rate = dict()
for p in cats:
    g_rate[p] = pd.cut(train.groupby(p).is_attributed.mean(),[0,.0018,.005,1],labels=False)
g_rate['default'] = 1
train = pd.concat([train[train.is_attributed==True],train[train.is_attributed==False].sample(3 * 10**6)])
gc.collect()
transform_code = '''
def transf(df,verbose=True):
            
    for p in freq:
        if verbose:
            print(p,pd.Timestamp.now('US/Eastern').strftime('%H:%M'))
        df = df.join(freq[p],on=p)
        df[df.columns[-1]].fillna(1,inplace=True)
    gc.collect()

    for p in top_k:
        df[p+'_top_k_pct'] = df[p].map(top_k[p]) / df[p+'_freq']
    gc.collect()
    
    if verbose:
        print('Conditional Probabilities')
    fmt = '{}_over_{}'
    for p in freq:
        p_code = '_'.join(p)
        if len(p) == 2:
            for a in p:
                df[fmt.format(p_code,a)] = df[p_code+'_freq'] / df[a+'_freq']
        if len(p) == 3:
            for a in p:
                df[fmt.format(p_code,a)] = df[p_code+'_freq'] / df[a+'_freq']
            for b_ in zip(p,p[1:]+p[0:1]):
                b = tuple(sorted(b_))
                b_code = '_'.join(b)
                df[fmt.format(p_code,b_code)] = df[p_code+'_freq'] / df[b_code+'_freq']
                
    for p in freq:
        p_code = '_'.join(p)
        df[p_code+'_pct'] = df.pop(p_code+'_freq') / freq_total
        gc.collect()
        
    for p in cats:
        df[p+'_g_rate'] = df[p].map(g_rate[p]).fillna(g_rate['default'])
    
    df['tm'] = df.click_time.dt.hour + df.click_time.dt.minute / 60
    return df
'''

exec(transform_code)

foo = train.iloc[-15:].copy()
foo = transf(foo,verbose=True)
foo.head(10)
train = transf(train.copy())
train.set_index('click_time',inplace=True)
train.sort_index(inplace=True)
train.info()
# Store Data Transformation Parameters
pickle.dump([top_k,freq,freq_total,g_rate,transform_code],open('td_dicts.pkl','wb'))
# Make room
del top_k
del freq
del g_rate
gc.collect()
predictors = train.columns
dropped = ['is_attributed','device_g_rate'] #,'device','os','app','channel','ip']
print('Dropped Predictors: ',*[p for p in predictors if p in dropped],sep='\t')
plist = [p for p in predictors if p not in dropped]
# train gbm
print('Begin Model Estimation ...')
evals_result = {}  # record of fit performance

gc.collect()

l_rates = 'lambda i: max(0.1, 0.25 * (0.999 ** i))'

params = {'boosting_type':'gbdt', 'objective': 'binary', 'metric': 'auc',
          'num_leaves': 2000, 'min_data_in_leaf': 4250,
          'bagging_freq':1, 'bagging_fraction':1, 'feature_fraction':.1,
          'max_depth': 7, 'scale_pos_weight':1  #, 'mc':monotone_constr
        }

settings = dict(init_model=None, num_boost_round=1250, verbose_eval=20, early_stopping_rounds=50)

gbm_train = gbm.Dataset(train.loc['2017-11-08 00:00:00':,plist],
                        train.is_attributed['2017-11-08 00:00:00':]
                       )#,categorical_feature=cats)
gbm_eval  = gbm.Dataset(train.loc[:'2017-11-08 00:00:00',plist],
                        train.is_attributed[:'2017-11-08 00:00:00'],
                        reference=gbm_train)
gbm_model = gbm.train(params, gbm_train, valid_sets=[gbm_train,gbm_eval], evals_result=evals_result,
                      learning_rates=eval(l_rates), **settings)
print('\nTraining Dataset Size = {:,d} / Validation Dataset Size = {:,d}'.format(gbm_train.num_data(),gbm_eval.num_data()))
print('Number of Features {}\n'.format(gbm_train.num_feature()))
gc.collect()
# export model
pickle.dump(gbm_model,open('model.pkl','wb'))
pickle.dump({'cats':cats,'agg_period':agg_period,'plist':plist},open('param.pkl','wb'))
_ = gbm.plot_metric(evals_result,metric='auc')
gbm.plot_importance(gbm_model,figsize=(8,11))
_ = gbm.plot_importance(gbm_model,importance_type='gain',figsize=(8,11))
m = gbm_model.dump_model()
pickle.dump([m['feature_names'],m['tree_info'][-1]['tree_structure']],open('model_out.pkl','wb'))