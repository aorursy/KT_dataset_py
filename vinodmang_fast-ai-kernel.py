import os
import numpy as np
import pandas as pd
import os
from kaggle.competitions import twosigmanews
from fastai.structured import *
from fastai.column_data import *
PATH_WRITE = "/kaggle/working/"
env = twosigmanews.make_env()
(market_train_df, _) = env.get_training_data()
market_train_df['label'] = market_train_df['returnsOpenNextMktres10'] > 0
def add_datepart(df, fldname, drop=False):
    fld = df[fldname]
    #if not np.issubdtype(fld.dtype, np.datetime64):
     #   df[fldname] = fld = pd.to_datetime(fld, 
      #                               infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 
            'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 
            'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt,n.lower())
    df[targ_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)
        
add_datepart(market_train_df,"time")
cat_vars = ['assetCode','timeYear', 'timeMonth',
       'timeWeek', 'timeDay', 'timeDayofweek', 'timeDayofyear',
       'timeIs_month_end', 'timeIs_month_start', 'timeIs_quarter_end',
       'timeIs_quarter_start', 'timeIs_year_end', 'timeIs_year_start',
       'timeElapsed']

#cat_vars = ['assetCode']
cont_vars = ['volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']

for v in cat_vars:
    market_train_df[v] = market_train_df[v].astype('category').cat.as_ordered()
for v in cont_vars + ['label']:
    market_train_df[v] = market_train_df[v].astype('float32').fillna(0).astype('float32')
    
#import sklearn
#from sklearn import model_selection
#train_indices,test_indices = sklearn.model_selection.train_test_split(market_train_df.index.values,test_size=.25,random_state=333)
samp_size = len(market_train_df)
#train_ratio = 0.75
train_ratio = 0.9
train_size = int(samp_size * train_ratio); train_size
#test_indices = list(range(train_size, len(market_train_df)))
train_indices,test_indices = sklearn.model_selection.train_test_split(market_train_df.index.values,test_size=.10,random_state=333)
df, y, nas,mapper = proc_df(market_train_df[cat_vars + cont_vars + ['label']], 'label',do_scale=True)
md = ColumnarModelData.from_data_frame(PATH_WRITE,test_indices,df,y,cat_flds=cat_vars,bs=4096*4)
cat_sz = [(c,len(market_train_df[c].cat.categories) + 1) for c in cat_vars]
emb_szs = [(c,min(50,(c+1)//2)) for _,c in cat_sz]
m = md.get_learner(emb_szs,len(df.columns) - len(cat_vars),.04,1,[1000,500],[.001,.01],y_range=[0,1],tmp_name=f"{PATH_WRITE}tmp", models_name=f"{PATH_WRITE}models")
lr = 1e-3
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce
m.lr_find()
m.sched.plot()
import tensorflow as tf
m.fit(lr, 4, metrics=[F.binary_cross_entropy])
pred_test = m.predict(False)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
confidence_valid = pred_test*2 -1
plt.hist(confidence_valid,bins="auto")
y_valid = market_train_df.loc[test_indices,"returnsOpenNextMktres10"] >0
print(accuracy_score(confidence_valid>0,y_valid))
class ColumnarDataset(Dataset):
    def __init__(self, cats, conts, y):
        n = len(cats[0]) if cats else len(conts[0])
        self.cats = np.stack(cats, 1).astype(np.int64) if cats else np.zeros((n,1))
        self.conts = np.stack(conts, 1).astype(np.float32) if conts else np.zeros((n,1))
        self.y = np.zeros((n,1)) if y is None else y[:,None]

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return [self.cats[idx], self.conts[idx], self.y[idx]]

    @classmethod
    def from_data_frames(cls, df_cat, df_cont, y=None):
        cat_cols = [c.values for n,c in df_cat.items()]
        cont_cols = [c.values for n,c in df_cont.items()]
        return cls(cat_cols, cont_cols, y)

    @classmethod
    def from_data_frame(cls, df, cat_flds, y=None):
        return cls.from_data_frames(df[cat_flds], df.drop(cat_flds, axis=1), y)