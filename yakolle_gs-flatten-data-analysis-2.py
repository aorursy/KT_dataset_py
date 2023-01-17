import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import re

import json
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import trange,tqdm
from collections import Counter

pd.options.display.max_rows = 8
pd.options.display.max_columns = 999
print(os.listdir("../input"))
data_dir = '../input/gs-flatten-data-analysis-1'
train_df_r = pd.read_pickle(os.path.join(data_dir, 'train_df_r'), compression='gzip')
test_df_r = pd.read_pickle(os.path.join(data_dir, 'test_df_r'), compression='gzip')
print(f'train_df_r: {train_df_r.shape}, test_df_r: {test_df_r.shape}')
def agg_most_common(_df, _cols, n=3, window_type='u', gp_col=None):
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    _df = _df.loc[:, _cols + gp_cols].set_index(gp_cols)
    gc.collect()
    cnt = _df.groupby(gp_cols)[_cols[0]].count()

    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[ids1].copy()
    for _col in _cols:
        for k in range(1, n):
            df1[f'{_col}_{window_type}_{k}'] = -127
            df1[f'{_col}_{window_type}_{k}'] = df1[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)

    ids2 = cnt.loc[2 == cnt].index
    gp = _df.loc[ids2].groupby(gp_cols)
    df2 = gp.min().join(gp.max(), rsuffix=f'_{window_type}_1')
    for _col in _cols:
        for k in range(2, n):
            df2[f'{_col}_{window_type}_{k}'] = -127
            df2[f'{_col}_{window_type}_{k}'] = df2[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)
        df2[f'{_col}_{window_type}_1'] = df2[f'{_col}_{window_type}_1'].astype(_df[_col].dtype)

    def most_common(rows):
        _cnt = sorted(Counter(rows).items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
        return [_cnt[i][0] if len(_cnt) > i else -127 for i in range(n)]

    ids3 = cnt.loc[cnt >= 3].index
    df3 = _df.loc[ids3].groupby(gp_cols).agg(most_common)
    for _col in _cols:
        for k in range(1, n):
            df3[f'{_col}_{window_type}_{k}'] = df3[_col].str.get(k)
            df3[f'{_col}_{window_type}_{k}'] = df3[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)
        df3[_col] = df3[_col].str.get(0)
        df3[_col] = df3[_col].astype(_df[_col].dtype)

    return df1.append([df2, df3], sort=False).rename(columns={_col: f'{_col}_{window_type}_0' for _col in _cols})
id_col = 'fullVisitorId'

cols = ['device.isMobile','totals.bounces','totals.newVisits','trafficSource.adwordsClickInfo.isVideoAd',
        'trafficSource.isTrueDirect','visit_weekend']
train_df_w = agg_most_common(train_df_r, cols, n=2, window_type='w', gp_col='window_id')
gc.collect()
test_df_w = agg_most_common(test_df_r, cols, n=2, window_type='w', gp_col='window_id')
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['channelGrouping','customDimensions','device.browser','device.deviceCategory','device.operatingSystem','geoNetwork.city',
        'geoNetwork.continent','geoNetwork.country','geoNetwork.metro','geoNetwork.networkDomain','geoNetwork.region',
        'geoNetwork.subContinent','trafficSource.adContent','trafficSource.adwordsClickInfo.adNetworkType',
        'trafficSource.adwordsClickInfo.slot','trafficSource.campaign','trafficSource.keyword','trafficSource.medium',
        'trafficSource.referralPath','trafficSource.source','ref_level_1','ref_level_2','ref_level_3','ref_level_4','ref_level_n',
        'src_level_1','src_level_2','src_level_3','src_level_n','visit_year']
train_df_w1 = agg_most_common(train_df_r, cols, n=3, window_type='w', gp_col='window_id')
train_df_w = train_df_w.join(train_df_w1)
del train_df_w1
gc.collect()
test_df_w1 = agg_most_common(test_df_r, cols, n=3, window_type='w', gp_col='window_id')
test_df_w = test_df_w.join(test_df_w1)
del test_df_w1
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['visit_quarter']
train_df_w1 = agg_most_common(train_df_r, cols, n=4, window_type='w', gp_col='window_id')
train_df_w = train_df_w.join(train_df_w1)
del train_df_w1
gc.collect()
test_df_w1 = agg_most_common(test_df_r, cols, n=4, window_type='w', gp_col='window_id')
test_df_w = test_df_w.join(test_df_w1)
del test_df_w1
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['visit_month','visit_day','visit_hour','visit_week','visit_dayofweek','visit_dayofyear']
train_df_w1 = agg_most_common(train_df_r, cols, n=7, window_type='w', gp_col='window_id')
train_df_w = train_df_w.join(train_df_w1)
del train_df_w1
gc.collect()
test_df_w1 = agg_most_common(test_df_r, cols, n=7, window_type='w', gp_col='window_id')
test_df_w = test_df_w.join(test_df_w1)
del test_df_w1
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')
def agg_numeric_cols(_df, gdf, cols, agg_methods, window_type='u', gp_col=None, positive=False):
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    data_types = _df.dtypes
    _df = _df.loc[:, cols + gp_cols].copy()
    if positive:
        for _col in cols:
            _df.loc[_df[_col] <= 0, _col] = np.nan
    
    gp = _df.groupby(gp_cols)
    if 'mean' in agg_methods:
        gdf = gdf.join(gp.mean().fillna(0))
        for _col in cols:
            gdf[_col] = gdf[_col].astype(data_types[_col])
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_mean' for _col in cols})
        gc.collect()
    if 'std' in agg_methods:
        gdf = gdf.join(gp.std().fillna(0).astype(np.float32))
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_std' for _col in cols})
        gc.collect()
    if 'sum' in agg_methods:
        gdf = gdf.join(gp.sum().fillna(0))
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_sum' for _col in cols})
        gc.collect()
    if 'min' in agg_methods:
        gdf = gdf.join(gp.min().fillna(0))
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_min' for _col in cols})
        for _col in cols:
            gdf[f'{_col}_{window_type}_min'] = gdf[f'{_col}_{window_type}_min'].astype(data_types[_col])        
        gc.collect()
    if 'max' in agg_methods:
        gdf = gdf.join(gp.max().fillna(0))
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_max' for _col in cols})                     
        for _col in cols:
            gdf[f'{_col}_{window_type}_max'] = gdf[f'{_col}_{window_type}_max'].astype(data_types[_col])
        gc.collect()
    return gdf
cols = ['max_diff_time','max_price','min_diff_time','min_price','totals.extraRevenue','totals.hits','totals.pageviews',
        'totals.sessionQualityDim','totals.timeOnSite','totals.totalTransactionRevenue','user_g_span','visit_day_pv',
       'visit_hour_pv']
agg_methods = ['mean', 'std', 'min', 'max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=True)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=True)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['visit_delay']
agg_methods = ['mean', 'std', 'min', 'max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['mean_diff_time','mean_price','mean_time','std_diff_time','std_price','max_price_ratio','mean_price_ratio',
        'min_price_ratio','totals.transactions','trafficSource.adwordsClickInfo.page','user_v_pv','user_s_pv','user_s_idle',
        'user_s_span','user_v_idle','user_v_span']
agg_methods = ['mean', 'min', 'max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=True)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=True)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['ref_level_cnt','src_level_cnt','visit_idle']
agg_methods = ['mean', 'min', 'max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['user_w_pv','user_w_idle','user_w_span']
agg_methods = ['max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['visitStartTime']
agg_methods = ['min', 'max']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')

cols = ['target']
agg_methods = ['sum']
train_df_w = agg_numeric_cols(train_df_r, train_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
test_df_w = agg_numeric_cols(test_df_r, test_df_w, cols, agg_methods, window_type='w', gp_col='window_id', positive=False)
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')
def stat_window_cnt(_df, gdf, cols=None, window_type='u', gp_col=None):
    cols = cols if cols else ['visitId','session_id']
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    _df = _df[cols + gp_cols]
    gc.collect()
    gdf = gdf.join(_df.groupby(gp_cols)[cols].nunique().rename(columns={_col: f'{_col}_{window_type}_cnt' for _col in cols}))
    gc.collect()
    gdf[f'visitId_{window_type}_cnt'] = gdf[f'visitId_{window_type}_cnt'].astype(np.int16)
    gdf[f'session_id_{window_type}_cnt'] = gdf[f'session_id_{window_type}_cnt'].astype(np.int8)
    gc.collect()
    return gdf
train_df_w = stat_window_cnt(train_df_r, train_df_w, window_type='w', gp_col='window_id')
gc.collect()
test_df_w = stat_window_cnt(test_df_r, test_df_w, window_type='w', gp_col='window_id')
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')
def concat_by_window(_df, gdf, cols, window_type='u', gp_col=None):
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    _df = _df.loc[:, cols + gp_cols].set_index(gp_cols)
    gc.collect()
    cnt = _df.groupby(gp_cols)[cols[0]].count()

    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[ids1]

    ids2 = cnt.loc[cnt >= 2].index
    df2 = _df.loc[ids2].groupby(gp_cols).agg(lambda rows: ' '.join(rows))
    gdf = gdf.join(df1.append(df2, sort=False).rename(columns={_col: f'{window_type}_{_col}' for _col in cols}))
    
    return gdf
cols = ['exitScreenNames_src','promoNames_src','productNames_src','productCategorys_src','hitProductNames_src','device.browser_src',
        'trafficSource.adContent_src','trafficSource.keyword_src','trafficSource.referralPath_src','trafficSource.source_src']
train_df_w = concat_by_window(train_df_r, train_df_w, cols, window_type='w', gp_col='window_id')
gc.collect()
test_df_w = concat_by_window(test_df_r, test_df_w, cols, window_type='w', gp_col='window_id')
gc.collect()
print(f'train_df_w: {train_df_w.shape}, test_df_w: {test_df_w.shape}')
s = train_df_w.dtypes
ss = list(s.loc[s == np.object].index)
print(len(ss))
print(ss)
ss = list(s.loc[s != np.object].index)
print(len(ss))
print(ss)
df = train_df_w
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
df = test_df_w
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
def expand_most_common(_df, _cols, n=3, window_type='u', gp_col='window_id', sort_col='visitStartTime'):
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    _df = _df.loc[:, _cols + gp_cols + [sort_col]].sort_values(by=sort_col)
    gc.collect()
    cnt = _df[id_col].value_counts()
    
    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[_df[id_col].isin(ids1)].copy()
    for _col in _cols:
        for k in range(1, n):
            df1[f'{_col}_{window_type}_{k}'] = -127
            df1[f'{_col}_{window_type}_{k}'] = df1[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)
            
    def most_common(eles):
        _cnt = Counter()
        commons = []
        for ele in eles:
            _cnt.update([ele])
            cnt_pair = sorted(_cnt.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
            commons.append([cnt_pair[i][0] if len(cnt_pair) > i else -127 for i in range(n)])
        return commons
    
    ids2 = cnt.loc[cnt >= 2].index
    df2 = _df.loc[_df[id_col].isin(ids2)]
    gdf = df2.groupby(id_col)[_cols].transform(most_common)
    gdf = gdf.loc[df2.groupby(gp_cols)[sort_col].idxmax()]
    for _col in _cols:
        for k in range(1, n):
            gdf[f'{_col}_{window_type}_{k}'] = gdf[_col].str.get(k)
            gdf[f'{_col}_{window_type}_{k}'] = gdf[f'{_col}_{window_type}_{k}'].astype(_df[_col].dtype)
        gdf[_col] = gdf[_col].str.get(0)
        gdf[_col] = gdf[_col].astype(_df[_col].dtype)
    for _col in gp_cols:
        gdf[_col] = df2[_col]
        
    gdf = df1.drop(sort_col, axis=1).append(gdf, sort=False).rename(
        columns={_col: f'{_col}_{window_type}_0' for _col in _cols}).set_index(gp_cols)
    gc.collect()
    
    return gdf
cols = ['device.isMobile','totals.bounces','totals.newVisits','trafficSource.adwordsClickInfo.isVideoAd',
        'trafficSource.isTrueDirect','visit_weekend']
train_df_u = expand_most_common(train_df_r, cols, n=2, window_type='u')
gc.collect()
test_df_u = expand_most_common(test_df_r, cols, n=2, window_type='u')
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['channelGrouping','customDimensions','device.browser','device.deviceCategory','device.operatingSystem','geoNetwork.city',
        'geoNetwork.continent','geoNetwork.country','geoNetwork.metro','geoNetwork.networkDomain','geoNetwork.region',
        'geoNetwork.subContinent','trafficSource.adContent','trafficSource.adwordsClickInfo.adNetworkType',
        'trafficSource.adwordsClickInfo.slot','trafficSource.campaign','trafficSource.keyword','trafficSource.medium',
        'trafficSource.referralPath','trafficSource.source','ref_level_1','ref_level_2','ref_level_3','ref_level_4','ref_level_n',
        'src_level_1','src_level_2','src_level_3','src_level_n','visit_year']
train_df_u1 = expand_most_common(train_df_r, cols, n=3, window_type='u')
train_df_u = train_df_u.join(train_df_u1)
del train_df_u1
gc.collect()
test_df_u1 = expand_most_common(test_df_r, cols, n=3, window_type='u')
test_df_u = test_df_u.join(test_df_u1)
del test_df_u1
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['visit_quarter']
train_df_u1 = expand_most_common(train_df_r, cols, n=4, window_type='u')
train_df_u = train_df_u.join(train_df_u1)
del train_df_u1
gc.collect()
test_df_u1 = expand_most_common(test_df_r, cols, n=4, window_type='u')
test_df_u = test_df_u.join(test_df_u1)
del test_df_u1
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['visit_month','visit_day','visit_hour','visit_week','visit_dayofweek','visit_dayofyear']
train_df_u1 = expand_most_common(train_df_r, cols, n=7, window_type='u')
train_df_u = train_df_u.join(train_df_u1)
del train_df_u1
gc.collect()
test_df_u1 = expand_most_common(test_df_r, cols, n=7, window_type='u')
test_df_u = test_df_u.join(test_df_u1)
del test_df_u1
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
def expand_numeric_cols(_df, _gdf, cols, expand_methods, window_type='u', gp_col='window_id', positive=False, 
                        sort_col='visitStartTime'):
    gp_cols = [id_col]
    if gp_col is not None:
        gp_cols.append(gp_col)
    
    data_types = _df.dtypes
    cnt = _df[id_col].value_counts()
    
    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[_df[id_col].isin(ids1)]
    gdf1 = pd.DataFrame()
    for _col in gp_cols:
        gdf1[_col] = df1[_col]
    for _col in cols:
        column = df1[_col]
        for agg_method in expand_methods:
            if 'std' != agg_method:
                gdf1[f'{_col}_{window_type}_{agg_method}'] = column
            else:
                gdf1[f'{_col}_{window_type}_std'] = 0
                gdf1[f'{_col}_{window_type}_std'] = gdf1[f'{_col}_{window_type}_std'].astype(np.float32)
    gdf1 = gdf1.set_index(gp_cols)
    del df1
    gc.collect()
    
    ids2 = cnt.loc[cnt > 1].index
    _df = _df.loc[_df[id_col].isin(ids2)]
    _df = _df.loc[:, cols + gp_cols + ([] if sort_col in cols else [sort_col])].sort_values(by=sort_col).copy()
    if positive:
        for _col in cols:
            _df.loc[_df[_col] <= 0, _col] = np.nan
    gc.collect()
    
    gp = _df.groupby(id_col)[cols].expanding()
    gdf = None
    if 'mean' in expand_methods:
        gdf = gp.mean().fillna(0)
        for _col in cols:
            gdf[_col] = gdf[_col].astype(data_types[_col])
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_mean' for _col in cols})
        gc.collect()
    if 'std' in expand_methods:
        cur_gdf = gp.std().fillna(0).astype(np.float32)
        gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_std' for _col in cols})
        gc.collect()
    if 'sum' in expand_methods:
        cur_gdf = gp.sum().fillna(0)
        gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_sum' for _col in cols})
        gc.collect()
    if 'min' in expand_methods:
        cur_gdf = gp.min().fillna(0)
        gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_min' for _col in cols})
        for _col in cols:
            gdf[f'{_col}_{window_type}_min'] = gdf[f'{_col}_{window_type}_min'].astype(data_types[_col])        
        gc.collect()
    if 'max' in expand_methods:
        cur_gdf = gp.max().fillna(0)
        gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
        gdf = gdf.rename(columns={_col: f'{_col}_{window_type}_max' for _col in cols})                     
        for _col in cols:
            gdf[f'{_col}_{window_type}_max'] = gdf[f'{_col}_{window_type}_max'].astype(data_types[_col])
        gc.collect()
        
    gdf.index=gdf.index.droplevel()
    gdf = gdf.loc[_df.groupby(gp_cols)[sort_col].idxmax()]
    for _col in gp_cols:
        gdf[_col] = _df[_col]
        
    gdf = gdf1.append(gdf.set_index(gp_cols), sort=False)        
    _gdf = _gdf.join(gdf)
    del _df,gdf1,gdf,gp
    gc.collect()
    
    return _gdf
cols = ['max_diff_time','max_price','min_diff_time','min_price','totals.extraRevenue','totals.hits','totals.pageviews',
        'totals.sessionQualityDim','totals.timeOnSite','totals.totalTransactionRevenue','user_g_span','visit_day_pv',
       'visit_hour_pv']
agg_methods = ['mean', 'std', 'min', 'max']
train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, agg_methods, positive=True)
gc.collect()
test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, agg_methods, positive=True)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['visit_delay']
agg_methods = ['mean', 'std', 'min', 'max']
train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, agg_methods, positive=False)
gc.collect()
test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, agg_methods, positive=False)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['mean_diff_time','mean_price','mean_time','std_diff_time','std_price','max_price_ratio','mean_price_ratio',
        'min_price_ratio','totals.transactions','trafficSource.adwordsClickInfo.page','user_v_pv','user_s_pv','user_s_idle',
        'user_s_span','user_v_idle','user_v_span']
agg_methods = ['mean', 'min', 'max']
train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, agg_methods, positive=True)
gc.collect()
test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, agg_methods, positive=True)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['ref_level_cnt','src_level_cnt','visit_idle']
agg_methods = ['mean', 'min', 'max']
train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, agg_methods, positive=False)
gc.collect()
test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, agg_methods, positive=False)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

cols = ['visitStartTime']
agg_methods = ['min', 'max']
train_df_u = expand_numeric_cols(train_df_r, train_df_u, cols, agg_methods, positive=False)
gc.collect()
test_df_u = expand_numeric_cols(test_df_r, test_df_u, cols, agg_methods, positive=False)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
def expand_target_cols(_df, _gdf, expand_methods, t_col='target_w_sum', window_type='u', sort_col='visitStartTime_w_min',
                       positive=False):
    cnt = _df.groupby(id_col)[sort_col].count()
    
    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[ids1, [t_col]].copy()
    column = df1[t_col]
    for agg_method in expand_methods:
        if 'std' != agg_method:
            df1[f'target_{window_type}_{agg_method}'] = column
        else:
            df1[f'target_{window_type}_std'] = 0
            df1[f'target_{window_type}_std'] = df1[f'target_{window_type}_std'].astype(np.int64)
    df1 = df1.drop(t_col, axis=1)
    gc.collect()
    
    ids2 = cnt.loc[cnt > 1].index
    _df = _df.loc[ids2, [t_col, sort_col]].sort_values(by=sort_col).copy()
    if positive:
        _df.loc[_df[t_col] <= 0, t_col] = np.nan
    gc.collect()

    gp = _df.groupby(id_col)[[t_col]].expanding()
    gdf = None
    for agg_method in expand_methods:
        cur_gdf = gp.agg(agg_method).fillna(0).astype(np.int64).rename(
            columns={t_col: f'target_{window_type}_{agg_method}'})
        gdf = gdf.join(cur_gdf) if gdf is not None else cur_gdf
        gc.collect()

    gdf.index=gdf.index.droplevel()
    gdf = df1.append(gdf, sort=False)
    _gdf = _gdf.join(gdf)
    del df1,_df,cur_gdf,gdf,gp
    gc.collect()
    
    return _gdf
agg_method = ['mean', 'std', 'sum', 'min', 'max']
train_df_u = expand_target_cols(train_df_w, train_df_u, agg_method, positive=True)
gc.collect()
test_df_u = expand_target_cols(test_df_w, test_df_u, agg_method, positive=True)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
def expand_window_cnt(_df, gdf, cols=None, window_type='u', sort_col='visitStartTime_w_min'):
    cols = cols if cols else ['visitId_w_cnt','session_id_w_cnt']
    
    _df = _df.sort_values(by=sort_col)
    gp = _df.groupby(id_col)
    
    cur_df = gp[sort_col].cumcount().astype(np.uint8)
    cur_df.name = f'window_id_{window_type}_cnt'
    gdf = gdf.join(cur_df)
    gc.collect()
    
    gdf = gdf.join(gp[cols].cumsum().rename(
        columns={_col: _col.replace('_w_', f'_{window_type}_') for _col in cols}))
    del _df,gp,cur_df
    gc.collect()
    
    return gdf
train_df_u = expand_window_cnt(train_df_w, train_df_u)
gc.collect()
test_df_u = expand_window_cnt(test_df_w, test_df_u)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')

train_df_u['user_u_span'] = train_df_u['visitStartTime_u_max'] - train_df_u['visitStartTime_u_min']
test_df_u['user_u_span'] = test_df_u['visitStartTime_u_max'] - test_df_u['visitStartTime_u_min']
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
def expanded_concat_by_window(_df, _gdf, cols, window_type='u', sort_col='visitStartTime_w_min'):
    cnt = _df.groupby(id_col)[sort_col].count()
    
    ids1 = cnt.loc[1 == cnt].index
    df1 = _df.loc[ids1, cols].rename(columns={_col: _col.replace('w_', f'{window_type}_') for _col in cols})
    gc.collect()
    
    def expanded_concat(eles):
        infos = []
        info = ''
        for ele in eles:
            info += ele + ' '
            infos.append(info)
        return infos
    
    ids2 = cnt.loc[cnt > 1].index
    _df = _df.loc[ids2].sort_values(by=sort_col)
    gc.collect()
    gdf = _df.groupby(id_col)[cols].transform(expanded_concat).rename(
        columns={_col: _col.replace('w_', f'{window_type}_') for _col in cols})
    gc.collect()
    gdf = df1.append(gdf, sort=False)
    _gdf = _gdf.join(gdf) if _gdf is not None else gdf
    del _df,df1,gdf
    gc.collect()
    
    return _gdf
cols = ['w_productNames_src','w_productCategorys_src','w_hitProductNames_src']
train_df_u = expanded_concat_by_window(train_df_w, train_df_u, cols)
gc.collect()
test_df_u = expanded_concat_by_window(test_df_w, test_df_u, cols)
gc.collect()
print(f'train_df_u: {train_df_u.shape}, test_df_u: {test_df_u.shape}')
s = train_df_u.dtypes
ss = list(s.loc[s == np.object].index)
print(len(ss))
print(ss)
ss = list(s.loc[s != np.object].index)
print(len(ss))
print(ss)
df = train_df_u
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
df = test_df_u
for col in ss:
    print(f'{col}:\t\t{np.sum(df[col].isnull())}\t{df[col].dtype.name}')
df.info()
train_df_r.to_pickle('train_df_r', compression='gzip')
del train_df_r
gc.collect()
test_df_r.to_pickle('test_df_r', compression='gzip')
del test_df_r
gc.collect()
train_df_w.to_pickle('train_df_w', compression='gzip')
del train_df_w
gc.collect()
test_df_w.to_pickle('test_df_w', compression='gzip')
del test_df_w
gc.collect()
train_df_u.to_pickle('train_df_u', compression='gzip')
del train_df_u
gc.collect()
test_df_u.to_pickle('test_df_u', compression='gzip')
del test_df_u
gc.collect()
 