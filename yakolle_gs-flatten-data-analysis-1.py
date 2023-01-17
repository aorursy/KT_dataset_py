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
from functools import reduce

pd.options.display.max_rows = 8
pd.options.display.max_columns = 999
print(os.listdir("../input"))
id_col = 'fullVisitorId'
cols = [id_col, 'visitStartTime']

test_df = pd.read_csv('../input/test_v2.csv', dtype={id_col: 'str'}).loc[:, cols].copy()
print(test_df.shape)
sub_df = pd.read_csv('../input/sample_submission_v2.csv', dtype={id_col: 'str'})
print(sub_df.shape)
s = test_df[id_col].unique()
sub_ids = sub_df[id_col].unique()
print(s.shape, sub_ids.shape)
sub_ids = np.intersect1d(s, sub_ids)
del sub_df, s
gc.collect()
print(sub_ids.shape)
def read_cols(file_tag, cols):
    data_reader = pd.read_csv(f'../input/{file_tag}_v2.csv', iterator=True, chunksize=20000,
                          dtype={id_col: 'str'})
    df = pd.DataFrame()
    for it in data_reader:
        df = df.append(it[cols].copy(), ignore_index=True)
        gc.collect()
    del data_reader, it
    gc.collect()
    
    return df


train_df = read_cols('train', cols)
gc.collect()
print(train_df.shape)
cids = np.intersect1d(train_df[id_col], test_df[id_col])
print(f'cids: {cids.shape}')

df = train_df.append(test_df, ignore_index=True)
gc.collect()
print(df.shape)
cnt = df[id_col].value_counts()
print(cnt.shape)
tr_ids = cnt.loc[cnt>1].index.values
print(f'tr_ids: {tr_ids.shape}')
tr_df = df.loc[df[id_col].isin(tr_ids)]
gc.collect()
print(tr_df.shape)
tm_col = 'visitStartTime'
print(df[tm_col].nunique())

time_frame_start1 = pd.to_datetime('2017-05-01').timestamp()
time_frame_end1 = pd.to_datetime('2017-10-01').timestamp()
time_frame_start2 = pd.to_datetime('2018-01-01').timestamp()
time_frame_end2 = pd.to_datetime('2018-06-01').timestamp()

tr_ids1 = cnt.loc[1==cnt].index.values
print(tr_ids1.shape)

tr_df1 = df.loc[df[id_col].isin(tr_ids1) & (df[tm_col]>=time_frame_start1) & (df[tm_col]<time_frame_end1)]
tr_df2 = df.loc[df[id_col].isin(tr_ids1) & (df[tm_col]>=time_frame_start2) & (df[tm_col]<time_frame_end2)]
print(tr_df1.shape, tr_df2.shape)
tr_ids1 = tr_df1[id_col].unique()
tr_ids2 = tr_df2[id_col].unique()
tr_ids = reduce(np.union1d, (tr_ids, tr_ids1, tr_ids2))
print(f'tr_ids: {tr_ids.shape}')
df = df.loc[df[id_col].isin(tr_ids)]
gc.collect()
print(df.shape)
del df,train_df,test_df,cnt,tr_ids1,tr_ids2,tr_df1,tr_df2
gc.collect()
def extract_hits(df):
    def _extract_hits(rec):
        target = rec['target']
        info = re.sub(r"\\?['\"]", "'", rec['hits'])

        meta_infos = re.findall(r"'time'\:\s*'(\d+)'.*?exitScreenName.*?/([^/]+?)',", info)
        times,exitScreenNames = zip(*meta_infos) if meta_infos else ([],[])
        times = [int(t) for t in times] if times else []
        exitScreenNames = ' '.join(set(exitScreenNames)) if exitScreenNames else 'None'

        promoNames = set(re.findall(r"promoName'\:\s*'(.+?)',", info))
        promoNames = ' '.join(promoNames) if promoNames else 'None'

        products = re.findall("v2ProductName'\:\s*'(.+?)',.*?v2ProductCategory'\:\s*'(.+?)',.*?'productPrice'\:\s*'(\d+)',", info)
        productNames,productCategorys,productPrices = zip(*products) if products else ([],[],[])
        productPrices = [int(price) for price in productPrices] if productPrices else []
        products = dict(zip(productNames, productPrices))
        productNames = ' '.join(products.keys()) if products else 'None'
        productPrices = list(products.values())
        productCategorys = ' '.join(set(productCategorys)) if products else 'None'
        hitProductNames = [name for name,price in products.items() if price==target] if target>0 and products else []
        hitProductNames = ' '.join(hitProductNames) if hitProductNames else 'None'

        hit_cnt = len(times)
        total_time = np.max(times) if times else 0
        mean_time = total_time//(hit_cnt-1) if hit_cnt>1 else 0
        diff_times = np.diff(times)
        mean_diff_time,min_diff_time,max_diff_time,std_diff_time = 0,0,0,0
        if diff_times.shape[0]>0:
            mean_diff_time = int(np.mean(diff_times))
            min_diff_time = np.min(diff_times)
            max_diff_time = np.max(diff_times)
            std_diff_time = int(np.std(diff_times))

        mean_price = int(np.mean(productPrices)) if productPrices else 0
        min_price = np.min(productPrices) if productPrices else 0
        max_price = np.max(productPrices) if productPrices else 0
        std_price = int(np.std(productPrices)) if productPrices else 0
        mean_price_ratio = target/mean_price if mean_price else -1.0
        min_price_ratio = target/min_price if min_price else -1.0
        max_price_ratio = target/max_price if max_price else -1.0

        return (total_time,mean_time,mean_diff_time,min_diff_time,max_diff_time,std_diff_time,exitScreenNames,promoNames,
                productNames,productCategorys,mean_price,min_price,max_price,std_price,mean_price_ratio,min_price_ratio,
                max_price_ratio,hitProductNames)    
    
    (df['total_time'],df['mean_time'],df['mean_diff_time'],df['min_diff_time'],df['max_diff_time'],df['std_diff_time'],
     df['exitScreenNames_src'],df['promoNames_src'],df['productNames_src'],df['productCategorys_src'],df['mean_price'],
     df['min_price'],df['max_price'],df['std_price'],df['mean_price_ratio'],df['min_price_ratio'],df['max_price_ratio'],
     df['hitProductNames_src']) = zip(*df[['hits','target']].apply(_extract_hits, axis=1))
    
    return df.drop('hits', axis=1)
def load_df(ids, data_tag='train'):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    data_reader = pd.read_csv(f'../input/{data_tag}_v2.csv', iterator=True, chunksize=100000,
                              converters={column: json.loads for column in JSON_COLUMNS},
                              dtype={id_col: 'str'})
    df = pd.DataFrame()
    for data in data_reader:
        data = data.loc[data[id_col].isin(ids)].reset_index(drop=True).copy()
        if data.shape[0] > 0:
            print(f'before: {data.shape}', end='\t')
            for column in JSON_COLUMNS:
                column_as_df = json_normalize(data[column])
                column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
                data = data.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
                del column_as_df
                gc.collect()
            
            data = data.rename(columns={'totals.transactionRevenue':'target'})
            data['target'] = data['target'].fillna(0) if 'target' in data.columns else 0
            data.target = data.target.astype(np.int64)
            gc.collect()
            data = extract_hits(data)
            gc.collect()
            
            df = df.append(data, ignore_index=True, sort=False)
            print(f'after: {data.shape}, total: {df.shape}')
            del data
            gc.collect()
    return df
train_df_r = load_df(tr_ids)
print(f'train_df_r: {train_df_r.shape}')
gc.collect()
test_df = load_df(tr_ids, 'test')
print(f'test_df: {test_df.shape}')
gc.collect()
train_df_r = train_df_r.append(test_df, ignore_index=True, sort=False)
del test_df
gc.collect()
print(f'train_df_r: {train_df_r.shape}')
def find_still_cols(df):
    still_cols = []
    for col in df.columns:
        cnt = df[col].nunique(dropna=False)
        if cnt <= 1:
            still_cols.append(col)
    return still_cols
still_cols = ['trafficSource.adwordsClickInfo.gclId', 'trafficSource.campaignCode'] + find_still_cols(train_df_r)
print(f'still_cols(len(still_cols)): {still_cols}')
train_df_r = train_df_r.drop(still_cols, axis=1)
gc.collect()
print(f'train_df_r: {train_df_r.shape}')
def encode_user_pv(_df, gp_col=id_col, _col_prefix='user', sort_col='visitStartTime', 
                   session_interval=7*24*3600, window_interval=60*24*3600):
    # global scope
    _df = _df.sort_values(by=[gp_col, sort_col])
    gp = _df.groupby(gp_col)[sort_col]
    _df[f'{_col_prefix}_g_span'] = gp.diff().fillna(0)
    _df = _df.join(gp.count(), on=gp_col, rsuffix='_g_pv').rename(
        columns={f'{sort_col}_g_pv': f'{_col_prefix}_g_pv'})
    
    # window scope
    _df = _df.join(gp.max(), on=gp_col, rsuffix='_rm').rename(columns={f'{sort_col}_rm': 'window_id'})
    _df['window_id'] = ((_df['window_id'] - _df[sort_col]) // window_interval).astype(np.uint8)
    gp_cols = [gp_col, 'window_id']
    gp = _df.groupby(gp_cols)
    _df = _df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_w_pv').rename(
        columns={f'{sort_col}_w_pv': f'{_col_prefix}_w_pv'})
    _df = _df.join(gp[sort_col].max()-gp[sort_col].min(), on=gp_cols, rsuffix='_w_span').rename(
        columns={f'{sort_col}_w_span': f'{_col_prefix}_w_span'})
    _df = _df.join(gp[f'{_col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_w_idle').rename(
        columns={f'{_col_prefix}_g_span_w_idle': f'{_col_prefix}_w_idle'})
    
    # session scope
    _df['session_id'] = _df[f'{_col_prefix}_g_span']
    _df.loc[_df['session_id']<=session_interval, 'session_id'] = 0
    _df.loc[_df['session_id']>session_interval, 'session_id'] = 1
    _df['session_id'] = _df.groupby(gp_col)['session_id'].cumsum().astype(np.uint16)
    gp_cols = [gp_col, 'session_id']
    gp = _df.groupby(gp_cols)
    _df = _df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_s_pv').rename(
        columns={f'{sort_col}_s_pv': f'{_col_prefix}_s_pv'}) 
    _df = _df.join(gp[sort_col].max()-gp[sort_col].min(), on=gp_cols, rsuffix='_s_span').rename(
        columns={f'{sort_col}_s_span': f'{_col_prefix}_s_span'})
    _df = _df.join(gp[f'{_col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_s_idle').rename(
        columns={f'{_col_prefix}_g_span_s_idle': f'{_col_prefix}_s_idle'})
    
    # visit scope
    gp_cols = [gp_col, 'visitId']
    gp = _df.groupby(gp_cols)
    _df = _df.join(gp[sort_col].count(), on=gp_cols, rsuffix='_v_pv').rename(
        columns={f'{sort_col}_v_pv': f'{_col_prefix}_v_pv'}) 
    _df = _df.join(gp[sort_col].max()-gp[sort_col].min(), on=gp_cols, rsuffix='_v_span').rename(
        columns={f'{sort_col}_v_span': f'{_col_prefix}_v_span'})
    _df = _df.join(gp[f'{_col_prefix}_g_span'].first(), on=gp_cols, rsuffix='_v_idle').rename(
        columns={f'{_col_prefix}_g_span_v_idle': f'{_col_prefix}_v_idle'})
    
    for flag in list('gwsv'):
        _df[f'{_col_prefix}_{flag}_pv'] = _df[f'{_col_prefix}_{flag}_pv'].astype(np.int16)
        _df[f'{_col_prefix}_{flag}_span'] = _df[f'{_col_prefix}_{flag}_span'].astype(np.int32)
    for flag in list('wsv'):
        _df[f'{_col_prefix}_{flag}_idle'] = _df[f'{_col_prefix}_{flag}_idle'].astype(np.int32)
    _df = _df.sort_index()

    return _df
train_df_r = encode_user_pv(train_df_r)
gc.collect()
print(f'train_df_r: {train_df_r.shape}')
print(list(train_df_r['device.browser'].unique()))
print(list(train_df_r['trafficSource.adContent'].unique()))
def flatten_counter(cnts):
    pairs = sorted(cnts.items(), key=lambda pair: (pair[1], pair[0]))
    last_cnt = 0
    new_pairs = []
    for k, v in pairs:
        if v > last_cnt:
            new_pairs.append((k, v))
            last_cnt = v
        else:
            last_cnt += 1
            new_pairs.append((k, last_cnt))
    return dict(new_pairs)


class CountEncoder:
    def __init__(self, nan_value=-127, count_unique=True):
        self.counter = None
        self.nan_value = nan_value
        self.unseen_code_value = nan_value
        self.count_unique = count_unique

    def fit(self, x, need_fill_na=True):
        x = x.fillna(self.nan_value) if need_fill_na else x
        self.counter = Counter(x)
        unseen_code = None
        if not x.loc[x == self.nan_value].shape[0]:
            unseen_code = sorted(self.counter.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)[0][0]
        self.counter = flatten_counter(self.counter) if self.count_unique else self.counter
        if unseen_code is not None:
            self.unseen_code_value = self.counter[unseen_code]

    def transform(self, x, need_fill_na=True):
        x = x.fillna(self.nan_value) if need_fill_na else x
        return x.apply(lambda ele: self.unseen_code_value if (
            ele == self.nan_value or ele not in self.counter) else self.counter[ele])

    def fit_transform(self, x):
        x = x.fillna(self.nan_value)
        self.fit(x, need_fill_na=False)
        return self.transform(x, need_fill_na=False)
    

def get_int_type(col):
    max_val = col.max()
    if max_val < 2 ** 7:
        return np.int8
    elif max_val < 2 ** 15:
        return np.int16
    elif max_val < 2 ** 31:
        return np.int32
    else:
        return np.int64


def encode_obj(df, cols=None, encoders=None, data_types=None, count_unique=True):
    if not encoders:
        encoders = {}
        data_types = {}
        for col in np.intersect1d(df.columns, cols):
            cer = CountEncoder(count_unique=count_unique)
            df[col] = cer.fit_transform(df[col])
            encoders[col] = cer
            data_type = get_int_type(df[col])
            data_types[col] = data_type
            df[col] = df[col].astype(data_type)
    else:
        for col in np.intersect1d(df.columns, cols):
            df[col] = encoders[col].transform(df[col]).astype(data_types[col])

    return df, encoders, data_types
obj_cols = ['channelGrouping', 'customDimensions', 'device.browser', 'device.deviceCategory', 'device.isMobile', 
            'device.operatingSystem', 'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
            'geoNetwork.networkDomain', 'geoNetwork.region', 'geoNetwork.subContinent', 'totals.bounces',
            'totals.newVisits', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
            'trafficSource.isTrueDirect', 'trafficSource.keyword', 'trafficSource.medium', 'trafficSource.referralPath',
            'trafficSource.source']

col = 'trafficSource.keyword'
train_df_r[col] = train_df_r[col].str.lower()
for col in ['device.browser','trafficSource.adContent','trafficSource.keyword','trafficSource.referralPath','trafficSource.source']:
    train_df_r[f'{col}_src'] = train_df_r[col].fillna('unknown')

train_df_r, cat_encoders, cat_data_types = encode_obj(train_df_r, obj_cols)
gc.collect()
def convert_data_type(df):
    int8_cols = ['totals.sessionQualityDim','totals.transactions','trafficSource.adwordsClickInfo.page']
    for col in int8_cols:
        df[col] = df[col].fillna(-127).astype(np.int8)

    col = 'totals.pageviews'
    df[col] = df[col].fillna(1)
    int16_cols = ['visitNumber','totals.pageviews','totals.hits']              
    for col in int16_cols:
        df[col] = df[col].fillna(-127).astype(np.int16)

    df.loc[df['totals.timeOnSite'].isnull() & (df['total_time']>0), 'totals.timeOnSite'] = np.round(
        df.loc[df['totals.timeOnSite'].isnull() & (df['total_time']>0), 'total_time']/1000)
    df = df.drop('total_time', axis=1)
    gc.collect()
    int32_cols = ['visitId','visitStartTime','totals.timeOnSite','mean_time','mean_diff_time','min_diff_time',
                  'max_diff_time','std_diff_time']
    for col in int32_cols:
        df[col] = df[col].fillna(-127).astype(np.int32)

    float32_cols = ['mean_price_ratio','min_price_ratio','max_price_ratio']
    for col in float32_cols:
        df[col] = df[col].fillna(-127).astype(np.float32)

    col = 'totals.totalTransactionRevenue'
    df[col] = df[col].fillna(0).astype(np.int64)
    df['totals.extraRevenue'] = df[col] - df['target']
    
    return df
train_df_r = convert_data_type(train_df_r)
gc.collect()
print(f'train_df_r: {train_df_r.shape}')
rec_num = 20
col = 'trafficSource.keyword_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode', min_df=2)
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
tvr = TfidfVectorizer(strip_accents='unicode',analyzer='char',ngram_range=(3,5), max_df=0.8, min_df=5)
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 20
col = 'trafficSource.referralPath_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode', min_df=5)
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
col = 'trafficSource.referralPath_src'
s = train_df_r[col].str.count('/')
print(s.loc[s<=0].shape)
print(s.quantile([0.7,0.8,0.9,0.95]))


def encode_ref_path(df, ref_encoders=None, ref_data_types=None):
    col = 'trafficSource.referralPath_src'
    
    df['ref_level_cnt'] = df[col].str.count('/').astype(np.int8)
    s = df[col].str.split('/')
    df['ref_level_1'] = '/' + s.str.get(1)
    for i in range(2, 5):
        df[f'ref_level_{i}'] = df[f'ref_level_{i-1}'] + '/' + s.str.get(i)
    df['ref_level_n'] = s.str.get(-1)
    
    cols=['ref_level_n'] + [f'ref_level_{i}' for i in range(1,5)]
    if not ref_encoders:
        df, ref_encoders, ref_data_types = encode_obj(df, cols)
    else:
        df, _, _ = encode_obj(df, cols, ref_encoders, ref_data_types)
    
    return df, ref_encoders, ref_data_types
train_df_r, ref_encoders, ref_data_types = encode_ref_path(train_df_r)
rec_num = 20
col = 'trafficSource.source_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
s = train_df_r['trafficSource.source_src']
ss = np.sort(s.unique())
print(list(np.append(ss[:rec_num],ss[-rec_num:])))

s = s.str.replace(r':\d+\s*$', '')
ss = np.sort(s.unique())
print(list(np.append(ss[:rec_num],ss[-rec_num:])))
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(s)
tokens = tvr.get_feature_names()
print(len(tokens))
col = 'trafficSource.source_src'
s = train_df_r[col].str.count('\.')
print(s.loc[s<=0].shape)
print(s.quantile([0.5,0.6,0.7,0.8,0.9,0.95,1]))


def encode_source(df, src_encoders=None, src_data_types=None):
    col = 'trafficSource.source_src'
    df[col] = df[col].str.replace(r':\d+\s*$', '')
    
    df['src_level_cnt'] = df[col].str.count('\.').astype(np.int8) + 1
    s = df[col].str.split('\.')
    for i in range(1, 4):
        df[f'src_level_{i}'] = s.str.get(-i)
    df['src_level_n'] = s.str.get(0)
    
    cols=['src_level_n'] + [f'src_level_{i}' for i in range(1,4)]
    if not src_encoders:
        df, src_encoders, src_data_types = encode_obj(df, cols)
    else:
        df, _, _ = encode_obj(df, cols, src_encoders, src_data_types)
    
    return df, src_encoders, src_data_types
train_df_r, src_encoders, src_data_types = encode_source(train_df_r)
rec_num = 20
col = 'device.browser_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 20
col = 'trafficSource.adContent_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 10
col = 'exitScreenNames_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 10
col = 'promoNames_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 2
col = 'productNames_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 10
col = 'productCategorys_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
rec_num = 10
col = 'hitProductNames_src'
s = train_df_r[col]
cnt = s.value_counts(dropna=False)
print(len(cnt), cnt.nunique())
print(cnt.head(rec_num).append(cnt.tail(rec_num)).to_dict())
tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode')
tr_keyword = tvr.fit_transform(train_df_r[col])
tokens = tvr.get_feature_names()
print(len(tokens))
def encode_dates(df, date_encoders=None, date_data_types=None):
    s = pd.to_datetime(df.date.astype(np.str))
    s1 = pd.to_datetime(df.visitStartTime, unit='s')

    df['visit_idle'] = (s1 - s).dt.days.astype(np.int8)
    df['visit_year'] = (2018 - s1.dt.year).astype(np.int8)
    df['visit_month'] = s1.dt.month.astype(np.int8)
    df['visit_day'] = s1.dt.day.astype(np.int8)
    df['visit_hour'] = s1.dt.hour.astype(np.int8)
    df['visit_week'] = s1.dt.week.astype(np.int8)
    df['visit_dayofweek'] = s1.dt.dayofweek.astype(np.int8)
    df['visit_dayofyear'] = s1.dt.dayofyear.astype(np.int16)
    df['visit_quarter'] = s1.dt.quarter.astype(np.int8)
    df['visit_weekend'] = df['visit_dayofweek'].isin([5,6]).astype(np.int8)
    
    df['visit_hour_pv'] = df.visitStartTime // 3600
    df['visit_day_pv'] = df.visit_hour_pv // 24
    df,date_encoders,date_data_types = encode_obj(df, [f'visit_{tag}_pv' for tag in ['hour', 'day']], 
                                                  date_encoders, date_data_types, count_unique=False)
    
    return df.drop('date', axis=1),date_encoders,date_data_types
train_df_r,date_encoders,date_data_types = encode_dates(train_df_r)
gc.collect()
print(f'train_df_r: {train_df_r.shape}')
s1 = train_df_r.visitId
s2 = train_df_r.visitStartTime
ss = s2 - s1
print(ss.loc[ss>0].shape)
print(ss.quantile([0.994,0.995,0.996,0.997,0.998,0.999]))
print(ss.loc[ss>1200].shape, ss.loc[ss>1800].shape, ss.loc[ss>3600].shape, list(ss.loc[ss>3600]))

train_df_r['visit_delay'] = (train_df_r.visitStartTime - train_df_r.visitId).astype(np.int16)
gc.collect()
train_df_r.to_pickle('train_df_r', compression='gzip')
del train_df_r
gc.collect()
tr_df = load_df(sub_ids)
print(f'tr_df: {tr_df.shape}')
gc.collect()
test_df_r = load_df(sub_ids, 'test')
print(f'test_df_r: {test_df_r.shape}')
gc.collect()
test_df_r = tr_df.append(test_df_r, ignore_index=True, sort=False)
del tr_df
gc.collect()
print(f'test_df_r: {test_df_r.shape}')
test_df_r = test_df_r.drop(still_cols, axis=1, errors='ignore')
gc.collect()
print(f'test_df_r: {test_df_r.shape}')
test_df_r = encode_user_pv(test_df_r)
gc.collect()
print(f'test_df_r: {test_df_r.shape}')
col = 'trafficSource.keyword'
test_df_r[col] = test_df_r[col].str.lower()
for col in ['device.browser','trafficSource.adContent','trafficSource.keyword','trafficSource.referralPath','trafficSource.source']:
    test_df_r[f'{col}_src'] = test_df_r[col].fillna('unknown')

test_df_r, cat_encoders, cat_data_types = encode_obj(test_df_r, obj_cols, cat_encoders, cat_data_types)
gc.collect()
test_df_r = convert_data_type(test_df_r)
gc.collect()
print(f'test_df_r: {test_df_r.shape}')

test_df_r, ref_encoders, ref_data_types = encode_ref_path(test_df_r, ref_encoders, ref_data_types)
gc.collect()
test_df_r, src_encoders, src_data_types = encode_source(test_df_r, src_encoders, src_data_types)
gc.collect()
print(f'test_df_r: {test_df_r.shape}')

test_df_r, date_encoders, date_data_types = encode_dates(test_df_r, date_encoders, date_data_types)
gc.collect()
test_df_r['visit_delay'] = (test_df_r.visitStartTime - test_df_r.visitId).astype(np.int16)
gc.collect()
print(f'test_df_r: {test_df_r.shape}')
test_df_r.to_pickle('test_df_r', compression='gzip')
del test_df_r
gc.collect()
train_df_r = pd.read_pickle('train_df_r', compression='gzip')
test_df_r = pd.read_pickle('test_df_r', compression='gzip')
print(f'train_df_r: {train_df_r.shape}, test_df_r: {test_df_r.shape}')
cnt1 = train_df_r.visit_week.value_counts()
print(len(cnt1), cnt1.to_dict())
cnt2 = test_df_r.visit_week.value_counts()
print(len(cnt2), cnt2.to_dict())

x1 = cnt1.sort_index().index.values
y1 = cnt1.sort_index().values
x2 = cnt2.sort_index().index.values
y2 = cnt2.sort_index().values
plt.figure(figsize=(12,6))
plt.plot(x1, y1)
plt.plot(x2, y2)
plt.show()
s = train_df_r.dtypes
ss = list(s.loc[s != np.object].index)
print(len(ss))
print(ss)
ss = list(s.loc[s == np.object].index)
print(len(ss))
print(ss)
train_df_r.info()
test_df_r.info()
 