# !git clone https://github.com/motefly/DeepGBM.git
import gc, collections, os

import numpy as np

import pandas as pd

import category_encoders as ce

from tqdm import tqdm

from pathlib import Path



dataset_path = Path("../input/zillow-prize-1/")
output_path = Path("./out")

output_tr_path = output_path / "train"

output_te_path = output_path / "test"



if not output_tr_path.exists():

    output_tr_path.mkdir(parents=True)



if not output_te_path.exists():

    output_te_path.mkdir(parents=True)
%ls ../input/zillow-prize-1
# reference: https://www.kaggle.com/aharless/xgboost-lightgbm-and-ols

prop = pd.read_csv(dataset_path / 'properties_2016.csv')

train = pd.read_csv(dataset_path / "train_2016_v2.csv")
print( "\nProcessing data for LightGBM ..." )

for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)
df_train = train.merge(prop, how='left', on='parcelid')
nume_col = ['bathroomcnt','bedroomcnt','calculatedbathnbr','threequarterbathnbr','finishedfloor1squarefeet','calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet','numberofstories','poolcnt','poolsizesum','roomcnt','unitcnt','yardbuildingsqft17','yardbuildingsqft17','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','taxdelinquencyyear','yearbuilt']

cate_col = ['architecturalstyletypeid', 'yearbuilt_cate', 'buildingqualitytypeid', 'propertyzoningdesc', 'regionidneighborhood', 'yardbuildingsqft26', 'fireplaceflag', 'propertycountylandusecode', 'hashottuborspa', 'basementsqft', 'fips', 'buildingclasstypeid', 'pooltypeid2', 'pooltypeid10', 'regionidcounty', 'heatingorsystemtypeid', 'rawcensustractandblock', 'censustractandblock', 'taxdelinquencyflag', 'airconditioningtypeid', 'pooltypeid7', 'regionidcity', 'regionidzip', 'decktypeid', 'typeconstructiontypeid', 'propertylandusetypeid', 'storytypeid']

label_col = 'logerror'
# ref: https://github.com/motefly/DeepGBM/issues/8

df_train["yearbuilt_cate"] = df_train["yearbuilt"]
df_train.to_csv("train.csv", index=False)
def unpackbits(x,num_bits):

    xshape = list(x.shape)

    x = x.reshape([-1,1])

    to_and = 2**np.arange(num_bits).reshape([1,num_bits])

    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])



class NumEncoder(object):

    def __init__(self, cate_col, nume_col, threshold, thresrate, label):

        self.label_name = label

        # cate_col = list(df.select_dtypes(include=['object']))

        self.cate_col = cate_col

        # nume_col = list(set(list(df)) - set(cate_col))

        self.dtype_dict = {}

        for item in cate_col:

            self.dtype_dict[item] = 'str'

        for item in nume_col:

            self.dtype_dict[item] = 'float'

        self.nume_col = nume_col

        self.tgt_nume_col = []

        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col)

        self.threshold = threshold

        self.thresrate = thresrate

        # for online update, to do

        self.save_cate_avgs = {}

        self.save_value_filter = {}

        self.save_num_embs = {}

        self.Max_len = {}

        self.samples = 0



    def fit_transform(self, inPath, outPath):

        print('----------------------------------------------------------------------')

        print('Fitting and Transforming %s .'%inPath)

        print('----------------------------------------------------------------------')

        df = pd.read_csv(inPath, dtype=self.dtype_dict)

        self.samples = df.shape[0]

        print('Filtering and fillna features')

        for item in tqdm(self.cate_col):

            value_counts = df[item].value_counts()

            num = value_counts.shape[0]

            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)

            rm_values = set(value_counts.index)-set(self.save_value_filter[item])

            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)

            df[item] = df[item].fillna('<UNK>')

            del value_counts

            gc.collect()



        for item in tqdm(self.nume_col):

            df[item] = df[item].fillna(df[item].mean())

            self.save_num_embs[item] = {'sum':df[item].sum(), 'cnt':df[item].shape[0]}



        print('Ordinal encoding cate features')

        # ordinal_encoding

        df = self.encoder.fit_transform(df)



        print('Target encoding cate features')

        # dynamic_targeting_encoding

        for item in tqdm(self.cate_col):

            feats = df[item].values

            labels = df[self.label_name].values

            feat_encoding = {'mean':[], 'count':[]}

            feat_temp_result = collections.defaultdict(lambda : [0, 0])

            self.save_cate_avgs[item] = collections.defaultdict(lambda : [0, 0])

            for idx in range(self.samples):

                cur_feat = feats[idx]

                # smoothing optional

                if cur_feat in self.save_cate_avgs[item]:

                    # feat_temp_result[cur_feat][0] = 0.9*feat_temp_result[cur_feat][0] + 0.1*self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1]

                    # feat_temp_result[cur_feat][1] = 0.9*feat_temp_result[cur_feat][1] + 0.1*self.save_cate_avgs[item][cur_feat][1]/idx

                    feat_encoding['mean'].append(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1])

                    feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1]/idx)

                else:

                    feat_encoding['mean'].append(0)

                    feat_encoding['count'].append(0)

                self.save_cate_avgs[item][cur_feat][0] += labels[idx]

                self.save_cate_avgs[item][cur_feat][1] += 1

            df[item+'_t_mean'] = feat_encoding['mean']

            df[item+'_t_count'] = feat_encoding['count']

            self.tgt_nume_col.append(item+'_t_mean')

            self.tgt_nume_col.append(item+'_t_count')

        

        print('Start manual binary encode')

        rows = None

        for item in tqdm(self.nume_col+self.tgt_nume_col):

            feats = df[item].values

            if rows is None:

                rows = feats.reshape((-1,1))

            else:

                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)

            del feats

            gc.collect()

        for item in tqdm(self.cate_col):

            feats = df[item].values

            Max = df[item].max()

            bit_len = len(bin(Max)) - 2

            samples = self.samples

            self.Max_len[item] = bit_len

            res = unpackbits(feats, bit_len).reshape((samples,-1))

            rows = np.concatenate([rows,res],axis=1)

            del feats

            gc.collect()

        trn_y = np.array(df[self.label_name].values).reshape((-1,1))

        del df

        gc.collect()

        trn_x = np.array(rows)

        np.save(outPath+'_features.npy', trn_x)

        np.save(outPath+'_labels.npy', trn_y)



    # for test dataset

    def transform(self, inPath, outPath):

        print('----------------------------------------------------------------------')

        print('Transforming %s .'%inPath)

        print('----------------------------------------------------------------------')

        df = pd.read_csv(inPath, dtype=self.dtype_dict)

        samples = df.shape[0]

        print('Filtering and fillna features')

        for item in tqdm(self.cate_col):

            value_counts = df[item].value_counts()

            rm_values = set(value_counts.index)-set(self.save_value_filter[item])

            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)

            df[item] = df[item].fillna('<UNK>')



        for item in tqdm(self.nume_col):

            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']

            df[item] = df[item].fillna(mean)



        print('Ordinal encoding cate features')

        # ordinal_encoding

        df = self.encoder.transform(df)



        print('Target encoding cate features')

        # dynamic_targeting_encoding

        for item in tqdm(self.cate_col):

            avgs = self.save_cate_avgs[item]

            df[item+'_t_mean'] = df[item].map(lambda x: avgs[x][0]/avgs[x][1] if x in avgs else 0)

            df[item+'_t_count'] = df[item].map(lambda x: avgs[x][1]/self.samples if x in avgs else 0)

        

        print('Start manual binary encode')

        rows = None

        for item in tqdm(self.nume_col+self.tgt_nume_col):

            feats = df[item].values

            if rows is None:

                rows = feats.reshape((-1,1))

            else:

                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)

            del feats

            gc.collect()

        for item in tqdm(self.cate_col):

            feats = df[item].values

            bit_len = self.Max_len[item]

            res = unpackbits(feats, bit_len).reshape((samples,-1))

            rows = np.concatenate([rows,res],axis=1)

            del feats

            gc.collect()

        vld_y = np.array(df[self.label_name].values).reshape((-1,1))

        del df

        gc.collect()

        vld_x = np.array(rows)

        np.save(outPath+'_features.npy', vld_x)

        np.save(outPath+'_labels.npy', vld_y)

    

    # for update online dataset

    def refit_transform(self, inPath, outPath):

        print('----------------------------------------------------------------------')

        print('Refitting and Transforming %s .'%inPath)

        print('----------------------------------------------------------------------')

        df = pd.read_csv(inPath, dtype=self.dtype_dict)

        samples = df.shape[0]

        print('Filtering and fillna features')

        for item in tqdm(self.cate_col):

            value_counts = df[item].value_counts()

            rm_values = set(value_counts.index)-set(self.save_value_filter[item])

            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)

            df[item] = df[item].fillna('<UNK>')



        for item in tqdm(self.nume_col):

            self.save_num_embs[item]['sum'] += df[item].sum()

            self.save_num_embs[item]['cnt'] += df[item].shape[0]

            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']

            df[item] = df[item].fillna(mean)



        print('Ordinal encoding cate features')

        # ordinal_encoding

        df = self.encoder.transform(df)



        print('Target encoding cate features')

        # dynamic_targeting_encoding

        for item in tqdm(self.cate_col):

            feats = df[item].values

            labels = df[self.label_name].values

            feat_encoding = {'mean':[], 'count':[]}

            for idx in range(samples):

                cur_feat = feats[idx]

                if self.save_cate_avgs[item][cur_feat][1] == 0:

                    pdb.set_trace()

                feat_encoding['mean'].append(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1])

                feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1]/(self.samples+idx))

                self.save_cate_avgs[item][cur_feat][0] += labels[idx]

                self.save_cate_avgs[item][cur_feat][1] += 1

            df[item+'_t_mean'] = feat_encoding['mean']

            df[item+'_t_count'] = feat_encoding['count']



        self.samples += samples

            

        print('Start manual binary encode')

        rows = None

        for item in tqdm(self.nume_col+self.tgt_nume_col):

            feats = df[item].values

            if rows is None:

                rows = feats.reshape((-1,1))

            else:

                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)

            del feats

            gc.collect()

        for item in tqdm(self.cate_col):

            feats = df[item].values

            bit_len = self.Max_len[item]

            res = unpackbits(feats, bit_len).reshape((samples,-1))

            rows = np.concatenate([rows,res],axis=1)

            del feats

            gc.collect()

        vld_y = np.array(df[self.label_name].values).reshape((-1,1))

        del df

        gc.collect()

        vld_x = np.array(rows)

        np.save(outPath+'_features.npy', vld_x)

        np.save(outPath+'_labels.npy', vld_y)

        # to do

        pass
class CateEncoder(object):

    def __init__(self, cate_col, nume_col, threshold, thresrate, bins, label):

        self.label_name = label

        # cate_col = list(df.select_dtypes(include=['object']))

        self.cate_col = cate_col 

        # nume_col = list(set(list(df)) - set(cate_col))

        self.dtype_dict = {}

        for item in cate_col:

            self.dtype_dict[item] = 'str'

        for item in nume_col:

            self.dtype_dict[item] = 'float'

        self.nume_col = nume_col

        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col+nume_col)

        self.threshold = threshold

        self.thresrate = thresrate

        self.bins = bins

        # for online update, to do

        self.save_value_filter = {}

        self.save_num_bins = {}

        self.samples = 0



    def save2npy(self, df, out_dir):

        if not os.path.isdir(out_dir):

            os.mkdir(out_dir)

        result = {'label':[], 'index':[],'feature_sizes':[]}

        result['label'] = df[self.label_name].values

        result['index'] = df[self.cate_col+self.nume_col].values

        for item in self.cate_col+self.nume_col:

            result['feature_sizes'].append(df[item].max()+1)

        for item in result:

            result[item] = np.array(result[item])

            np.save(out_dir + '_' + item +'.npy', result[item])



    def fit_transform(self, inPath, outPath):

        print('----------------------------------------------------------------------')

        print('Fitting and Transforming %s .'%inPath)

        print('----------------------------------------------------------------------')

        df = pd.read_csv(inPath, dtype=self.dtype_dict)

        print('Filtering and fillna features')

        for item in tqdm(self.cate_col):

            value_counts = df[item].value_counts()

            num = value_counts.shape[0]

            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)

            rm_values = set(value_counts.index)-set(self.save_value_filter[item])

            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)

            df[item] = df[item].fillna('<UNK>')



        print('Fillna and Bucketize numeric features')

        for item in tqdm(self.nume_col):

            q_res = pd.qcut(df[item], self.bins, labels=False, retbins=True, duplicates='drop')

            df[item] = q_res[0].fillna(-1).astype('int')

            self.save_num_bins[item] = q_res[1]



        print('Ordinal encoding cate features')

        # ordinal_encoding

        df = self.encoder.fit_transform(df)

        self.save2npy(df, outPath)

        # df.to_csv(outPath, index=False)



    # for test dataset

    def transform(self, inPath, outPath):

        print('----------------------------------------------------------------------')

        print('Transforming %s .'%inPath)

        print('----------------------------------------------------------------------')

        df = pd.read_csv(inPath, dtype=self.dtype_dict)

        print('Filtering and fillna features')

        for item in tqdm(self.cate_col):

            value_counts = df[item].value_counts()

            rm_values = set(value_counts.index)-set(self.save_value_filter[item])

            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)

            df[item] = df[item].fillna('<UNK>')



        for item in tqdm(self.nume_col):

            df[item] = pd.cut(df[item], self.save_num_bins[item], labels=False, include_lowest=True).fillna(-1).astype('int')



        print('Ordinal encoding cate features')

        # ordinal_encoding

        df = self.encoder.transform(df)

        self.save2npy(df, outPath)

        # df.to_csv(outPath, index=False)
# set default value

threshold = 10

thresrate = 0.99

num_bins = 32



ec = NumEncoder(cate_col, nume_col, threshold, thresrate, label_col)
ec.fit_transform("train.csv", str(output_tr_path))

# ec.transform(args['test_csv_path'], args['out_dir']+'/test')
ce = CateEncoder(cate_col, nume_col, threshold, thresrate, num_bins, label_col)

ce.fit_transform("train.csv", str(output_tr_path))
%ls out/