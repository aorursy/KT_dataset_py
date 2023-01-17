# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import pprint



import numpy as np

import pandas as pd



from joblib import Parallel, delayed

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import cohen_kappa_score, make_scorer

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import scipy as sp



from sklearn.model_selection import train_test_split, GridSearchCV



from functools import partial

from collections import Counter



import random

import math
#%% 评价函数 Metric used for this competition 

# (Quadratic Weigthed Kappa aka Quadratic Cohen Kappa Score)

def metric(y1,y2):

    return cohen_kappa_score(y1, y2, weights = 'quadratic')





# Make scorer for scikit-learn

scorer = make_scorer(metric)
from sklearn.model_selection import StratifiedKFold



#

def split_score(model, x_train, y_train, x_test, n=10):

    

    y_pre = np.zeros(x_train.shape[0])

    oof_test = np.zeros((x_test.shape[0], n))

    

    kfold = StratifiedKFold(n_splits=n, random_state=4)

    aaa = 0

    

    for train_index, test_index in kfold.split(x_train, y_train):

        model.fit(x_train.iloc[train_index], y_train.iloc[train_index])

        y_pre[test_index] = model.predict(x_train.iloc[test_index])

        test_pred = model.predict(x_test)

        oof_test[:, aaa] = test_pred

        

        print(aaa)

        aaa += 1

    

#    score = metric(y_pre, y)

    print("{}折后的Kappa加权得分为:带补充".format(n))

    

    return y_pre, oof_test



#

def fix_y(y, coef):

    y_fix = np.copy(y)

    for i, pred in enumerate(y_fix):

        if pred < coef[0]:

            y_fix[i] = 0

        elif pred >= coef[0] and pred < coef[1]:

            y_fix[i] = 1

        elif pred >= coef[1] and pred < coef[2]:

            y_fix[i] = 2

        elif pred >= coef[2] and pred < coef[3]:

            y_fix[i] = 3

        else:

            y_fix[i] = 4    

    return y_fix



# 

def _kappa_loss(y, y_true, coef):

    y_fix = np.copy(y)

    for i, pred in enumerate(y_fix):

        if pred < coef[0]:

            y_fix[i] = 0

        elif pred >= coef[0] and pred < coef[1]:

            y_fix[i] = 1

        elif pred >= coef[1] and pred < coef[2]:

            y_fix[i] = 2

        elif pred >= coef[2] and pred < coef[3]:

            y_fix[i] = 3

        else:

            y_fix[i] = 4

            

    loss = metric(y_fix, y_true)

    return -loss



# 寻找分类的最佳参数

def search_coef(x1, x2):

    loss_partial = partial(_kappa_loss, x1, x2)

    initial_coef = [1.55, 2.05, 2.5, 3]

    coef = sp.optimize.basinhopping(loss_partial, initial_coef, niter=500, T=1,

                                              stepsize=0.2, minimizer_kwargs={"method": 'nelder-mead'}, 

                                              take_step=None, accept_test=None, callback=None, 

                                              interval=100, disp=True, niter_success=10, seed=None)



    return coef
df_train  = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

df_test   = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')



train = df_train.copy()

test  = df_test.copy()



labels_breed = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

labels_state = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')

labels_color = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')
def extract_sentiment_feature(i, x):    

#    feature_sentiment = pd.DataFrame(columns=['PetID', 'token', 'sentence_magnitude', 'sentence_score','document_magnitude', 'document_score'])

    feature_sentiment = pd.DataFrame()



    if x == 'train':

        set_file = 'train'

    else:

        set_file = 'test' 

        

    file_name = '../input/petfinder-adoption-prediction/{}_sentiment/{}.json'.format(set_file,i)

    try:

        with open(file_name, 'r', encoding='utf-8') as f:

            sentiment_file = json.load(f)



            token = [x['name'] for x in sentiment_file['entities']]

            token = ' '.join(token)



            sentences_sentiment = [x['sentiment'] for x in sentiment_file['sentences']]

            sentences_sentiment = pd.DataFrame.from_dict(

                sentences_sentiment, orient='columns')



            

            docementSentiment_magnitude = sentiment_file['documentSentiment']['magnitude']

            documentSentiment_score     = sentiment_file['documentSentiment']['score']

            

            new = pd.DataFrame(

                    {'PetID'    : i, 



                    'magnitude_sum' : sentences_sentiment['magnitude'].sum(axis=0),

                    'score_sum'     : sentences_sentiment['score'].sum(axis=0),

                    'magnitude_mean': sentences_sentiment['magnitude'].mean(axis=0),

                    'score_mean'    : sentences_sentiment['score'].mean(axis=0),

                    'magnitude_var' : sentences_sentiment['magnitude'].var(axis=0),

                    'score_var'     : sentences_sentiment['score'].var(axis=0),



                    'document_magnitude'  : [docementSentiment_magnitude], 

                    'document_score'      : [documentSentiment_score]})  

            feature_sentiment = feature_sentiment.append(new)

    except:

        print('{}没找到'.format(file_name))

    

    for each in feature_sentiment.columns:

        if each not in ['PetID','token']:

            feature_sentiment[each] = feature_sentiment[each].astype(float)



    return feature_sentiment



#%%

train_feature_sentiment = Parallel(n_jobs=8, verbose=1)(

        delayed(extract_sentiment_feature)(i, 'train') for i in train.PetID)

train_feature_sentiment = [x for x in train_feature_sentiment]

train_feature_sentiment = pd.concat(train_feature_sentiment, ignore_index=True, sort=False)



test_feature_sentiment = Parallel(n_jobs=8, verbose=1)(

        delayed(extract_sentiment_feature)(i, 'test') for i in test.PetID)

test_feature_sentiment = [x for x in test_feature_sentiment]

test_feature_sentiment = pd.concat(test_feature_sentiment, ignore_index=True, sort=False)
picture_metadata  = pd.read_csv(r'../input/32111111/picture.csv')

picture_size      = pd.read_csv(r'../input/32111111/picture_size.csv')

img_features      = pd.read_csv(r'../input/32111111/img_features.csv')



#%% 连接sentiment和metadata和原始数据

x_train = df_train.merge(train_feature_sentiment, how='left', on='PetID')

x_train = x_train.merge(picture_metadata, how='left', on='PetID')

x_train = x_train.merge(picture_size, how='left', on='PetID')

x_train = x_train.merge(img_features, how='left', on='PetID')



y_train = x_train['AdoptionSpeed']

x_train.drop(['AdoptionSpeed'], axis=1, inplace=True)



x_test = df_test.merge(test_feature_sentiment, how='left', on='PetID')

x_test = x_test.merge(picture_metadata, how='left', on='PetID')

x_test = x_test.merge(picture_size, how='left', on='PetID')

x_test  = x_test.merge(img_features, how='left', on='PetID')



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import SparsePCA, TruncatedSVD, LatentDirichletAllocation, NMF



col_text = ['Description']



x = x_train.append(x_test).reset_index()

x = x[['Description', 'PetID']]



n_components = 50



x[col_text] = x[col_text].fillna('MISSING')

text_features = []





for i in  ['Description']:

    svd_ = TruncatedSVD(n_components=n_components)

        

    tfv = CountVectorizer(min_df=3,  

                          max_df=0.9,

                          stop_words = 'english')

    

    tfidf_col = tfv.fit_transform(x.loc[:, i])



    

    svd_col = svd_.fit_transform(tfidf_col)

    svd_col = pd.DataFrame(svd_col)

    svd_col = svd_col.add_prefix('SVD_{}_'.format(i))





    text_features.append(svd_col)    

    

    x.drop(i, axis=1, inplace=True)

    

# Combine all extracted features:

text_features = pd.concat(text_features, axis=1)



# Concatenate with main DF:

x = pd.concat([x, text_features], axis=1)



x_train = x_train.merge(x, how='left', on='PetID')

x_test  = x_test.merge(x, how='left', on='PetID')

from textblob import TextBlob



x = x_train.append(x_test)

x = x[['PetID', 'Description']]



x['Description'] = x['Description'].fillna("Missing")

x['Description'] = x['Description'].apply(lambda x:TextBlob(x))



x['polarity']     = x['Description'].apply(lambda x:x.sentiment[0])

x['subjectivity'] = x['Description'].apply(lambda x:x.sentiment[1])



#对情感进行分箱

bin=[-2,0,0.3,2]

x['polarity'] = pd.cut(x['polarity'], bins=bin, labels=range(3))

x['polarity'] = x['polarity'].astype(np.int32)



x_train = x_train.merge(x[['PetID', 'polarity']], how='left', on='PetID')

x_test  = x_test.merge(x[['PetID', 'polarity']], how='left', on='PetID')
#是否是免费的

x_train['IsFree'] = x_train['Fee'].apply(lambda x:True if x == 0 else False)

x_test['IsFree']  = x_test['Fee'].apply(lambda x:True if x == 0 else False)



#年龄（按年）

x_train['Year'] = x_train['Age'].apply(lambda x:math.floor(x/12))

x_test['Year']  = x_test['Age'].apply(lambda x:math.floor(x/12))



#年龄分箱,按频划成5份

x = x_train.append(x_test)

x['Age_qcut'] = pd.qcut(x['Age'], 5,  duplicates='drop')

x['Age_qcut'] = pd.factorize(x['Age_qcut'])[0]

x_train = x_train.merge(x[['PetID','Age_qcut']], how='left', on='PetID')

x_test  = x_test.merge(x[['PetID','Age_qcut']], how='left', on='PetID')





#血缘的种类

x = x_train.append(x_test).reset_index()

Breed1_count = x.groupby('Breed1').size().to_frame('Breed1_count').reset_index()

x_train = x_train.merge(Breed1_count, how='left', on='Breed1')

x_test  = x_test.merge(Breed1_count, how='left', on='Breed1')



#是否稀有

a = x['Breed1'].value_counts().sort_values(ascending = False).cumsum()/len(x)

rare1_index = a[a > 0.85].index.tolist()

x_train['IsRare1'] = x_train['Breed1'].isin(rare1_index).apply(lambda x:True if x == True else False)

x_test['IsRare1']  = x_test['Breed1'].isin(rare1_index).apply(lambda x:True if x == True else False)

rare2_index = a[a > 0.72].index.tolist()

x_train['IsRare2'] = x_train['Breed1'].isin(rare2_index).apply(lambda x:True if x == True else False)

x_test['IsRare2']  = x_test['Breed1'].isin(rare2_index).apply(lambda x:True if x == True else False)



#是否常见

x_train['Is_COMMON'] = x_train['Breed1'].apply(lambda x:True if (x == 265 or x == 307 or x == 266) else False)

x_test['Is_COMMON']  = x_test['Breed1'].apply(lambda x:True if (x == 265 or x == 307 or x == 266) else False)



#照片分箱

bin=[-0.5,0.5,1.5,4.5,1000]

x_train['Photo_cut'] = pd.cut(x_train['PhotoAmt'], bins=bin, labels=range(4)).astype(np.int32)

x_test['Photo_cut']  = pd.cut(x_test['PhotoAmt'], bins=bin, labels=range(4)).astype(np.int32)



# 是否是稀有颜色

x_train['Is_rare_color1'] = x_train['Color1'].apply(lambda x:True if x==5 or x==6 or x==7 else False)

x_test['Is_rare_color1'] = x_test['Color1'].apply(lambda x:True if x==5 or x==6 or x==7 else False)

x_train['Is_rare_color2'] = x_train['Color2'].apply(lambda x:True if x==6 else False)

x_test['Is_rare_color2'] = x_test['Color2'].apply(lambda x:True if x==6 else False)



#年龄是否小于二月

x_train['Is_less_than_2month']= x_train['Age'].apply(lambda x:True if x<3 else False)

x_test['Is_less_than_2month'] = x_test['Age'].apply(lambda x:True if x<3 else False)

#%% RescuerID 处理



df = df_train.append(df_test)

data_rescuer = df.groupby(['RescuerID'])['PetID'].size().reset_index()

data_rescuer.columns = ['RescuerID', 'RescuerID_count']

#data_rescuer['rank_Rescuer_count'] = data_rescuer['RescuerID_count'].rank(pct=True)



x_train = x_train.merge(data_rescuer, how='left', on='RescuerID')

x_test  = x_test.merge(data_rescuer, how='left', on='RescuerID')



x = x_train.append(x_test)

x['RescuerID_count_cut'] = pd.qcut(x['RescuerID_count'], 5, labels=range(4), duplicates='drop').astype(np.int32)



x_train = x_train.merge(x[['PetID', 'RescuerID_count_cut']], how='left', on='PetID')

x_test  = x_test.merge(x [['PetID', 'RescuerID_count_cut']], how='left', on='PetID')



#x_train.drop(['RescuerID_count'], axis=1, inplace=True)

#x_test.drop(['RescuerID_count'], axis=1, inplace=True)
# 增加特征 是否有第二血统

x_train['HasSecondBreed'] = x_train['Breed2'].map(lambda x:True if x != 0 else False)

x_test['HasSecondBreed'] = x_test['Breed2'].map(lambda x:True if x != 0 else False)



train_breed_main = x_train[['Breed1']].merge(

    labels_breed, how='left',

    left_on='Breed1', right_on='BreedID',

    suffixes=('', '_main_breed'))



train_breed_main = train_breed_main.iloc[:, 2:]

train_breed_main = train_breed_main.add_prefix('main_breed_')



train_breed_second = x_train[['Breed2']].merge(

    labels_breed, how='left',

    left_on='Breed2', right_on='BreedID',

    suffixes=('', '_second_breed'))



train_breed_second = train_breed_second.iloc[:, 2:]

train_breed_second = train_breed_second.add_prefix('second_breed_')



x_train = pd.concat(

    [x_train, train_breed_main, train_breed_second], axis=1)



##############

test_breed_main = x_test[['Breed1']].merge(

    labels_breed, how='left',

    left_on='Breed1', right_on='BreedID',

    suffixes=('', '_main_breed'))



test_breed_main = test_breed_main.iloc[:, 2:]

test_breed_main = test_breed_main.add_prefix('main_breed_')



test_breed_second = x_test[['Breed2']].merge(

    labels_breed, how='left',

    left_on='Breed2', right_on='BreedID',

    suffixes=('', '_second_breed'))



test_breed_second = test_breed_second.iloc[:, 2:]

test_breed_second = test_breed_second.add_prefix('second_breed_')



x_test = pd.concat(

    [x_test, test_breed_main, test_breed_second], axis=1)



print(x_train.shape, x_test.shape)



categorical_columns = ['main_breed_BreedName', 'second_breed_BreedName']

#for i in categorical_columns:

#    x_train.loc[:, i] = pd.factorize(x_train.loc[:, i])[0]

#    x_test.loc[:,i]   = pd.factorize(x_test.loc[:, i])[0]



# 增加特征 是否纯种

x_train['True_Pure'] = False

x_train.loc[(x_train['main_breed_BreedName'] != 'Mixed Breed')&

                    ((x_train['main_breed_BreedName'] == x_train['second_breed_BreedName'])|

                   (x_train['second_breed_BreedName'].isnull())),'True_Pure'] = True





x_test['True_Pure'] = False

x_test.loc[(x_test['main_breed_BreedName'] != 'Mixed Breed')&

                    ((x_test['main_breed_BreedName'] == x_test['second_breed_BreedName'])|

                   (x_test['second_breed_BreedName'].isnull())),'True_Pure'] = True



# 是否纯种狗

x_train['Is_Pure_Dog'] = (x_train['True_Pure'] == True) & (x_train['Type'] == 1)

x_test['Is_Pure_Dog']  = (x_test['True_Pure'] == True)  & (x_test['Type'] == 1)







#删除没用特征

x_train.drop(['main_breed_BreedName', 'second_breed_BreedName', 'main_breed_Type', 'second_breed_Type'], axis=1, inplace=True)

x_test.drop(['main_breed_BreedName', 'second_breed_BreedName', 'main_breed_Type', 'second_breed_Type'], axis=1, inplace=True)
drop_columns = ['Name', 'RescuerID', 'Description', 'PetID', 'token', 'annots_top_desc']

drop_columns = ['Name', 'RescuerID', 'Description', 'PetID']



x_train.drop(drop_columns, axis=1, inplace=True)

x_test.drop(drop_columns, axis=1, inplace=True)



x_train = x_train.fillna(0)

x_test  = x_test.fillna(0)

#属性标签

c = ['Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',

    'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity',

    'State', 'IsFree', 'Year', 'Age_qcut', 'IsRare1', 'IsRare2',

    'Is_COMMON', 'Photo_cut', 'Is_rare_color1', 'Is_rare_color2', 'Is_less_than_2month',

    'RescuerID_count_cut', 'HasSecondBreed', 'True_Pure', 'Is_Pure_Dog']



for each in c:

    x_train[each] = x_train[each].astype('category')

    x_test[each] = x_test[each].astype('category')
from lightgbm.sklearn import LGBMRegressor



model_lgb = LGBMRegressor(

        learning_rate    = 0.01,

        n_estimators     = 2000,

        max_depth        = 4,

        num_leaves       = 12 ,

        subsample        = 0.8,      #训练时采样一定比例的数据	

        colsample_bytree = 0.8,

        n_jobs           = -1,

        random_state     = 44,

        objective        = 'regression',

#        reg_alpha        = 1,

        eval_metric      = 'scorer',

        min_child_samples = 15         #叶子节点具有的最小记录数	

        )

        
y_lgb, y_test_pre = split_score(model_lgb, x_train, y_train, x_test)

y_test_pre = y_test_pre.mean(axis=1)



coe = search_coef(y_lgb, y_train)

best_lgb_coe = coe['x']

print('lgb的最佳系数为{}'.format(best_lgb_coe))



model_lgb.fit(x_train, y_train)

result_lgb_fix = fix_y(y_test_pre, best_lgb_coe)

print('lgb后的分布:',Counter(result_lgb_fix))
submission_lgb = pd.DataFrame({'PetID': df_test['PetID'].values, 'AdoptionSpeed': result_lgb_fix.astype(np.int32)})

submission_lgb.to_csv('submission.csv', index=False)
#submission_xgb = pd.DataFrame({'PetID': df_test['PetID'].values, 'AdoptionSpeed': result_xgb_fix.astype(np.int32)})

#submission_xgb.to_csv('submission.csv', index=False)