import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgbm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from sklearn.metrics import recall_score, precision_score, precision_recall_curve, fbeta_score, roc_auc_score, make_scorer

from datetime import datetime, timedelta

from imblearn.over_sampling import SMOTE



%matplotlib inline
#导入原始数据集

ini_user = pd.read_csv('../input/jd-customer-purchase-intention-prediction/JData_User.csv')

ini_product = pd.read_csv('../input/jd-customer-purchase-intention-prediction/JData_Product.csv')

ini_comment = pd.read_csv('../input/jd-customer-purchase-intention-prediction/JData_Comment.csv')

ini_action_201603 = pd.read_csv('../input/jd-customer-purchase-intention-prediction/JData_Action_201603.csv')

ini_action_201604 = pd.read_csv('../input/jd-customer-purchase-intention-prediction/JData_Action_201604.csv')
#删除各数据集中完全重复的行



user_dedup = ini_user.drop_duplicates()

product_dedup = ini_product.drop_duplicates()

comment_dedup = ini_comment.drop_duplicates()

action03_dedup = ini_action_201603.drop_duplicates()

action04_dedup = ini_action_201604.drop_duplicates()

    

print('shape of user_dedup: ', user_dedup.shape)

print('shape of product_dedup: ', product_dedup.shape)

print('shape of comment_dedup: ', comment_dedup.shape)

print('shape of action03_dedup: ', action03_dedup.shape)

print('shape of action04_dedup: ', action04_dedup.shape)
#检查action表的用户与user表中的用户是否一致,检查action表的产品与product表中的产品是否一致



def action_user_product_check():

    print('action user match:')

    print('Is action03_user match user? ', len(action03_dedup) == len(pd.merge(action03_dedup, user_dedup)))

    print('Is action04_user match user? ', len(action04_dedup) == len(pd.merge(action04_dedup, user_dedup)))

    print('\naction product match:')

    print('Is action03_product match product? ', len(action03_dedup) == len(pd.merge(action03_dedup, product_dedup, on='sku_id')))

    print('Is action04_product match product? ', len(action04_dedup) == len(pd.merge(action04_dedup, product_dedup, on='sku_id')))

    



action_user_product_check()
#除去action表中不存在于product表的产品



action03_depro = action03_dedup[action03_dedup['sku_id'].isin(list(product_dedup['sku_id']))]

action04_depro = action04_dedup[action04_dedup['sku_id'].isin(list(product_dedup['sku_id']))]

print('shape of action03_dedup is (18831340, 7) and shape of action03_depro is', action03_depro.shape)

print('shape of action04_dedup is (9527224, 7) and shape of action04_depro is', action04_depro.shape)
#检查comment表中是否含有product表中不含有的产品



print('Is comment_product match product? ', len(comment_dedup) == len(pd.merge(comment_dedup, product_dedup)))
comment_depro = comment_dedup[comment_dedup['sku_id'].isin(list(product_dedup['sku_id']))]

print('shape of comment_dedup is (558552, 5) and shape of comment_depro is', comment_depro.shape)
# 查看每个特征的缺失值数量与比例



def missing_values_table(data):

    #计算缺失值数量与占比

    mis_val = data.isna().sum()

    mis_val_percent = 100 * data.isna().sum() / len(data)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    

    #重命名表列

    mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0:'missing values', 1:'% of total values'})

    

    #按照缺失值占比对表格进行排序

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] !=0].sort_values(

    '% of total values', ascending = False).round(1)

    

    return mis_val_table_ren_columns
missing_values_table(user_dedup)
missing_values_table(product_dedup)
missing_values_table(comment_depro)
missing_values_table(action03_depro)
missing_values_table(action04_depro)
user_dena = user_dedup.fillna(method='ffill')

action03_dena = action03_depro.drop('model_id', axis=1)

action04_dena = action04_depro.drop('model_id', axis=1)
user_dena.describe()
product_dedup.describe()
comment_depro.describe()
action03_dena.describe()
action04_dena.describe()
#查看user表各特征数据类型

user_dena.info()
#把user表中user_reg_tm特征转换为datetime格式

user_dena['user_reg_tm'] = pd.to_datetime(user_dena['user_reg_tm'])

user_dena.info()
#查看product表各特征数据类型

product_dedup.info()
#查看comment表各特征数据类型

comment_depro.info()
#把comment表中dt特征转换为datetime格式

comment_depro['dt'] = pd.to_datetime(comment_depro['dt'])

comment_depro.info()
#查看action03表各特征数据类型

action03_dena.info()
#把action03表中time特征转换为datetime格式（在转换之前先保留一份未转换的）

action03_date = action03_dena.copy()

action03_dena['time'] = pd.to_datetime(action03_dena['time'])

action03_dena.info()
#查看action04表各特征数据类型

action04_dena.info()
#把action04表中time特征转换为datetime格式（在转换之前先保留一份未转换的）

action04_date = action04_dena.copy()

action04_dena['time'] = pd.to_datetime(action04_dena['time'])

action04_dena.info()
#计算Q1与Q3

first_quartile = comment_depro['bad_comment_rate'].describe()['25%']

third_quartile = comment_depro['bad_comment_rate'].describe()['75%']

#计算IQR

iqr = third_quartile - first_quartile

#剔除离群点

comment_depro = comment_depro[

    (comment_depro['bad_comment_rate'] > (first_quartile - 3*iqr))&

(comment_depro['bad_comment_rate'] < (third_quartile + 3*iqr))]
#对行为时间在2016-04-16及之后的记录进行检查

action04_dena.loc[action04_dena['time'] >= '2016-04-16']
#对用户注册时间在2016-04-16及之后的记录进行检查

user_dena.loc[user_dena['user_reg_tm'] >= '2016-04-16']
#计算注册时间众数

user_dena['user_reg_tm'].value_counts()
#用2015-11-11替换用户注册时间在2016-04-16及之后的记录

user_dena2 = user_dena.copy()

user_dena2.replace(list(user_dena2[user_dena2['user_reg_tm'] >= '2016-04-16']['user_reg_tm']), '2015-11-11',inplace=True)

user_dena2['user_reg_tm'] = pd.to_datetime(user_dena2['user_reg_tm'])

user_dena2.loc[user_dena2['user_reg_tm'] >= '2016-04-16']
user = user_dena2

product = product_dedup

comment = comment_depro

action03 = action03_dena

action04 = action04_dena

action = pd.concat([action03, action04], ignore_index=True)
action_date = pd.concat([action03_date, action04_date], ignore_index=True)

action_date['date'] = action_date['time'].map(lambda x: x.split(' ')[0])

action_date['date'] = pd.to_datetime(action_date['date'])

del action_date['time']
#提取标签值



def get_label(test_end_date, test_period):

    #提取标签值所在时间区间的数据

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    label = action[action['time'] >= test_start_date]

    #构造标签值

    label = label[label['type'] == 4]

    label = label[['user_id', 'sku_id', 'type']]

    label = label.groupby(['user_id', 'sku_id'], as_index=False).sum()

    label['label'] = 1

    label = label[['user_id', 'sku_id', 'label']]  

    return label
#为user、product和comment表加上label，构造用于数据探索的数据集



labels = get_label('2016-04-15', 5)



user_label = pd.merge(user, labels, how='left', on='user_id')

user_label = user_label.fillna(0)



product_label = pd.merge(product, labels, how='left', on='sku_id')

product_label = product_label.fillna(0)



comment_label = pd.merge(comment, labels, how='left', on='sku_id')

comment_label = comment_label.fillna(0)
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='age', data=user_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('age',hue='label',data=user_label, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='sex', data=user_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('sex',hue='label',data=user_label, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='user_lv_cd', data=user_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('user_lv_cd',hue='label',data=user_label, palette='hls')
#这里需要先对用户注册时间分箱

user_label2 = user_label.copy()

user_label2['user_reg_tmbin'] = pd.cut(user_label2['user_reg_tm'], 10)



plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='user_reg_tmbin', data=user_label2, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('user_reg_tmbin',hue='label',data=user_label2, palette='hls')

plt.xticks(rotation='vertical')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='a1', data=product_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('a1',hue='label',data=product_label, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='a2', data=product_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('a2',hue='label',data=product_label, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='a3', data=product_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('a3',hue='label',data=product_label, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='brand', data=product_label, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('brand',hue='label',data=product_label, palette='hls')
comment_label2 = comment_label.copy()

comment_label2 = comment_label2[comment_label2['dt'] == '2016-04-04']



plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='comment_num', data=comment_label2, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('comment_num',hue='label',data=comment_label2, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.countplot(y='has_bad_comment', data=comment_label2, palette='hls');

plt.subplot(1, 2, 2)

sns.countplot('has_bad_comment',hue='label',data=comment_label2, palette='hls')
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(20, 5))

plt.subplot(1, 2, 1)

sns.distplot(comment_label2.loc[comment_label2['label'] == 1]['bad_comment_rate'], kde_kws={'label':'buy'})

sns.distplot(comment_label2.loc[comment_label2['label'] == 0]['bad_comment_rate'], kde_kws={'label':'not buy'})
#构造周购买记录分析数据集



action_pr = action[['user_id', 'sku_id', 'type', 'time']]

action_pr = action_pr[action_pr['type'] == 4]

action_pr = action_pr[['user_id', 'sku_id', 'time']]

action_pr['time'] = action_pr['time'].apply(lambda x: x.weekday() + 1)

action_pr = action_pr.groupby('time').size().to_frame().reset_index()

action_pr.columns = (['weekday', 'purchase_num'])



#画图



fig = plt.figure(figsize=(8, 5))

sns.barplot(x='weekday', y='purchase_num', data=action_pr, palette='hls')
#构造月购买记录分析数据集



def get_action_mpr(action_set):

    action_mpr = action_set[action_set['type'] == 4][['user_id', 'sku_id', 'time']]

    action_mpr['time'] = action_mpr['time'].apply(lambda x: x.day)

    action_mpr = action_mpr.groupby('time').size().to_frame().reset_index()

    action_mpr.columns = (['day', 'purchase_num'])

    return action_mpr



#画图



fig = plt.figure(figsize=(20, 20))



plt.subplot(2, 1, 1)

sns.barplot(x='day', y='purchase_num', data=get_action_mpr(action03), palette='hls')

plt.title('March purchase record')



plt.subplot(2, 1, 2)

sns.barplot(x='day', y='purchase_num', data=get_action_mpr(action04), palette='hls')

plt.title('April purchase record')
#提取特征集的index（user_id与sku_id）

def get_feature_index(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    fea_index = data[data['time'] < test_start_date]

    fea_index = fea_index[['user_id', 'sku_id']]

    fea_index = fea_index.drop_duplicates()

    

    return fea_index
#提取年龄特征age（data用user）

def get_age(data):

    age = data[['user_id', 'age']]

    return age
#构造性别特征sex（data用user）

def get_sex(data):

    

    sex = data[['user_id', 'sex']]

    sex = pd.get_dummies(sex, columns=['sex'], prefix='sex')

    

    return sex
#提取用户等级user_lv_cd（data用user）

def get_user_lv(data):

    user_lv = data[['user_id', 'user_lv_cd']]

    return user_lv
#构造用户注册距今时间特征user_reg_tillnow（data用action）

def get_user_reg_tillnow(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    

    user_reg_tillnow = data[['user_id', 'user_reg_tm']]

    df = test_start_date - user_reg_tillnow['user_reg_tm']

    

    user_reg_tillnow = pd.concat([user_reg_tillnow, df], axis=1)

    user_reg_tillnow.columns = (['user_id', 'user_reg_tm', 'reg_tillnow'])

    user_reg_tillnow = user_reg_tillnow[['user_id', 'reg_tillnow']]

    user_reg_tillnow['reg_tillnow'] = user_reg_tillnow['reg_tillnow'].map(lambda x: x.days)

    user_reg_tillnow['reg_tillnow'] = user_reg_tillnow['reg_tillnow'].astype('int')

    

    return user_reg_tillnow
#为属性特征进行onehot编码（data用product）

def get_pro_atrr(data):

    

    pro_atrr = data[['sku_id', 'a1', 'a2', 'a3']]

    pro_atrr = pd.get_dummies(pro_atrr, columns=['a1', 'a2', 'a3'])

    

    return pro_atrr
#提取商品品牌特征brand（data用product）

def get_brand(data):

    brand = data[['sku_id', 'brand']]

    return brand
#商品累计评论数分段comment_num（data用comment）

def get_comment_num(data):

    

    comment_num = data[data['dt'] == '2016-04-04'][['sku_id', 'comment_num']]

    comment_num = pd.get_dummies(comment_num, columns=['comment_num'])

    

    return comment_num
#商品是否有差评has_bad_comment（data用comment）

def get_has_bad_comment(data):

    

    has_bad_comment = data[data['dt'] == '2016-04-04'][['sku_id', 'has_bad_comment']]

    has_bad_comment = pd.get_dummies(has_bad_comment, columns=['has_bad_comment'])

    

    return has_bad_comment
#商品差评率bad_comment_rate（data用comment）

def get_bad_comment_rate(data):

    bad_comment_rate = data[data['dt'] == '2016-04-04'][['sku_id', 'bad_comment_rate']]

    return bad_comment_rate
#各用户行为总数user_allact（data用action）

def get_user_allact(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    user_allact = data[data['time'] < test_start_date][['user_id', 'type']]

    user_allact = user_allact.groupby('user_id').size().to_frame().reset_index()

    user_allact.columns = ['user_id', 'user_allact']

    

    return user_allact
#各用户各种行为数user_act_num_（data用action）

def get_user_act_num_(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df = data[data['time'] < test_start_date][['user_id', 'type']]

    df = pd.get_dummies(df, columns=['type'], prefix='user_act_num')

    df = df.groupby('user_id').sum().reset_index()

    

    return df
#各商品行为总数pro_allact（data用action）

def get_pro_allact(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    pro_allact = data[data['time'] < test_start_date][['sku_id', 'type']]

    pro_allact = pro_allact.groupby('sku_id').size().to_frame().reset_index()

    pro_allact.columns = ['sku_id', 'pro_allact']

    

    return pro_allact
#各商品各种行为数pro_act_num_（data用action）

def get_pro_act_num_(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df1 = data[data['time'] < test_start_date][['sku_id', 'type']]

    df1 = pd.get_dummies(df1, columns=['type'], prefix='pro_act_num')

    df1 = df1.groupby('sku_id').sum().reset_index()

    

    return df1
#用户&商品行为总数up_allact（data用action）

def get_up_allact(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    up_allact = data[data['time'] < test_start_date][['user_id', 'sku_id', 'type']]

    up_allact = up_allact.groupby(['user_id', 'sku_id']).size().to_frame().reset_index()

    up_allact.columns = ['user_id', 'sku_id', 'up_allact']

    

    return up_allact
#用户&商品各种行为数up_act_num_（data用action）

def get_up_act_num_(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df2 = data[data['time'] < test_start_date][['user_id', 'sku_id', 'type']]

    df2 = pd.get_dummies(df2, columns=['type'], prefix='up_act_num')

    df2 = df2.groupby(['user_id', 'sku_id']).sum().reset_index()

    

    return df2
#各品牌行为总数brand_allact（data用action）

def get_brand_allact(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    brand_allact = data[data['time'] < test_start_date][['brand', 'type']]

    brand_allact = brand_allact.groupby('brand').size().to_frame().reset_index()

    brand_allact.columns = ['brand', 'brand_allact']

    

    return brand_allact
#各品牌各种行为数brand_act_num_（data用action）

def get_brand_act_num_(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df3 = data[data['time'] < test_start_date][['brand', 'type']]

    df3 = pd.get_dummies(df3, columns=['type'], prefix='brand_act_num')

    df3 = df3.groupby('brand').sum().reset_index()

    

    return df3
#构造单一滑窗期的用户滑窗统计特征（data用action）

def get_user_slidewindow_fea(test_end_date, test_period, data, days):

    

    #获取索引

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    index = data[data['time'] < test_start_date]['user_id'].drop_duplicates()

    

    #获取滑窗期间的数据

    day_before = test_start_date - timedelta(days)

    df_sw = data[(data['time'] >= day_before) & (data['time'] < test_start_date)]

    df_sw = df_sw[['user_id', 'type']]

    

    #获取滑窗期用户总行为数user_allact_before_n_

    user_allact_before = df_sw.groupby('user_id').size().to_frame().reset_index()

    user_allact_before.columns = ['user_id', 'user_allact_before_%s'%days]

    

    #获取滑窗期用户各行为数user_actnum_before_n_

    user_actnum_before = pd.get_dummies(df_sw, columns=['type'], prefix='user_actnum_before_%s'%days)

    user_actnum_before = user_actnum_before.groupby('user_id').sum().reset_index()

    

    #第一次特征汇总

    df_sw = pd.merge(index, user_allact_before, on='user_id', how='left')

    df_sw = pd.merge(df_sw, user_actnum_before, on='user_id', how='left')

    

    #获取滑窗期用户总购买率user_buy_rate_before_n（4：下单）

    df_sw['user_buy_rate_before_%s'%days] = df_sw['user_actnum_before_%s_4'%days] / df_sw['user_allact_before_%s'%days]

    

    #获取滑窗期用户浏览、加购、关注的购买转化率user_actx_buy_rate_before_n（1浏览，2加购，4下单，5关注）

    df_sw['user_act1_buy_rate_before_%s'%days] = df_sw['user_actnum_before_%s_4'%days] / df_sw['user_actnum_before_%s_1'%days]

    df_sw['user_act2_buy_rate_before_%s'%days] = df_sw['user_actnum_before_%s_4'%days] / df_sw['user_actnum_before_%s_2'%days]

    df_sw['user_act5_buy_rate_before_%s'%days] = df_sw['user_actnum_before_%s_4'%days] / df_sw['user_actnum_before_%s_5'%days]

    

    return df_sw
#获取所有滑窗期的用户滑窗统计特征（data用action）

def get_all_user_slidewindow_fea(test_end_date, test_period, data):

    

    df_sw = None

    for days in (1, 3, 7, 15):

        if df_sw is None:

            df_sw = get_user_slidewindow_fea(test_end_date, test_period, data, days)

        else:

            df_sw1 = get_user_slidewindow_fea(test_end_date, test_period, data, days)

            df_sw = pd.merge(df_sw, df_sw1, on='user_id', how='left')

    

    return df_sw
#构造单一滑窗期的商品滑窗统计特征（data用action）

def get_pro_slidewindow_fea(test_end_date, test_period, data, days):

    

    #获取索引

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    index = data[data['time'] < test_start_date]['sku_id'].drop_duplicates()

    

    #获取滑窗期间的数据

    day_before = test_start_date - timedelta(days)

    df_sw = data[(data['time'] >= day_before) & (data['time'] < test_start_date)]

    df_sw = df_sw[['sku_id', 'type']]

    

    #获取滑窗期商品总行为数pro_allact_before_n_

    pro_allact_before = df_sw.groupby('sku_id').size().to_frame().reset_index()

    pro_allact_before.columns = ['sku_id', 'pro_allact_before_%s'%days]

    

    #获取滑窗期商品各行为数pro_actnum_before_n_

    pro_actnum_before = pd.get_dummies(df_sw, columns=['type'], prefix='pro_actnum_before_%s'%days)

    pro_actnum_before = pro_actnum_before.groupby('sku_id').sum().reset_index()

    

    #第一次特征汇总

    df_sw = pd.merge(index, pro_allact_before, on='sku_id', how='left')

    df_sw = pd.merge(df_sw, pro_actnum_before, on='sku_id', how='left')

    

    #获取滑窗期商品总购买率pro_buy_rate_before_n（4：下单）

    df_sw['pro_buy_rate_before_%s'%days] = df_sw['pro_actnum_before_%s_4'%days] / df_sw['pro_allact_before_%s'%days]

    

    #获取滑窗期商品浏览、加购、关注的购买转化率pro_actx_buy_rate_before_n（1浏览，2加购，4下单，5关注）

    df_sw['pro_act1_buy_rate_before_%s'%days] = df_sw['pro_actnum_before_%s_4'%days] / df_sw['pro_actnum_before_%s_1'%days]

    df_sw['pro_act2_buy_rate_before_%s'%days] = df_sw['pro_actnum_before_%s_4'%days] / df_sw['pro_actnum_before_%s_2'%days]

    df_sw['pro_act5_buy_rate_before_%s'%days] = df_sw['pro_actnum_before_%s_4'%days] / df_sw['pro_actnum_before_%s_5'%days]

    

    return df_sw
#获取所有滑窗期的商品滑窗统计特征（data用action）

def get_all_pro_slidewindow_fea(test_end_date, test_period, data):

    

    df_sw = None

    for days in (1, 3, 7, 15):

        if df_sw is None:

            df_sw = get_pro_slidewindow_fea(test_end_date, test_period, data, days)

        else:

            df_sw1 = get_pro_slidewindow_fea(test_end_date, test_period, data, days)

            df_sw = pd.merge(df_sw, df_sw1, on='sku_id', how='left')

    

    return df_sw
#构造单一滑窗期的用户&商品滑窗统计特征（data用action）

def get_up_slidewindow_fea(test_end_date, test_period, data, days):

    

    #获取索引

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    index = data[data['time'] < test_start_date][['user_id', 'sku_id']].drop_duplicates()

    

    #获取滑窗期间的数据

    day_before = test_start_date - timedelta(days)

    df_sw = data[(data['time'] >= day_before) & (data['time'] < test_start_date)]

    df_sw = df_sw[['user_id', 'sku_id', 'type']]

    

    #获取滑窗期用户&商品总行为数up_allact_before_n_

    up_allact_before = df_sw.groupby(['user_id', 'sku_id']).size().to_frame().reset_index()

    up_allact_before.columns = ['user_id', 'sku_id', 'up_allact_before_%s'%days]

    

    #获取滑窗期用户&商品各行为数up_actnum_before_n_

    up_actnum_before = pd.get_dummies(df_sw, columns=['type'], prefix='up_actnum_before_%s'%days)

    up_actnum_before = up_actnum_before.groupby(['user_id', 'sku_id']).sum().reset_index()

    

    #第一次特征汇总

    df_sw = pd.merge(index, up_allact_before, on=['user_id', 'sku_id'], how='left')

    df_sw = pd.merge(df_sw, up_actnum_before, on=['user_id', 'sku_id'], how='left')

    

    #获取滑窗期用户&商品总购买率up_buy_rate_before_n（4：下单）

    df_sw['up_buy_rate_before_%s'%days] = df_sw['up_actnum_before_%s_4'%days] / df_sw['up_allact_before_%s'%days]

    

    #获取滑窗期用户&商品浏览、加购、关注的购买转化率up_actx_buy_rate_before_n（1浏览，2加购，4下单，5关注）

    df_sw['up_act1_buy_rate_before_%s'%days] = df_sw['up_actnum_before_%s_4'%days] / df_sw['up_actnum_before_%s_1'%days]

    df_sw['up_act2_buy_rate_before_%s'%days] = df_sw['up_actnum_before_%s_4'%days] / df_sw['up_actnum_before_%s_2'%days]

    df_sw['up_act5_buy_rate_before_%s'%days] = df_sw['up_actnum_before_%s_4'%days] / df_sw['up_actnum_before_%s_5'%days]

    

    return df_sw
#获取所有滑窗期的用户&商品滑窗统计特征（data用action）

def get_all_up_slidewindow_fea(test_end_date, test_period, data):

    

    df_sw = None

    for days in (1, 3, 7, 15):

        if df_sw is None:

            df_sw = get_up_slidewindow_fea(test_end_date, test_period, data, days)

        else:

            df_sw1 = get_up_slidewindow_fea(test_end_date, test_period, data, days)

            df_sw = pd.merge(df_sw, df_sw1, on=['user_id', 'sku_id'], how='left')

    

    return df_sw
#各品牌累计评论数分段brand_comment_num

def get_brand_comment_num():

   

    #连表获取品牌的评论数据

    comment1 = comment[comment['dt'] == '2016-04-04']

    product1 = product[['sku_id', 'brand']]

    df_cm = pd.merge(comment1, product1, on='sku_id', how='right').fillna(0)

    df_cm = df_cm[['brand', 'comment_num']]

    

    #计算品牌累计评论数分段

    brand_comment_num = df_cm.groupby('brand', as_index=False).mean()

    brand_comment_num.columns = ['brand', 'brand_comment_num']

    return brand_comment_num
#各品牌差评率brand_bad_comment_rate

def get_brand_bad_comment_rate():

    

    #连表获取品牌的评论数据

    comment1 = comment[comment['dt'] == '2016-04-04']

    product1 = product[['sku_id', 'brand']]

    df_cm = pd.merge(comment1, product1, on='sku_id', how='right').fillna(0)

    df_cm = df_cm[['brand', 'bad_comment_rate']]

    

    #计算品牌累计评论数分段

    brand_bad_comment_rate = df_cm.groupby('brand', as_index=False).mean()

    brand_bad_comment_rate.columns = ['brand', 'brand_bad_comment_rate']

    return brand_bad_comment_rate
#各品牌购买率brand_buy_rate（data用action）

def get_brand_buy_rate(test_end_date, test_period, data):

    

    df_cm = pd.merge(get_brand_allact(test_end_date, test_period, data), get_brand_act_num_(test_end_date, test_period, data), on='brand')

    df_cm['brand_buy_rate'] = df_cm['brand_act_num_4'] / df_cm['brand_allact']

    df_cm = df_cm[['brand', 'brand_buy_rate']]

    brand_buy_rate = df_cm

    

    return brand_buy_rate
#各品牌用户数brand_user_num（data用action）

def get_brand_user_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['brand', 'user_id']].drop_duplicates(['brand', 'user_id'])

    

    df_b = df_b.groupby('brand').size().to_frame().reset_index()

    df_b.columns = ['brand', 'brand_user_num']

    brand_user_num = df_b

    

    return brand_user_num
#各品牌购买用户数brand_buy_user_num（data用action）

def get_brand_buy_user_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[(data['time'] < test_start_date) & (data['type'] == 4)][['brand', 'user_id']].drop_duplicates(['brand', 'user_id'])

    

    df_b = df_b.groupby('brand').size().to_frame().reset_index()

    df_b.columns = ['brand', 'brand_buy_user_num']

    brand_buy_user_num = df_b

    

    return brand_buy_user_num
#各品牌用户购买率brand_buy_user_rate（data用action）

def get_brand_buy_user_rate(test_end_date, test_period, data):

    

    df_b = pd.merge(get_brand_buy_user_num(test_end_date, test_period, data), get_brand_user_num(test_end_date, test_period, data), on='brand')

    df_b['brand_buy_user_rate'] = df_b['brand_buy_user_num'] / df_b['brand_user_num']

    df_b = df_b[['brand', 'brand_buy_user_rate']]

    brand_buy_user_rate = df_b

    

    return brand_buy_user_rate
#各品牌出现频次占比brand_occur_rate（data用action）

def get_brand_occur_rate(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['brand','type']]

    

    all_act = df_b['type'].count()

    df_b = df_b.groupby('brand').size().to_frame().reset_index()

    df_b.columns = ['brand', 'type_num']

    df_b['brand_occur_rate'] = df_b['type_num'] / all_act

    brand_occur_rate = df_b[['brand', 'brand_occur_rate']]

    

    return brand_occur_rate 
#各品牌用户复购率brand_repurchase_rate（data用action）

def get_brand_repurchase_rate(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['brand', 'user_id', 'type']]

    

    df_b = pd.get_dummies(df_b, columns=['type'])

    df_b = df_b.groupby(['brand', 'user_id']).sum().reset_index()

    df_b = df_b[df_b['type_4'] >= 2]

    df_b = df_b.groupby('brand').size().to_frame().reset_index()

    df_b.columns = ['brand', 'repurchase_user_num']

    

    df_b = pd.merge(df_b, get_brand_buy_user_num(test_end_date, test_period, data), on='brand')

    df_b['brand_repurchase_rate'] = df_b['repurchase_user_num'] / df_b['brand_buy_user_num']

    

    brand_repurchase_rate = df_b[['brand', 'brand_repurchase_rate']]

    

    return brand_repurchase_rate
#各用户商品数user_pro_num（data用action）

def get_user_pro_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[data['time'] < test_start_date][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])

    

    df_u = df_u.groupby('user_id').size().to_frame().reset_index()

    df_u.columns = ['user_id', 'user_pro_num']

    user_pro_num = df_u

    

    return user_pro_num
### 各用户品牌数user_brand_num（data用action）

def get_user_brand_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[data['time'] < test_start_date][['user_id', 'brand']].drop_duplicates(['user_id', 'brand'])

    

    df_u = df_u.groupby('user_id').size().to_frame().reset_index()

    df_u.columns = ['user_id', 'user_brand_num']

    user_brand_num = df_u

    

    return user_brand_num
#各用户活动天数user_active_days（data用action_date）

def get_user_active_days(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[data['date'] < test_start_date][['user_id', 'date']].drop_duplicates(['user_id', 'date'])

   

    df_u = df_u.groupby('user_id').size().to_frame().reset_index()

    df_u.columns = ['user_id', 'user_active_days']

    user_active_days = df_u

    

    return user_active_days
#用户最后活动距今时间user_finallact_tillnow（data用action_date）

def get_user_finallact_tillnow(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[data['date'] < test_start_date][['user_id', 'date']].drop_duplicates(['user_id', 'date'])



    df_u = df_u.groupby('user_id', group_keys=False).apply(lambda x: x.sort_values(by=['date'], ascending=False))

    df_u = df_u.groupby('user_id').max().reset_index()

    df_u['user_finallact_tillnow'] = test_start_date - df_u['date']

    del df_u['date']

    df_u['user_finallact_tillnow'] = df_u['user_finallact_tillnow'].map(lambda x: x.days)

    df_u['user_finallact_tillnow'] = df_u['user_finallact_tillnow'].astype('int')

    user_finallact_tillnow = df_u

    

    return user_finallact_tillnow
#用户最后购买距今时间user_finallbuy_tillnow（data用actione_date）

def get_user_finallbuy_tillnow(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[(data['date'] < test_start_date) & (data['type'] == 4)][['user_id', 'date']].drop_duplicates(['user_id', 'date'])



    df_u = df_u.groupby('user_id', group_keys=False).apply(lambda x: x.sort_values(by=['date'], ascending=False))

    df_u = df_u.groupby('user_id').max().reset_index()

    df_u['user_finallbuy_tillnow'] = test_start_date - df_u['date']

    del df_u['date']

    df_u['user_finallbuy_tillnow'] = df_u['user_finallbuy_tillnow'].map(lambda x: x.days)

    df_u['user_finallbuy_tillnow'] = df_u['user_finallbuy_tillnow'].astype('int')

    user_finallbuy_tillnow = df_u

    

    return user_finallbuy_tillnow
#各用户购买前浏览次数user_liulan_num_before_buy（data用action）

def get_user_liulan_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['user_id', 'type', 'time']]

    

    #计算总浏览次数

    df_liulan = df_b[df_b['type'] == 1]

    del df_liulan['type']

    df_liulan = df_liulan.groupby('user_id', as_index=False).count()

    df_liulan.columns = ['user_id', 'liulan_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('user_id', as_index=False).count()

    df_buy.columns = ['user_id', 'buy_num']

    

    #计算购买前平均浏览次数

    df_b = pd.merge(df_liulan, df_buy, on='user_id')

    df_b['user_liulan_num_before_buy'] = df_b['liulan_num'] / df_b['buy_num']

    user_liulan_num_before_buy = df_b[['user_id', 'user_liulan_num_before_buy']]

    

    return user_liulan_num_before_buy
#各用户购买前关注次数user_guanzhu_num_before_buy（data用action）

def get_user_guanzhu_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['user_id', 'type', 'time']]

    

    #计算总关注次数

    df_guanzhu = df_b[df_b['type'] == 5]

    del df_guanzhu['type']

    df_guanzhu = df_guanzhu.groupby('user_id', as_index=False).count()

    df_guanzhu.columns = ['user_id', 'guanzhu_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('user_id', as_index=False).count()

    df_buy.columns = ['user_id', 'buy_num']

    

    #计算购买前平均关注次数

    df_b = pd.merge(df_guanzhu, df_buy, on='user_id')

    df_b['user_guanzhu_num_before_buy'] = df_b['guanzhu_num'] / df_b['buy_num']

    user_guanzhu_num_before_buy = df_b[['user_id', 'user_guanzhu_num_before_buy']]

    

    return user_guanzhu_num_before_buy
#各用户购买前加购（加入购物车）次数user_jiagou_num_before_buy（data用action）

def get_user_jiagou_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['user_id', 'type', 'time']]

    

    #计算总加购次数

    df_jiagou = df_b[df_b['type'] == 2]

    del df_jiagou['type']

    df_jiagou = df_jiagou.groupby('user_id', as_index=False).count()

    df_jiagou.columns = ['user_id', 'jiagou_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('user_id', as_index=False).count()

    df_buy.columns = ['user_id', 'buy_num']

    

    #计算购买前平均加购次数

    df_b = pd.merge(df_jiagou, df_buy, on='user_id')

    df_b['user_jiagou_num_before_buy'] = df_b['jiagou_num'] / df_b['buy_num']

    user_jiagou_num_before_buy = df_b[['user_id', 'user_jiagou_num_before_buy']]

    

    return user_jiagou_num_before_buy
#各用户预测期前3天行为占总行为数比例user_act_before_pre_3（data用action_date）

def get_user_act_before_pre_3(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['date'] < test_start_date][['user_id', 'date']]

    

    #总行为数

    df_allact = df_b.groupby('user_id', as_index=False).count()

    df_allact.columns = ['user_id', 'allact_num']

    

    #前三天行为数

    day3 = test_start_date - timedelta(3)

    df_before = data[(data['date'] < test_start_date) & (data['date'] >= day3)][['user_id', 'date']]

    df_before = df_before.groupby('user_id', as_index=False).count()

    df_before.columns = ['user_id', 'act_before3_num']

    

    #计算比例

    df_b = pd.merge(df_allact, df_before, on='user_id')

    df_b['user_act_before_pre_3'] = df_b['act_before3_num'] / df_b['allact_num']

    user_act_before_pre_3 = df_b[['user_id', 'user_act_before_pre_3']]

    

    return user_act_before_pre_3
#各商品用户数pro_user_num（data用action）

def get_pro_user_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_u = data[data['time'] < test_start_date][['user_id', 'sku_id']].drop_duplicates(['user_id', 'sku_id'])

    

    df_u = df_u.groupby('sku_id').size().to_frame().reset_index()

    df_u.columns = ['sku_id', 'pro_user_num']

    pro_user_num = df_u

    

    return pro_user_num
#各商品购买用户数pro_buy_user_num（data用action）

def get_pro_buy_user_num(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[(data['time'] < test_start_date) & (data['type'] == 4)][['sku_id','user_id']].drop_duplicates(['sku_id', 'user_id'])

    

    df_b = df_b.groupby('sku_id').size().to_frame().reset_index()

    df_b.columns = ['sku_id', 'pro_buy_user_num']

    pro_buy_user_num = df_b

    

    return pro_buy_user_num
#各品牌购买率pro_buy_user_rate（data用action）

def get_pro_buy_user_rate(test_end_date, test_period, data):

    

    df_b = pd.merge(get_pro_buy_user_num(test_end_date, test_period, data), get_pro_user_num(test_end_date, test_period, data), on='sku_id')

    df_b['pro_buy_user_rate'] = df_b['pro_buy_user_num'] / df_b['pro_user_num']

    df_b = df_b[['sku_id', 'pro_buy_user_rate']]

    pro_buy_user_rate = df_b

    

    return pro_buy_user_rate
#各商品出现频次占比pro_occur_rate（data用action）

def get_pro_occur_rate(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['sku_id','type']]

    

    all_act = df_b['type'].count()

    df_b = df_b.groupby('sku_id').size().to_frame().reset_index()

    df_b.columns = ['sku_id', 'type_num']

    df_b['pro_occur_rate'] = df_b['type_num'] / all_act

    pro_occur_rate = df_b[['sku_id', 'pro_occur_rate']]

    

    return pro_occur_rate 
#各商品用户复购率pro_repurchase_rate（data用action）

def get_pro_repurchase_rate(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['sku_id', 'user_id', 'type']]

    

    df_b = pd.get_dummies(df_b, columns=['type'])

    df_b = df_b.groupby(['sku_id', 'user_id']).sum().reset_index()

    df_b = df_b[df_b['type_4'] >= 2]

    df_b = df_b.groupby('sku_id').size().to_frame().reset_index()

    df_b.columns = ['sku_id', 'repurchase_user_num']

    

    df_b = pd.merge(df_b, get_pro_buy_user_num(test_end_date, test_period, data), on='sku_id')

    df_b['pro_repurchase_rate'] = df_b['repurchase_user_num'] / df_b['pro_buy_user_num']

    

    pro_repurchase_rate = df_b[['sku_id', 'pro_repurchase_rate']]

    

    return pro_repurchase_rate
#各商品购买前浏览次数pro_liulan_num_before_buy（data用action）

def get_pro_liulan_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['sku_id', 'type', 'time']]

    

    #计算总浏览次数

    df_liulan = df_b[df_b['type'] == 1]

    del df_liulan['type']

    df_liulan = df_liulan.groupby('sku_id', as_index=False).count()

    df_liulan.columns = ['sku_id', 'liulan_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('sku_id', as_index=False).count()

    df_buy.columns = ['sku_id', 'buy_num']

    

    #计算购买前平均浏览次数

    df_b = pd.merge(df_liulan, df_buy, on='sku_id')

    df_b['pro_liulan_num_before_buy'] = df_b['liulan_num'] / df_b['buy_num']

    pro_liulan_num_before_buy = df_b[['sku_id', 'pro_liulan_num_before_buy']]

    

    return pro_liulan_num_before_buy
#各商品购买前关注次数pro_guanzhu_num_before_buy（data用action）

def get_pro_guanzhu_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['sku_id', 'type', 'time']]

    

    #计算总关注次数

    df_guanzhu = df_b[df_b['type'] == 5]

    del df_guanzhu['type']

    df_guanzhu = df_guanzhu.groupby('sku_id', as_index=False).count()

    df_guanzhu.columns = ['sku_id', 'guanzhu_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('sku_id', as_index=False).count()

    df_buy.columns = ['sku_id', 'buy_num']

    

    #计算购买前平均关注次数

    df_b = pd.merge(df_guanzhu, df_buy, on='sku_id')

    df_b['pro_guanzhu_num_before_buy'] = df_b['guanzhu_num'] / df_b['buy_num']

    pro_guanzhu_num_before_buy = df_b[['sku_id', 'pro_guanzhu_num_before_buy']]

    

    return pro_guanzhu_num_before_buy
#各商品购买前加购（加入购物车）次数pro_jiagou_num_before_buy（data用action）

def get_pro_jiagou_num_before_buy(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['time'] < test_start_date][['sku_id', 'type', 'time']]

    

    #计算总加购次数

    df_jiagou = df_b[df_b['type'] == 2]

    del df_jiagou['type']

    df_jiagou = df_jiagou.groupby('sku_id', as_index=False).count()

    df_jiagou.columns = ['sku_id', 'jiagou_num']

    

    #计算购买次数

    df_buy = df_b[df_b['type'] == 4]

    del df_buy['type']

    df_buy = df_buy.groupby('sku_id', as_index=False).count()

    df_buy.columns = ['sku_id', 'buy_num']

    

    #计算购买前平均加购次数

    df_b = pd.merge(df_jiagou, df_buy, on='sku_id')

    df_b['pro_jiagou_num_before_buy'] = df_b['jiagou_num'] / df_b['buy_num']

    pro_jiagou_num_before_buy = df_b[['sku_id', 'pro_jiagou_num_before_buy']]

    

    return pro_jiagou_num_before_buy
#各商品预测期前3天行为占总行为数比例pro_act_before_pre_3（data用action_date）

def get_pro_act_before_pre_3(test_end_date, test_period, data):

    

    test_start_date = datetime.strptime(test_end_date, '%Y-%m-%d') - timedelta(test_period - 1)

    df_b = data[data['date'] < test_start_date][['sku_id', 'date']]

    

    #总行为数

    df_allact = df_b.groupby('sku_id', as_index=False).count()

    df_allact.columns = ['sku_id', 'allact_num']

    

    #前三天行为数

    day3 = test_start_date - timedelta(3)

    df_before = data[(data['date'] < test_start_date) & (data['date'] >= day3)][['sku_id', 'date']]

    df_before = df_before.groupby('sku_id', as_index=False).count()

    df_before.columns = ['sku_id', 'act_before3_num']

    

    #计算比例

    df_b = pd.merge(df_allact, df_before, on='sku_id')

    df_b['pro_act_before_pre_3'] = df_b['act_before3_num'] / df_b['allact_num']

    pro_act_before_pre_3 = df_b[['sku_id', 'pro_act_before_pre_3']]

    

    return pro_act_before_pre_3 
#获取索引

dataset = get_feature_index('2016-04-15', 5, action)



#获取特征

test_end_date = '2016-04-15'

test_period = 5



data = user



dataset = pd.merge(dataset, get_age(data), on='user_id', how='left')

dataset = pd.merge(dataset, get_sex(data), on='user_id', how='left')

dataset = pd.merge(dataset, get_user_lv(data), on='user_id', how='left')

dataset = pd.merge(dataset, get_user_reg_tillnow(test_end_date, test_period, data), on='user_id', how='left')



data = product



dataset = pd.merge(dataset, get_pro_atrr(data), on='sku_id', how='left')

dataset = pd.merge(dataset, get_brand(data), on='sku_id', how='left')



data = comment 



dataset = pd.merge(dataset, get_comment_num(data), on='sku_id', how='left')

dataset = pd.merge(dataset, get_has_bad_comment(data), on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_bad_comment_rate(data), on='sku_id', how='left').fillna(0)
data = action 
#获取特征



dataset = pd.merge(dataset, get_user_allact(test_end_date, test_period, data), on='user_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_user_act_num_(test_end_date, test_period, data), on='user_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_allact(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_act_num_(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)
#获取特征



dataset = pd.merge(dataset, get_up_allact(test_end_date, test_period, data), on=['user_id', 'sku_id'], how='left').fillna(0)

dataset = pd.merge(dataset, get_up_act_num_(test_end_date, test_period, data), on=['user_id', 'sku_id'], how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_allact(test_end_date, test_period, data), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_act_num_(test_end_date, test_period, data), on='brand', how='left').fillna(0)
#这里先保存一份不含新开发变量的数据集

dataset1 = dataset.copy()
#继续获取特征

dataset = pd.merge(dataset, get_all_user_slidewindow_fea(test_end_date, test_period, data), on='user_id', how='left').fillna(0)
#继续获取特征

dataset = pd.merge(dataset, get_all_pro_slidewindow_fea(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)
#继续获取特征

dataset = pd.merge(dataset, get_all_up_slidewindow_fea(test_end_date, test_period, data), on=['user_id', 'sku_id'], how='left').fillna(0)
#继续获取特征



dataset = pd.merge(dataset, get_brand_comment_num(), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_bad_comment_rate(), on='brand').fillna(0)

dataset = pd.merge(dataset, get_brand_buy_rate(test_end_date, test_period, data), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_user_num(test_end_date, test_period, data), on='brand', how='left').fillna(0)
#继续获取特征



dataset = pd.merge(dataset, get_brand_buy_user_num(test_end_date, test_period, data), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_buy_user_rate(test_end_date, test_period, data), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_occur_rate(test_end_date, test_period, data), on='brand', how='left').fillna(0)

dataset = pd.merge(dataset, get_brand_repurchase_rate(test_end_date, test_period, data), on='brand', how='left').fillna(0)
#继续获取特征



dataset = pd.merge(dataset, get_user_pro_num(test_end_date, test_period, data), on='user_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_user_brand_num(test_end_date, test_period, data), on='user_id', how='left').fillna(0)
#继续获取特征



dataset = pd.merge(dataset, get_pro_user_num(test_end_date, test_period, data) , on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_buy_user_num(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_buy_user_rate(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_occur_rate(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_repurchase_rate(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)
data = action_date
#继续获取特征

dataset = pd.merge(dataset, get_user_active_days(test_end_date, test_period, data), on='user_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_user_act_before_pre_3(test_end_date, test_period, data), on='user_id', how='left').fillna(0)

dataset = pd.merge(dataset, get_pro_act_before_pre_3(test_end_date, test_period, data), on='sku_id', how='left').fillna(0)
#继续获取特征

dataset = pd.merge(dataset, get_user_finallact_tillnow(test_end_date, test_period, data), on='user_id', how='left')

dataset = pd.merge(dataset, get_user_finallbuy_tillnow(test_end_date, test_period, data), on='user_id', how='left')
data = action
#继续获取特征



dataset = pd.merge(dataset, get_user_liulan_num_before_buy(test_end_date, test_period, data), on='user_id', how='left')

dataset = pd.merge(dataset, get_user_guanzhu_num_before_buy(test_end_date, test_period, data), on='user_id', how='left')

dataset = pd.merge(dataset, get_user_jiagou_num_before_buy(test_end_date, test_period, data), on='user_id', how='left')
#继续获取特征



dataset = pd.merge(dataset, get_pro_liulan_num_before_buy(test_end_date, test_period, data), on='sku_id', how='left')

dataset = pd.merge(dataset, get_pro_guanzhu_num_before_buy(test_end_date, test_period, data), on='sku_id', how='left')

dataset = pd.merge(dataset, get_pro_jiagou_num_before_buy(test_end_date, test_period, data), on='sku_id', how='left')
#添加标签值label



dataset = pd.merge(dataset, get_label(test_end_date, test_period), on=['user_id', 'sku_id'], how='left').fillna({'label':0})
#添加标签值label



dataset1 = pd.merge(dataset1, get_label(test_end_date, test_period), on=['user_id', 'sku_id'], how='left').fillna({'label':0})
dataset
#查看标签值分布情况



plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(8, 6))

plt.subplot(1, 2, 1)

sns.countplot('label', data=dataset, palette='hls')
#正负样本比例

positive = dataset[dataset['label'] == 1]['label'].count()

negetive = dataset[dataset['label'] == 0]['label'].count()



positive_rate = positive / (positive + negetive)



print('正样本的数量为：', positive)

print('负样本的数量为：', negetive)

print('正样本占所有样本比例：', positive_rate)
#随机采集8000个负样本

finall_data = dataset[dataset['label'] == 0]

finall_data = finall_data.sample(n=8000, random_state=0)

    

#与正样本拼接在一起，构成我们的最终数据集

finall_data = pd.concat([finall_data, dataset[dataset['label'] == 1]], ignore_index=True)

    

#删除不必要的列

del finall_data['user_id']

del finall_data['sku_id']

finall_data.fillna(100, inplace=True)
#对数据集进行洗牌

finall_data_shuffle = finall_data.sample(frac=1, random_state=2).reset_index(drop=True)
#构造特征集与标签集

dataset_feature = finall_data_shuffle.drop(['label'], axis=1)

dataset_target = finall_data_shuffle['label']
 #对inf值和nan值进行替换

dataset_feature = dataset_feature.replace([np.inf, -np.inf], np.nan)

dataset_feature.fillna(0, inplace=True)
#切分训练集与测试集

X_train, X_test, y_train, y_test = train_test_split(dataset_feature, dataset_target, test_size=0.3, random_state=4)
#SMOTE过采样



sm = SMOTE(sampling_strategy=0.25, random_state=6)

X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

#基本模型

lgb = lgbm.LGBMClassifier(objective='binary')
#模型拟合训练集

lgb.fit(X_train_sm, y_train_sm)



#在测试集上进行预测

y_pre = lgb.predict(X_test)



#模型在测试集上的Fbeta得分

f_score = fbeta_score(y_test, y_pre, beta=2)



#auc分值

y_proba = lgb.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_proba)



print('原始模型在测试集上的fbeta(beta=2)得分为：', f_score)

print('原始模型在测试集上的auc得分为：', auc)
## 查看特征重要性

importance = lgb.feature_importances_

importance_table = {'fea':[], 'importance':[]}



for i in list(range(len(importance))):

    importance_table['fea'].append(finall_data.columns[i])

    importance_table['importance'].append(importance[i])

              

importance_table = pd.DataFrame(importance_table).sort_values(by='importance', ascending=False)



pd.options.display.max_rows = None

importance_table
#删除重要性为0的特征

get_columns = list(importance_table[importance_table['importance'] > 0]['fea'])

dataset_del0 = dataset_feature[get_columns]



#删除重要性为10及以下的特征

get_columns = list(importance_table[importance_table['importance'] > 10]['fea'])

dataset_del10 = dataset_feature[get_columns]



#删除重要性为20及以下的特征

get_columns = list(importance_table[importance_table['importance'] > 20]['fea'])

dataset_del20 = dataset_feature[get_columns]



#删除重要性为30及以下的特征

get_columns = list(importance_table[importance_table['importance'] > 30]['fea'])

dataset_del30 = dataset_feature[get_columns]



#删除重要性为40及以下的特征

get_columns = list(importance_table[importance_table['importance'] > 40]['fea'])

dataset_del40 = dataset_feature[get_columns]
%%time



#num_leaves / max_depth网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=200, )



para_grid = {'num_leaves': [25, 30, 35, 40, 45, 50, 60], 'max_depth': [5, 6, 7, 8, 9]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
%%time



#min_child_samples/min_split_gain网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=200, num_leaves=60, max_depth=9)



para_grid = {'min_child_samples': [15, 20, 30, 40, 50], 'min_split_gain': [0, 0.05, 0.1, 0.2, 0.5, 2]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
%%time



#subsample/colsample_bytree网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=200, num_leaves=60, max_depth=9, min_child_samples=15, min_split_gain=0.05)



para_grid = {'subsample': [0.6, 0.7, 0.8, 0.9, 1], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
%%time



#reg_lambda网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=200, 

                          num_leaves=60, max_depth=9, min_child_samples=15, min_split_gain=0.05, 

                         colsample_bytree=1, subsample=0.6)



para_grid = {'reg_lambda': [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
%%time



#n_estimators网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', learning_rate=0.1, 

                          num_leaves=60, max_depth=9, min_child_samples=15, min_split_gain=0.05, 

                         colsample_bytree=1, subsample=0.6, reg_lambda=0.07)



para_grid = {'n_estimators': [50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
%%time



#learning_rate网格搜索



lgb = lgbm.LGBMClassifier(objective='binary', n_estimators=50, 

                          num_leaves=60, max_depth=9, min_child_samples=15, min_split_gain=0.05, 

                         colsample_bytree=1, subsample=0.6, reg_lambda=0.07)



para_grid = {'learning_rate': [0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.1, 0.2]}



f2_score = make_scorer(fbeta_score, beta=2)



gridsearch = GridSearchCV(lgb, param_grid=para_grid, cv=3, scoring=f2_score)



gridsearch.fit(X_train_sm, y_train_sm)
gridsearch.best_params_, gridsearch.best_score_
lgb2 = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=60, max_depth=9, min_child_samples=15, min_split_gain=0.05, 

                         colsample_bytree=1, subsample=0.6, reg_lambda=0.07)
#模型拟合训练集

lgb2.fit(X_train_sm, y_train_sm)



#在测试集上进行预测

y_pre = lgb2.predict(X_test)



#模型在测试集上的Fbeta得分

f_score = fbeta_score(y_test, y_pre, beta=2)



#auc分值

y_proba = lgb2.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_proba)



print('模型在测试集上的fbeta(beta=2)得分为：', f_score)

print('模型在测试集上的auc得分为：', auc)
#模型在训练集上的得分



y_pre = lgb2.predict(X_train_sm)



f_score = fbeta_score(y_train_sm, y_pre, beta=2)

y_proba = lgb2.predict_proba(X_train_sm)[:,1]

auc = roc_auc_score(y_train_sm, y_proba)



print('模型在训练集上的fbeta(beta=2)得分为：', f_score)

print('模型在训练集上的auc得分为：', auc)
#调整num_leaves



leaves_table = {'num_leaves':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (25, 30, 35, 40, 45, 50, 55, 60):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=i, reg_lambda=0.07)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    leaves_table['num_leaves'].append(i)

    leaves_table['fscore_test'].append(fscore_test)

    leaves_table['fscore_train'].append(fscore_train)

    leaves_table['auc_test'].append(auc_test) 

    leaves_table['auc_train'].append(auc_train) 
pd.DataFrame(leaves_table)
#调整max_depth



max_depth_table = {'max_depth':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (3, 4, 5, 6, 7, 8, 9):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, reg_lambda=0.07, max_depth=i)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    max_depth_table['max_depth'].append(i)

    max_depth_table['fscore_test'].append(fscore_test)

    max_depth_table['fscore_train'].append(fscore_train)

    max_depth_table['auc_test'].append(auc_test) 

    max_depth_table['auc_train'].append(auc_train) 
pd.DataFrame(max_depth_table)
#调整min_child_samples



min_child_samples_table = {'min_child_samples':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (15, 20, 30, 40, 50, 60, 80):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, reg_lambda=0.07, min_child_samples=i)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    min_child_samples_table['min_child_samples'].append(i)

    min_child_samples_table['fscore_test'].append(fscore_test)

    min_child_samples_table['fscore_train'].append(fscore_train)

    min_child_samples_table['auc_test'].append(auc_test) 

    min_child_samples_table['auc_train'].append(auc_train) 
pd.DataFrame(min_child_samples_table)
#调整min_split_gain



min_split_gain_table = {'min_split_gain':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (0.05, 0.1, 0.2, 0.5, 1, 2, 2.5, 3):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, reg_lambda=0.07, min_split_gain=i)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    min_split_gain_table['min_split_gain'].append(i)

    min_split_gain_table['fscore_test'].append(fscore_test)

    min_split_gain_table['fscore_train'].append(fscore_train)

    min_split_gain_table['auc_test'].append(auc_test) 

    min_split_gain_table['auc_train'].append(auc_train) 

    

pd.DataFrame(min_split_gain_table)
#调整colsample_bytree



colsample_bytree_table = {'colsample_bytree':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (0.5, 0.6, 0.7, 0.8, 0.9):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, reg_lambda=0.07, colsample_bytree=i)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    colsample_bytree_table['colsample_bytree'].append(i)

    colsample_bytree_table['fscore_test'].append(fscore_test)

    colsample_bytree_table['fscore_train'].append(fscore_train)

    colsample_bytree_table['auc_test'].append(auc_test) 

    colsample_bytree_table['auc_train'].append(auc_train) 

    

pd.DataFrame(colsample_bytree_table)
#调整reg_lambda



reg_lambda_table = {'reg_lambda':[], 'fscore_test':[], 'fscore_train':[], 'auc_test':[], 'auc_train':[]}



for i in (0.5, 0.6, 0.7, 1, 1.5, 2, 3):

    lgb_x = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, reg_lambda=i)

    

    lgb_x.fit(X_train_sm, y_train_sm)

    y_pre1 = lgb_x.predict(X_test)

    fscore_test = fbeta_score(y_test, y_pre1, beta=2)

    y_proba1 = lgb_x.predict_proba(X_test)[:,1]

    auc_test = roc_auc_score(y_test, y_proba1)

    

    y_pre2 = lgb_x.predict(X_train_sm)

    fscore_train = fbeta_score(y_train_sm, y_pre2, beta=2)

    y_proba2 = lgb_x.predict_proba(X_train_sm)[:,1]

    auc_train = roc_auc_score(y_train_sm, y_proba2)

    

    reg_lambda_table['reg_lambda'].append(i)

    reg_lambda_table['fscore_test'].append(fscore_test)

    reg_lambda_table['fscore_train'].append(fscore_train)

    reg_lambda_table['auc_test'].append(auc_test) 

    reg_lambda_table['auc_train'].append(auc_train) 

    

pd.DataFrame(reg_lambda_table)
lgb3 = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, max_depth=6, min_child_samples=60, min_split_gain=3, 

                         colsample_bytree=0.7, subsample=0.6, reg_lambda=3)
#模型拟合训练集

lgb3.fit(X_train_sm, y_train_sm)



#在测试集上进行预测

y_pre = lgb3.predict(X_test)



#模型在测试集上的Fbeta得分

f_score = fbeta_score(y_test, y_pre, beta=2)



#auc分值

y_proba = lgb3.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, y_proba)



print('模型在测试集上的fbeta(beta=2)得分为：', f_score)

print('模型在测试集上的auc得分为：', auc)
#模型在训练集上的得分



y_pre = lgb3.predict(X_train_sm)



f_score = fbeta_score(y_train_sm, y_pre, beta=2)

y_proba = lgb3.predict_proba(X_train_sm)[:,1]

auc = roc_auc_score(y_train_sm, y_proba)



print('模型在训练集上的fbeta(beta=2)得分为：', f_score)

print('模型在训练集上的auc得分为：', auc)
#根据特征重要性做删减的特征选择方案：dataset_del10, dataset_del20, dataset_del30



X_train1, X_test1, y_train1, y_test1 = train_test_split(dataset_del30, dataset_target, test_size=0.3, random_state=4)



sm = SMOTE(sampling_strategy=0.25, random_state=6)

X_train_sm1, y_train_sm1 = sm.fit_sample(X_train1, y_train1)
lgb3 = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, max_depth=6, min_child_samples=60, min_split_gain=3, 

                         colsample_bytree=0.7, subsample=0.6, reg_lambda=3)
#模型拟合训练集

lgb3.fit(X_train_sm1, y_train_sm1)



#在测试集上进行预测

y_pre = lgb3.predict(X_test1)



#模型在测试集上的Fbeta得分

f_score = fbeta_score(y_test1, y_pre, beta=2)



#auc分值

y_proba = lgb3.predict_proba(X_test1)[:,1]

auc = roc_auc_score(y_test1, y_proba)



print('模型在测试集上的fbeta(beta=2)得分为：', f_score)

print('模型在测试集上的auc得分为：', auc)
#模型在训练集上的得分



y_pre = lgb3.predict(X_train_sm1)



f_score = fbeta_score(y_train_sm1, y_pre, beta=2)

y_proba = lgb3.predict_proba(X_train_sm1)[:,1]

auc = roc_auc_score(y_train_sm1, y_proba)



print('模型在训练集上的fbeta(beta=2)得分为：', f_score)

print('模型在训练集上的auc得分为：', auc)
lgb3 = lgbm.LGBMClassifier(objective='binary', n_estimators=50, learning_rate=0.05,

                          num_leaves=30, max_depth=6, min_child_samples=60, min_split_gain=3, 

                         colsample_bytree=0.7, subsample=0.6, reg_lambda=3)
lgb3.fit(X_train_sm, y_train_sm)
y_proba = lgb3.predict_proba(X_test, raw_score=True)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label='recall')

    

    plt.xlabel('thresholds', fontsize=15)

    plt.ylim([0,1])

    plt.legend(fontsize=15)
plt.figure(figsize=(8,5))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#尝试把预测正负类别的阈值修改为-1~-1.5



y_proba2 = y_proba.copy()

y_proba2[y_proba2 > -1.25] = 1

y_proba2[y_proba2 != 1] = 0

y_pre_adjust = y_proba2
f_score2 = fbeta_score(y_test, y_pre_adjust, beta=2)



print('修改阈值后，模型在测试集上的fbeta(beta=2)得分为：', f_score2)