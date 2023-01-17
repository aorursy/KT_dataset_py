import pandas as pd

import warnings

import os

import numpy as np

import folium

from folium import plugins

from sklearn.preprocessing import MinMaxScaler

from copy import deepcopy

import math

import time

from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from  sklearn.cluster import DBSCAN

from sklearn import mixture

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
# KFCM算法的指数

m = 1.1

n = 5
all_data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
all_data.head()
all_data = all_data[['eventid', 'iyear', 'imonth', 'iday', 'extended', 'country', 'region', 'vicinity', 'latitude', 'longitude', 'specificity', 'crit1', 

                    'crit2', 'crit3', 'doubtterr', 'multiple', 'attacktype1', 'success', 'suicide', 'weaptype1', 'targtype1', 'nkill', 'nkillter', 

                    'nwound', 'nwoundte', 'property', 'propextent', 'ishostkid', 'ransom', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']]
all_data = all_data.dropna(subset=['latitude'])

all_data = all_data.dropna(subset=['longitude'])
all_data['doubtterr'].fillna(value=-9, inplace=True)

all_data['multiple'].fillna(value=0, inplace=True)

all_data['nkill'].fillna(value=0, inplace=True)

all_data['nkillter'].fillna(value=0, inplace=True)

all_data['nwound'].fillna(value=0, inplace=True)

all_data['nwoundte'].fillna(value=0, inplace=True)

all_data['propextent'].fillna(value=4, inplace=True)

all_data['ishostkid'].fillna(value=-9, inplace=True)

all_data['ransom'].fillna(value=-9, inplace=True)
all_data.shape
all_data['specificity'] = pd.to_numeric(all_data['specificity'], downcast='integer')

all_data['doubtterr'] = pd.to_numeric(all_data['doubtterr'], downcast='integer')

all_data['multiple'] = pd.to_numeric(all_data['multiple'], downcast='integer')

all_data['nkill'] = pd.to_numeric(all_data['nkill'], downcast='integer')

all_data['nkillter'] = pd.to_numeric(all_data['nkillter'], downcast='integer')

all_data['nwound'] = pd.to_numeric(all_data['nwound'], downcast='integer')

all_data['nwoundte'] = pd.to_numeric(all_data['nwoundte'], downcast='integer')

all_data['ishostkid'] = pd.to_numeric(all_data['ishostkid'], downcast='integer')

all_data['ransom'] = pd.to_numeric(all_data['ransom'], downcast='integer')
print('all_data占据内存约: {:.2f} GB'.format(all_data.memory_usage().sum()/ (1024**3)))
id_ = list(all_data['eventid'])

all_data.drop('eventid',axis=1, inplace=True)
all_data = all_data[['crit1', 'crit2', 'crit3', 'doubtterr', 'multiple', 'attacktype1', 'success', 'suicide', 'weaptype1', 'targtype1', 

                     'nkill', 'nkillter', 'nwound', 'nwoundte', 'property', 'propextent', 'ishostkid', 'ransom']]
"""

author: zhenyu wu

time: 2019/04/07 21:48

function: 对数据样本进行归一化处理，将区间放缩到某个范围内

   在MinMaxScaler中是给定了一个明确的最大值与最小值。它的计算公式如下: X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) | X_scaled = X_std / (max - min) + min 每个特征中的最小值变成了MIN，最大值变成了MAX。这样做是为了在熵权法取对数过程中避免向下溢出。

params: 

    data: 数据集

    MIN: 下限

    MAX: 上限

return:

    data_norm: 归一化后的数据集

"""

def normalization(data, MIN=0.002, MAX=0.998):

    min_max_scaler = MinMaxScaler(feature_range=(MIN, MAX))

    data_norm = min_max_scaler.fit_transform(data) 

    data_norm = pd.DataFrame(data_norm)

    data_norm.columns = data.columns

    return data_norm
all_data_norm = normalization(all_data, MIN=0.002, MAX=0.998)
"""

author: zhenyu wu

time: 2019/04/07 21:47

function: 熵权法确定特征权重

params: 

    data_norm: 数据集

    threshold: 阈值

return:

    Entropy: 特征的熵值

    difference_coefficient: 特征的差异系数

    important_features: 重要特征名称序列

    entropy_weight: 特征熵权

    overall: 样本的重要性分数

"""

def entropy(data_norm, data_id, threshold=0.8):

    feature_weight = pd.DataFrame({'temp': list(np.zeros(len(data_norm)))})

    for i in data_norm.columns:                     # 计算特征比重

        Sum = data_norm[i].sum()

        temp = data_norm[i]/Sum

        feature_weight[i] = temp

    feature_weight.drop('temp', axis=1, inplace=True)

    Entropy = {}

    for i in feature_weight.columns:                # 计算每一项指标的熵值

        Sum = 0

        column = list(deepcopy(feature_weight[i]))

        for j in range(len(feature_weight)):

            Sum += column[j] * math.log(column[j])

        Entropy[i] = (-1 / (math.log(len(feature_weight)))) * Sum

    important_features = []

    for key, value in Entropy.items():

        # if value <= threshold:                      # 提取重要特征进行分析,控制此处的阈值

        important_features.append(key)

    difference_coefficient = {}

    for i in important_features:                    # 计算差异系数

        difference_coefficient[i] = 1 - Entropy[i]

    # print('特征的差异系数为：\n', difference_coefficient)

    Diff_sum = sum(list(difference_coefficient.values()))

    entropy_weight = {}

    for i in important_features:                    # 计算熵权

        entropy_weight[i] = difference_coefficient[i] / Diff_sum

    # print('特征的熵权为：\n', entropy_weight)

    feature_weight = feature_weight[important_features]

    feature_weight = np.mat(feature_weight)

    weight = np.array(list(entropy_weight.values()))

    overall_merit = weight * (feature_weight.T)     # 计算各个评价对象的综合评价值

    overall_merit = overall_merit.T

    overall_merit = np.array(overall_merit)

    overall_list = []

    for i in range(len(feature_weight)):

        overall_list.append(overall_merit[i][0])

    overall = pd.DataFrame({'eventid': data_id, 'overall': overall_list})

    overall = overall.sort_values(by=['overall'], ascending=(False))

    overall.index = list(np.arange(len(data_norm)))

    data_norm = data_norm[important_features]

    overall = overall.sort_values(by=['eventid'], ascending=(True))

    overall.index = list(np.arange(len(data_norm)))

    feature_names = list(entropy_weight.keys())

    entropy_weight = list(entropy_weight.values())

    norm_entropy_weight = []

    for sub_weight in entropy_weight:

        norm_entropy_weight.append(sub_weight/sum(entropy_weight))

    entropy_weight = dict(zip(feature_names, norm_entropy_weight))

    return entropy_weight, overall
entropy_weight, overall = entropy(all_data_norm, id_, threshold=1)
print(entropy_weight)
"""

author: zhenyu wu

time: 2019/04/09 16:51

function: 利用模糊层次分析法确定指标的权重

params: 

    r_1_n: r11~r1n的模糊关系

return:

    B: FAHP的权重计算结果

"""

def FAHP(r_1_n):

    R = np.zeros((len(r_1_n), len(r_1_n)))

    E = np.zeros((len(r_1_n), len(r_1_n)))

    B = np.zeros((len(r_1_n)))

    R[0] = r_1_n

    col_1 = 1-R[0]

    for i in range(len(r_1_n)):

        R[i] = R[0]-(R[0][0]-col_1[i])

    e = R.sum(axis=1)

    for i in range(len(e)):

        for j in range(len(e)):

            E[i][j] = (e[i]-e[j])/(2*len(e))+0.5

    e = E.sum(axis=0)

    for i in range(len(e)):

        B[i] = (2*e[i]-1)/(len(e)*(len(e)-1))

    return B
"""

author: zhenyu wu

time: 2019/04/09 17:39

function: 利用模糊层次分析法确定所有指标权重

params: 

    feature_names: 特征名称

return:

    all_fAHP_weight: 所有指标的fAHP权重

"""

def fAHP_weight(feature_names):

    # 事件信息、攻击信息、目标/受害者信息、伤亡和后果

    r_1_n = [0.5, 0.8, 0.8, 0.8]

    level_1_fAHP_weight = FAHP(r_1_n)

    # 年、月、日、是否为持续事件

    # r_1_n = [0.5, 0.5, 0.5, 0.8]

    # level_2_date_fAHP_weight = FAHP(r_1_n)

    # level_2_date_fAHP_weight = list(level_2_date_fAHP_weight/sum(level_2_date_fAHP_weight)*level_1_fAHP_weight[0])

    # 国家、地区、附近地区、纬度、经度、地理编码特征

    # r_1_n = [0.5, 0.6, 0.7, 0.5, 0.5, 0.6]

    # level_2_loc_fAHP_weight = FAHP(r_1_n)

    # level_2_loc_fAHP_weight = list(level_2_loc_fAHP_weight/sum(level_2_loc_fAHP_weight)*level_1_fAHP_weight[1])

    # 入选标准1、入选标准2、入选标准3、疑似恐怖主义、事件组的一部分

    r_1_n = [0.5, 0.6, 0.7, 0.8, 0.8]

    level_2_std_fAHP_weight = FAHP(r_1_n)

    level_2_std_fAHP_weight = list(level_2_std_fAHP_weight/sum(level_2_std_fAHP_weight)*level_1_fAHP_weight[0])

    # 攻击类型、成功的攻击、自杀式袭击、武器类型

    r_1_n = [0.5, 0.7, 0.8, 0.8]

    level_2_att_fAHP_weight = FAHP(r_1_n)

    level_2_att_fAHP_weight = list(level_2_att_fAHP_weight/sum(level_2_att_fAHP_weight)*level_1_fAHP_weight[1])

    # 目标/受害者类型

    level_2_targ_fAHP_weight = [level_1_fAHP_weight[2]]

    # level_2_targ_fAHP_weight = [level_2_targ_fAHP_weight]

    # 死亡总数、凶手死亡人数、受伤总数、凶手受伤人数、财产损失、财产损害程度、人质或绑架的受害者、索要赎金

    r_1_n = [0.5, 0.5, 0.4, 0.4, 0.3, 0.4, 0.6, 0.6]

    level_2_kill_fAHP_weight = FAHP(r_1_n)

    level_2_kill_fAHP_weight = list(level_2_kill_fAHP_weight/sum(level_2_kill_fAHP_weight)*level_1_fAHP_weight[3])

    # 国际后勤、 国际的意识形态、 国际杂类、 国际‐以上任意一类

    # r_1_n = [0.5, 0.6, 0.5, 0.6]

    # level_2_word_fAHP_weight = FAHP(r_1_n)

    # level_2_word_fAHP_weight = list(level_2_word_fAHP_weight/sum(level_2_word_fAHP_weight)*level_1_fAHP_weight[6])

    all_fAHP_weight = level_2_std_fAHP_weight+level_2_att_fAHP_weight+level_2_targ_fAHP_weight+level_2_kill_fAHP_weight

    all_fAHP_weight = all_fAHP_weight/sum(all_fAHP_weight)

    AHP_weight = dict(zip(feature_names, all_fAHP_weight))

    return AHP_weight
fAHP_weight = fAHP_weight(all_data_norm.columns)
print(fAHP_weight)
"""

author: zhenyu wu

time: 2019/04/09 22:29

function: 利用基于博弈论组合赋权对fAHP以及熵权法的计算结果进行组合

params: 

    weight_1: 第一组权重

    weight_2: 第二组权重

return:

    final_weight: 最终结果

"""

def Game_Theory(weight_1, weight_2, feature_names):

    U = np.zeros((2, 2))

    Y = np.zeros((2, 1))

    u11 = sum(np.multiply(np.array(weight_1),np.array(weight_1)).tolist())

    u12 = sum(np.multiply(np.array(weight_1),np.array(weight_2)).tolist())

    u21 = sum(np.multiply(np.array(weight_2),np.array(weight_1)).tolist())

    u22 = sum(np.multiply(np.array(weight_2),np.array(weight_2)).tolist())

    U[0][0] = u11

    U[0][1] = u12

    U[1][0] = u21

    U[1][1] = u22

    Y[0][0] = u11

    Y[1][0] = u22

    U = np.mat(U)

    Y = np.mat(Y)

    alpha = U.I*Y

    final_alpha = []

    for sub_alpha in alpha:

        final_alpha.append(np.array(sub_alpha)[0][0])

    final_alpha = final_alpha/sum(final_alpha)

    new_weight_1 = [x*final_alpha[0] for x in weight_1]

    new_weight_2 = [x*final_alpha[1] for x in weight_2]

    final_weight = np.sum([new_weight_1, new_weight_2], axis=0)

    final_weight = dict(zip(feature_names, final_weight))

    print(final_alpha)

    return final_weight
final_weight = Game_Theory(list(entropy_weight.values()), list(fAHP_weight.values()), all_data_norm.columns)
"""

author: zhenyu wu

time: 2019/04/08 14:48

function: 对数据进行加权处理

params: 

    useful_feature: 加权前的样本

    entropy_weight: 样本权重

return:

    useful_feature: 加权后的样本

"""

def add_weight(useful_feature, entropy_weight):

    weight = {}

    feature = useful_feature.columns

    for i in feature:

        weight[i] = entropy_weight[i]

    Key = []

    Value = []

    for key, value in weight.items():

        Key.append(key)

        Value.append(value)

    Sum = sum(Value)

    Value2 = []

    for i in Value:

        Value2.append(i / Sum)

    weight = {}

    for i in range(len(Value)):

        weight[Key[i]] = Value2[i]

    for i in feature:

        useful_feature[i] = useful_feature[i] * weight[i]

    return useful_feature
all_data_norm = add_weight(all_data_norm, final_weight)
all_data_norm.head()
def Bayesian_Gaussian_Mixture_cluser(useful_feature, data_id, data_score, important_features, n, cv_type):

    model = mixture.BayesianGaussianMixture(n_components=n, covariance_type=cv_type, random_state=0, max_iter=10000)

    model.fit(useful_feature)

    labels = model.predict(useful_feature)

    score_sil = metrics.silhouette_score(useful_feature, labels, metric='euclidean')

    print("当聚为%d簇时，贝叶斯高斯混合模型轮廓系数Silhouette Coefficient为：%f" % (n, score_sil))             # 计算轮廓系数

    score_cal = metrics.calinski_harabaz_score(useful_feature, labels) 

    print("当聚为%d簇时，贝叶斯高斯混合模型轮廓系数Calinski-Harabaz Index为：%f" % (n, score_cal))

    score_dbi = metrics.davies_bouldin_score(useful_feature, labels)                                     # DBI值越小越好（说明分散程度低）

    print("当聚为%d簇时，贝叶斯高斯混合模型 Davies-Bouldin分数值为：%f" % (n, score_dbi))

    Bayesian_Gaussian_result = pd.concat([useful_feature, pd.Series(labels, index=useful_feature.index)], axis=1)

    Bayesian_Gaussian_result.columns = list(useful_feature.columns) + ['label']

    Bayesian_Gaussian_result.insert(0, 'eventid', data_id)

    Bayesian_Gaussian_result['overall'] = data_score

    center_overall_sum = {}

    for i in range(n):

        temp = (Bayesian_Gaussian_result[Bayesian_Gaussian_result.label == i])['overall'].sum()

        key_word = '第' + str(i) + '类'

        center_overall_sum[key_word] = temp

    center_overall_sum = sorted(center_overall_sum.items(), key = lambda x:x[1], reverse = True)

    Old_label = []

    New_label = []

    num = 0

    for i in center_overall_sum:

        num += 1

        Old_label.append(int(i[0][1]))

        New_label.append(num)

    label = list(deepcopy(Bayesian_Gaussian_result['label']))

    temp = []

    for i in range(len(label)):

        for j in range(n):

            if label[i] == Old_label[j]:

                temp.append(New_label[j])

    Bayesian_Gaussian_result.drop('label', axis=1, inplace=True)

    Bayesian_Gaussian_result['label'] = temp

    return Bayesian_Gaussian_result
start = time.clock()

Bayesian_Gaussian_result = Bayesian_Gaussian_Mixture_cluser(all_data_norm, id_, list(overall['overall']), all_data_norm.columns, n, 'spherical')

end = time.clock()

print('Bayesian Gaussian running time', end-start)
Bayesian_Gaussian_result.head()
Bayesian_Gaussian_result['label'].value_counts()
def Gaussian_Mixture_cluser(useful_feature, data_id, data_score, important_features, n, cv_type):

    model = mixture.GaussianMixture(n_components=n, covariance_type=cv_type, random_state=4, max_iter=10000)

    model.fit(useful_feature)

    labels = model.predict(useful_feature)

    score_sil = metrics.silhouette_score(useful_feature, labels, metric='euclidean')

    print("当聚为%d簇时，高斯混合模型轮廓系数Silhouette Coefficient为：%f" % (n, score_sil))             # 计算轮廓系数

    score_cal = metrics.calinski_harabaz_score(useful_feature, labels) 

    print("当聚为%d簇时，高斯混合模型轮廓系数Calinski-Harabaz Index为：%f" % (n, score_cal))

    score_dbi = metrics.davies_bouldin_score(useful_feature, labels)                                     # DBI值越小越好（说明分散程度低）

    print("当聚为%d簇时，高斯混合模型 Davies-Bouldin分数值为：%f" % (n, score_dbi))

    Gaussian_result = pd.concat([useful_feature, pd.Series(labels, index=useful_feature.index)], axis=1)

    Gaussian_result.columns = list(useful_feature.columns) + ['label']

    Gaussian_result.insert(0, 'eventid', data_id)

    Gaussian_result['overall'] = data_score

    center_overall_sum = {}

    for i in range(n):

        temp = (Gaussian_result[Gaussian_result.label == i])['overall'].sum()

        key_word = '第' + str(i) + '类'

        center_overall_sum[key_word] = temp

    center_overall_sum = sorted(center_overall_sum.items(), key = lambda x:x[1], reverse = True)

    Old_label = []

    New_label = []

    num = 0

    for i in center_overall_sum:

        num += 1

        Old_label.append(int(i[0][1]))

        New_label.append(num)

    label = list(deepcopy(Gaussian_result['label']))

    temp = []

    for i in range(len(label)):

        for j in range(n):

            if label[i] == Old_label[j]:

                temp.append(New_label[j])

    Gaussian_result.drop('label', axis=1, inplace=True)

    Gaussian_result['label'] = temp

    return Gaussian_result
start = time.clock()

Gaussian_result = Gaussian_Mixture_cluser(all_data_norm, id_, list(overall['overall']), all_data_norm.columns, n, 'spherical')

end = time.clock()

print('Gaussian running time', end-start)