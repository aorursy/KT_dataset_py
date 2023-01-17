##################################################################

#                                                                #

#             Wonsoek Bae. Chung-Ang Univ. 2019.05.24            #

#                                                                #

##################################################################



#======================================================================================================================



print("0. 필요한 모듈 import")



import numpy as np

from sklearn.metrics import roc_curve, auc

"""fpr, tpr, thresholds = roc_curve(y, model.decision_function(X)) 

입력값은 타켓 y 벡터와 변하는 판결값(함수의 반환값)

출력값 false positive rate, true positive rate, 판별값들 묶음. 이것들로  plot 그리면 roc curve 그릴 수 있음

auc : roc_curve 의 넓이"""



import pandas as pd

"1. pandas.dataframe : 데이터 집합형식-인덱스가 있는 다차원 배열(행렬)"

import lightgbm as lgb

"""1. lightgbm.Dataset : lgb 함수에 쓰이는 데이터 집합형식

 2. lightgbm.cv : 입력값으로 Booster params 와, Data to be trained 를 받아서, 딕셔너리 타입의 Evaluation history 를 반환.

 3. lightgbm.train : 입력값으로 Parameters for training 와 Data to be trained 를 받아서 학습된 부스터 모델을 반환"""



from bayes_opt import BayesianOptimization

from sklearn.metrics import roc_auc_score

import warnings

import time

warnings.filterwarnings("ignore")



# 필요한 함수들

"""여기 함수는 CERN과 함께 이 프로젝트의 스폰서인 yandexdataschool 에서 제공

https://github.com/yandexdataschool/flavours-of-physics-start/blob/master/evaluation.py"""



"코릴레이션 검증을 위한 함수"

def __rolling_window(data, window_size):

    """

    Rolling window: take window with definite size through the array



    :param data: array-like

    :param window_size: size

    :return: the sequence of windows



    Example: data = array(1, 2, 3, 4, 5, 6), window_size = 4

        Then this function return array(array(1, 2, 3, 4), array(2, 3, 4, 5), array(3, 4, 5, 6))

    """

    shape = data.shape[:-1] + (data.shape[-1] - window_size + 1, window_size)

    strides = data.strides + (data.strides[-1],)

    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)



def __cvm(subindices, total_events):

    """

    Compute Cramer-von Mises metric.

    Compared two distributions, where first is subset of second one.

    Assuming that second is ordered by ascending



    :param subindices: indices of events which will be associated with the first distribution

    :param total_events: count of events in the second distribution

    :return: cvm metric

    """

    target_distribution = np.arange(1, total_events + 1, dtype='float') / total_events

    subarray_distribution = np.cumsum(np.bincount(subindices, minlength=total_events), dtype='float')

    subarray_distribution /= 1.0 * subarray_distribution[-1]

    return np.mean((target_distribution - subarray_distribution) ** 2)



def compute_cvm(predictions, masses, n_neighbours=100, step=50):

    """

    Computing Cramer-von Mises (cvm) metric on background events: take average of cvms calculated for each mass bin.

    In each mass bin global prediction's cdf is compared to prediction's cdf in mass bin.



    :param predictions: array-like, predictions

    :param masses: array-like, in case of Kaggle tau23mu this is reconstructed mass

    :param n_neighbours: count of neighbours for event to define mass bin

    :param step: step through sorted mass-array to define next center of bin

    :return: average cvm value

    """

    predictions = np.array(predictions)

    masses = np.array(masses)

    assert len(predictions) == len(masses)



    # First, reorder by masses

    predictions = predictions[np.argsort(masses)]



    # Second, replace probabilities with order of probability among other events

    predictions = np.argsort(np.argsort(predictions, kind='mergesort'), kind='mergesort')



    # Now, each window forms a group, and we can compute contribution of each group to CvM

    cvms = []

    for window in __rolling_window(predictions, window_size=n_neighbours)[::step]:

        cvms.append(__cvm(subindices=window, total_events=len(predictions)))

    return np.mean(cvms)





"어그리먼트 검증을 위한 함수"

def __roc_curve_splitted(data_zero, data_one, sample_weights_zero, sample_weights_one):

    """

    Compute roc curve



    :param data_zero: 0-labeled data

    :param data_one:  1-labeled data

    :param sample_weights_zero: weights for 0-labeled data

    :param sample_weights_one:  weights for 1-labeled data

    :return: roc curve

    """

    labels = [0] * len(data_zero) + [1] * len(data_one)

    weights = np.concatenate([sample_weights_zero, sample_weights_one])

    data_all = np.concatenate([data_zero, data_one])

    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)

    return fpr, tpr



def compute_ks(data_prediction, mc_prediction, weights_data, weights_mc):

    """

    Compute Kolmogorov-Smirnov (ks) distance between real data predictions cdf and Monte Carlo one.



    :param data_prediction: array-like, real data predictions

    :param mc_prediction: array-like, Monte Carlo data predictions

    :param weights_data: array-like, real data weights

    :param weights_mc: array-like, Monte Carlo weights

    :return: ks value

    """

    assert len(data_prediction) == len(weights_data), 'Data length and weight one must be the same'

    assert len(mc_prediction) == len(weights_mc), 'Data length and weight one must be the same'



    data_prediction, mc_prediction = np.array(data_prediction), np.array(mc_prediction)

    weights_data, weights_mc = np.array(weights_data), np.array(weights_mc)



    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'Data predictions are out of range [0, 1]'

    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'



    weights_data /= np.sum(weights_data)

    weights_mc /= np.sum(weights_mc)



    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)



    Dnm = np.max(np.abs(fpr - tpr))

    return Dnm



"평가를 위한 함수"

def roc_auc_truncated(labels, predictions, tpr_thresholds=(0.2, 0.4, 0.6, 0.8), roc_weights=(4, 3, 2, 1, 0)):

    """

    These weights were chosen to match the evaluation methodology used by CERN scientists.

    Note that the weighted AUC is calculated

    only for events (simulated signal events for tau->µµµ and real background events for tau->µµµ) with min_ANNmuon > 0.4

    """



    """

    Compute weighted area under ROC curve.



    :param labels: array-like, true labels

    :param predictions: array-like, predictions

    :param tpr_thresholds: array-like, true positive rate thresholds delimiting the ROC segments

    :param roc_weights: array-like, weights for true positive rate segments

    :return: weighted AUC

    """

    assert np.all(predictions >= 0.) and np.all(predictions <= 1.), 'Data predictions are out of range [0, 1]'

    assert len(tpr_thresholds) + 1 == len(roc_weights), 'Incompatible lengths of thresholds and weights'

    fpr, tpr, _ = roc_curve(labels, predictions)

    area = 0.

    tpr_thresholds = [0.] + list(tpr_thresholds) + [1.]

    for index in range(1, len(tpr_thresholds)):

        tpr_cut = np.minimum(tpr, tpr_thresholds[index])

        tpr_previous = np.minimum(tpr, tpr_thresholds[index - 1])

        area += roc_weights[index - 1] * (auc(fpr, tpr_cut, reorder=True) - auc(fpr, tpr_previous, reorder=True))

    tpr_thresholds = np.array(tpr_thresholds)

    # roc auc normalization to be 1 for an ideal classifier

    area /= np.sum((tpr_thresholds[1:] - tpr_thresholds[:-1]) * np.array(roc_weights))

    return area
print("1. 데이터 가져오기")



train_set = pd.read_csv('../input/training.csv', index_col='id') # [67553 rows x 50 columns]

test = pd.read_csv('../input/test.csv', index_col='id') # [855819 rows x 46 columns]

check_agreement = pd.read_csv('../input/check_agreement.csv', index_col='id') # [331147 rows x 48 columns]

check_correlation = pd.read_csv('../input/check_correlation.csv', index_col='id') # [5514 rows x 47 columns]

# id를 행 인덱스(세로축 레이블)로 .csv 데이터 파일을 끌어온다.
print("2. 데이터 전처리")



# 2.1 feature(레이블, 대게 물리량) engineering : 특성조합



def add_features(df):

    df['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']) ** 2

    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)

    df['NEW_FD_SUMP'] =df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])

    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3

    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']

    df['NEW_IP_dira'] = df['IP']*df['dira']

    # features from phunter

    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']

    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']

    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)

    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)

    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)

    # features from UGBC GS

    df['NEW_iso_abc'] = df['isolationa']*df['isolationb']*df['isolationc']

    df['NEW_iso_def'] = df['isolationd']*df['isolatione']*df['isolationf']

    df['NEW_pN_IP'] = df['p0_IP']+df['p1_IP']+df['p2_IP']

    df['NEW_pN_p']  = df['p0_p']+df['p1_p']+df['p2_p']

    df['NEW_IP_pNpN'] = df['IP_p0p2']*df['IP_p1p2']

    df['NEW_pN_IPSig'] = df['p0_IPSig']+df['p1_IPSig']+df['p2_IPSig']

    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']

    return df



# 2.2 새 레이블을 이용해서 데이터 가공



train_add = add_features(train_set) # [67553 rows x 68 columns]

test_add = add_features(test) # [855819 rows x 64 columns]

agr_add = add_features(check_agreement) # [331147 rows x 66 columns]

cor_add = add_features(check_correlation) # [5514 rows x 65 columns]



"기존 피쳐 지우기" # idea from UGBC GS

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits','CDF1', 'CDF2', 'CDF3',

              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',

              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',

              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_IP', 'p1_IP', 'p2_IP', 'IP_p0p2', 'IP_p1p2',

              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig',

              'DOCAone', 'DOCAtwo', 'DOCAthree']



"새 레이블에만 있고 옛 레이블에 없는 것만 모아서 col 이라는 레이블 만들기"

col = list(f for f in train_add.columns if f not in filter_out) # 28 columns

# col에 id없어도 id는 이미 행레이블로 들어갔음.
print("3 Bayesian Optimization으로 최적의 Hyper parameters 찾기")



def bayes_parameter_opt_lgb(init_round=20, opt_round=30, n_folds=5, random_seed=42, n_estimators=50000, learning_rate=0.0001):

    # prepare data

    train_data = lgb.Dataset(train_add[col], train_set['signal']) # 두 번째 변수가 True인 것만 뽑음

    # parameters

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['lambda_l1'] = max(lambda_l1, 0)

        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, metrics=['auc'])     

        

        max_value_niter = np.argmax(cv_result['auc-mean']) # CV의 리턴값이 딕셔너리로 나올텐데, 그 중에서 auc-mean 키의 밸류들 중에서 최대값에 대응되는 인덱스

        print('Best number of iterations: {}'.format(max_value_niter)) # Best number of iterations: Best number of iterations: 49999

        max_value_score = cv_result['auc-mean'][max_value_niter] # cv['auc-mean'] 중에서 그 인덱스의 값 = 즉 최대값

        print('Best CV score: {}'.format(max_value_score)) # Best CV score: 0.9407012723236857

        

        return max(cv_result['auc-mean'])

    

    # range 

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (20, 614),# num_leaves = 2^maxmax_depth * 60% 

                                            'feature_fraction': (0.8, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (5, 10),

                                            'lambda_l1': (0, 5),

                                            'lambda_l2': (0, 3),

                                            'min_split_gain': (0.001, 0.1),

                                            'min_child_weight': (5, 50)}, random_state=0)

        

    # optimize

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    return lgbBO.res



opt_params = bayes_parameter_opt_lgb(init_round=1, opt_round=1, n_folds=5, random_seed=42, n_estimators=10, learning_rate=0.0001)

print(opt_params)
print("4. lgb.cv랑 lgb.train 으로 가장 좋은 부스터 모델 만들기")



# 4.1 Light gradient boosting machine 에 들어갈 3에서 찾은 최적의 하이퍼 파라메터



# Best CV score: 0.934151783702392

#|  31       |  0.9342   |  1.0      |  0.8      |  4.648    |  0.0      |  9.99     |  5.0      |  0.04201  |  45.0

# 옵티마이제이션으로 최적의 하이퍼 파라미터를 찾은 뒤, 다른 조건으로 여러 번 옵티마이제이션을 작동 시켰기 때문에

# 이전에 찾았던 최적의 하이퍼 파라미터를 수동으로 입력해서 사용.

params = {'application':'binary','num_iterations': 10, 'learning_rate':0.0001, 'metric':'auc'} # params이라는 딕셔너리 만들기

params["num_leaves"] = int(round(45.0 )) # number of leaves in one tree

params['feature_fraction'] = 0.8 # 0~1 사이의 값. 0.8이면, 각각의 tree에 80%의 features 사용. 학습 속도 높이고 over-fitting 막는데 씀

params['bagging_fraction'] = 1.0 # 0~1 사이의 값. 데이터를 랜덤 추출하게 함. 학습 속도 높이고 over-fitting 막는데 씀

params['max_depth'] = int(round(9.99)) # Limit the max depth for tree model. 데이터 작을 때 over-fitting 막는데 씀

params['lambda_l1'] = 4.638 # regularization 해서 over-fitting 막는데 씀

params['lambda_l2'] = 0.0 # regularization 해서 over-fitting 막는데 씀

params['min_split_gain'] = 0.04201 # split 을 만들기 위한 최소 gain. 트리 안에서 useful split 수 조절

params['min_child_weight'] = 5.0 # leaf에서 필요한 instance weight (hessian)의 최소 합 .



"""application':'binary = 참/거짓 이진판별, num_iterations = number of boosting iterations/trees 

metrics (string or list of strings) – Evaluation metrics to be watched in CV.

metrics ex)

             실제 참   실제 거짓

  예측 참    10000개     10개

  예측 거짓     10개    10000개

"""





"가공된 훈련 데이터를 lgb 함수에 넣을 수 있게 lgb 데이터 형식으로 만들어 주기"

train = lgb.Dataset(train_add[col], train_set['signal']) # 두 번째 변수가 True인 것만 뽑음

"train = train <lightgbm.basic.Dataset object at 0x000002940095D438> 출력하면 그냥 28행 레이블만 나옴"



# 4.2 Best boosting number 구하기 : cross validation

cv = lgb.cv(params, train, nfold=5, seed=42, stratified=True, verbose_eval =100, metrics=['auc'])

"""입력값으로 Booster params 와, Data to be trained 를 받아서, 딕셔너리 타입의 Evaluation history 를 반환.

  하리퍼 파라메터를 가지고, 데이터를 nfold 개수로 나눈 다음, 파라메터를 바꿔가면서 교차 평가하면 평가 (auc) 점수가 오락가락 함 

  nfold에서 교차해본 평균값이 auc-mean

  

  입력값 :

  params, Dataset, 

  nfold : dataset 을 몇 개의 균등한 크기의 subsamples 로 나눌 건가.(train data가 60000개 안 되서 5개로 나눔)

  seed - fold 별로 데이터를 구성을 random 하게 나눠주는 것.

  stratified - sampling of folds should be stratified by the values of outcome labels.)

  verbose_eval - the eval metric on the valid set is printed at every verbose_eval boosting stage. 100번째마다 출력 

  ex) [100]	cv_agg's auc: 0.914896 + 0.0033455

      [200]	cv_agg's auc: 0.915306 + 0.00328177

  

  반환값 :

  딕셔너리 타입으로 Evaluation history 를 반환. 

  그 중 최대 auc값과 그 때의 인덱스를 찾을 것.

  {'auc-mean': [0.900584369149491, 0.9068785796788049, ... 0.9183340848474986, 0.9183348220084587]

  , 'auc-stdv': [0.003673678624173514, 0.00354709321385367, ... 0.003214837891260049, 0.003214937592912147]}"""



best_niter = np.argmax(cv['auc-mean']) # CV의 리턴값이 딕셔너리로 나올텐데, 그 중에서 auc-mean 키의 밸류들 중에서 최대값에 대응되는 인덱스

print('Best number of iterations: {}'.format(best_niter)) # Best number of iterations: Best number of iterations: 49999

best_score = cv['auc-mean'][best_niter] # cv['auc-mean'] 중에서 그 인덱스의 값 = 즉 최대값

print('Best CV score: {}'.format(best_score)) # Best CV score: 0.9407012723236857



# 4.3 분류기 학습시키기

clf = lgb.train(params, train, num_boost_round=best_niter)

# 여기가 트레인 돌리기.  이게 우리가 만든 분류기. clf = classifier

#  입력값 : 

#      params (dict), train_set (Dataset), num_boost_round (int) – Number of boosting iterations

#      "num_boost_round = Number of boosting iterations. 이게 몇 학습기를 몇 개 연결시켜줄거냐는 것?"

#      "light bgm 에서 학습기 연결 개수가 tree 개수?"

#  반환값 : 

#      booster – The trained Booster model. ex N번 부스터 해서 학습된 부스터 모델을 반환
print("5. 입자물리학 실험에서 요구하는 검증 조건 확인하기")



# 5.1 어그리먼트 테스트

"""분류기가 학습하는 훈련 데이터에는, 실제 데이터와 시뮬레이션(mc)으로 만들어 진 데이터가 섞여 있음.

 따라서, 시뮬레이션 데이터가 충분히 정확하기 않게 만들어지고, 분류기가 그것에 대해 학습할 수 있음.

 그래서, τ → 3μ 붕괴 데이터와 유사한 Ds → φπ 붕괴의 실제 데이터와 시뮬레이션 데이터를 분류시켜보고, 

 그것들이 τ → 3μ 붕괴가 아님을 충분히 구별해야, 학습된 분류기를 신뢰할 수 있다. 

 이를 Kolmogorov-Smirnov (KS) 테스트라 한다. * https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

 

 여기에서는 KS 값이 0.09보다 작아야 한다. 이것을 검증하기 위한 함수(코드)는 CERN에서 제공 : compute_ks, roc_curve_splitted """



agreement_probs = clf.predict(agr_add[col])

"가공된 어그리먼트 데이터를 학습된 분류기 모델에 넣어서 (참/거짓)에 대한 prediction 값을 출력 "

"[0.60230961 0.579581   0.60094461 ... 0.62754844 0.6404412  0.6404412 ]"



ks = compute_ks(agreement_probs[check_agreement['signal'].values == 0], agreement_probs[check_agreement['signal'].values == 1],

check_agreement[check_agreement['signal'] == 0]['weight'].values, check_agreement[check_agreement['signal'] == 1]['weight'].values)

""" 인자 1: data_prediction = signal 값이 0인 것들의 agreement_probs 정보, 

인자 2: mc_prediction = signal 값이 0인 것들의 agreement_probs 정보

인자 3 :weights_data, = signal 값이 0인 것들의 weight 정보가 id : 값 딕셔너리 형태로 출력되는데, 거기서 weight 값,

인자 4 : weights_mc = signal 값이 1인 것들의 weight 정보가 id : 값 딕셔너리 형태로 출력되는데, 거기서 weight 값"""

print ('KS metric : ', ks, ks < 0.09) # KS metric 0.05148947099501236 True



# 5.2 코릴레이션 테스트

"""주어진 데이터의 질량은 이미 τ → 3μ 붕괴 후보로서 가능한 값들이다. 따라서 만들어진 모델을 질량과 입자 예측치 간의 

상관관계가 없어야 한다. (사실 입자 실험에서 질량 값은 적당한 추정치로만 얻어지며, 과학자들이 모델을 만들 때 정확하게 

신뢰해서는 안 되는 값이다.) Test 데이터에는 잘량 값은 없지만, 다른 물리량들 관계 속에 숨겨진 질량 값을 를 사용하여 

질량 값과 입자 예측치 간의 상관관계가 있는 지 검증한다. 이를 Cramer-von Mises (cvm) 테스트라 한다. 

a) 전체 데이터 집합에 대한 제출에서 예측 된 값

b) 전체 질량 범위를 따라 롤링 윈도우 방식으로 특정 질량 영역 내의 예측 된 값.

비교 하고 둘의 평균 값을 반환. 이 CVM 값이 0.02 보다 작아야 한다. 

이것을 검증하기 위한 함수(코드)는 CERN에서 제공 : rolling_window, cvm, compute_cvm"""





correlation_probs = clf.predict(cor_add[col])

"가공된 코릴레이션 데이터를 학습된 분류기 모델에 넣어서 (참/거짓)에 대한 prediction 값을 출력"

cvm = compute_cvm(correlation_probs, check_correlation['mass'])

"인자 1: 확률, 인자 2: 질량"

print ('CVM metric : ', cvm, cvm < 0.002) # CvM metric 0.0010729350995806089 True
print("6. 결과 파일 만들기(각각의 id에 대한 참 거짓 prediction 값)")



#if ks < 0.09 and cvm < 0.002:

result = pd.DataFrame({'id': test.index})

result['prediction'] = clf.predict(test_add[col])

"가공된 test 데이터를 학습된 분류기 모델에 넣어서 (참/거짓)에 대한 prediction 값을 출력"

result.to_csv('result.csv', index=False, sep=',') # 캐글에 올릴 결과 파일.



#result

#id       index

#0       14711831

#1       16316387

 

#855818  12395271

    

#result['prediction'] 

#0         0.542113

#1         0.542876

    

#855818    0.597033

    

#만들어서 둘이 합치기
print("7. 테스트 데이터의 정답은 CERN에서 제공하지 않으며, 제출한 파일에 대해 평가만 해준다. \n 따라서, 위에서 얻은 결과에 대한 roc curve/precision-recall-threshold 그래프는 얻을 수 없다. \n 따라서, 훈련 데이터를 8:2로 나눠서 80%의 데이터에 대해 다시 훈련시키고, 20%의 데이터에 대해 그것들을 평가해본다.")



from sklearn.model_selection import train_test_split

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve





# 1. 데이터 나누기

train_set1,train_set2 = train_test_split(train_set, test_size=0.2, random_state=42)

train_add1 = add_features(train_set1)

train_add2 = add_features(train_set2)



# 2. CV 돌리기



train = lgb.Dataset(train_add1[col], train_set1['signal']) # 두 번째 변수가 True인 것만 뽑음

cv = lgb.cv(params, train, nfold=5, seed=42, stratified=True, verbose_eval =100, metrics=['auc'])



best_niter = np.argmax(cv['auc-mean']) # CV의 리턴값이 딕셔너리로 나올텐데, 그 중에서 auc-mean 키의 밸류들 중에서 최대값에 대응되는 인덱스

print('Best number of iterations: {}'.format(best_niter)) # Best number of iterations: Best number of iterations: 49999

best_score = cv['auc-mean'][best_niter] # cv['auc-mean'] 중에서 그 인덱스의 값 = 즉 최대값

print('Best CV score: {}'.format(best_score)) # Best CV score: 0.9407012723236857



# 3. 분류기 학습시키기

clf = lgb.train(params, train, num_boost_round=best_niter)



# 4. 분류하기

train_eval = train_add2

train_eval_probs = clf.predict(train_eval[col]) # test_add[col] train_add[col]



# 5. roc 값 구하기

AUC = roc_auc_score(train_eval['signal'], train_eval_probs)  # 0/1 값(실제 참/거짓) 과 (예측 참/거짓)를 함수에 넣음

print('AUC : ', AUC)



# 6. roc curve 구하기

precisions, recalls, thresholds = precision_recall_curve(train_eval['signal'], train_eval_probs)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0, 1])

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()



#7. precision-recall-threshold 그래프

fpr, tpr, thresholds = roc_curve(train_eval['signal'], train_eval_probs)



def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')



plot_roc_curve(fpr, tpr)

plt.show()