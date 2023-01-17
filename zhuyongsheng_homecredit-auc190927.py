# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#改变:1.DAYS_EMPLOYED，DAYS_BIRTH都取绝对值，并且变为年

#2.DAYS_EMPLOYED等于365243的，用np.nan取代，并创建DAYS_EMPLOYED_ANOM。因为分析发现这些异常数据的样本违约率明显较低。

#3.增加YEARS_EMPLOYED_DIF，表示年龄-工龄

#4.建立婚姻状态*小孩数量的新特征，发现与违约率有明显线性关系

import numpy as np 

import pandas as pd

import gc

import time

from contextlib import contextmanager

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





@contextmanager

#计算时间

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



# 非数量特征One-hot encoding

def one_hot_encoder(df, nan_as_category = True):

    original_columns = list(df.columns)

    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)

    new_columns = [c for c in df.columns if c not in original_columns]

    return df, new_columns



# 处理application_train.csv and application_test.csv

def application_train_test(num_rows = None, nan_as_category = False):

    # Read data and merge

    df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv', nrows= num_rows)

    test_df = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv', nrows= num_rows)

    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))

    df = df.append(test_df).reset_index()

    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)

    df = df[df['CODE_GENDER'] != 'XNA']

    df['New_Family_Status']=df['NAME_FAMILY_STATUS'].apply(lambda x:0 if x in ['Married','Widow'] else 1)

    df['CH_FA']=df['CNT_CHILDREN']*df['New_Family_Status']

    # 处理两种取值的非数量特征 (0 or 1; two categories)

    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:

        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    # 非数量特征 with One-Hot encode

    df, cat_cols = one_hot_encoder(df, nan_as_category)

    

    df['DAYS_EMPLOYED_ANOM']=(df['DAYS_EMPLOYED']==365243)

    df['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)

    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs() / 365

    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs() / 365

    df.rename(columns={'DAYS_BIRTH': 'YEARS_BIRTH', 'DAYS_EMPLOYED': 'YEARS_EMPLOYED'}, inplace=True)

    df['YEARS_EMPLOYED_PREC']=df['YEARS_EMPLOYED']/df['YEARS_BIRTH']

    df['YEARS_EMPLOYED_DIF']=df['YEARS_BIRTH']-df['YEARS_EMPLOYED']

    df['INCOME_CREDIT_PERC']=df['AMT_INCOME_TOTAL']/df['AMT_CREDIT']

    df['INCOME_PER_PERSON']=df['AMT_INCOME_TOTAL']/df['CNT_FAM_MEMBERS']

    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    df['AVG_CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']

   

    

    del test_df

    gc.collect()

    return df



# 处理 bureau.csv and bureau_balance.csv

def bureau_and_balance(num_rows = None, nan_as_category = True):

    bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv', nrows = num_rows)

    bb = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv', nrows = num_rows)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)

    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    

    # bureau_balance对表内相同SK_ID_BUREAU的所有行合并

    #将Bureau balance并入bureau.csv

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

    for col in bb_cat:

        bb_aggregations[col] = ['mean']

    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)

    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)

    del bb, bb_agg

    gc.collect()

    

    # 对bureau and bureau_balance合并后的报表，数量特征的合并方式

    num_aggregations = {

        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],

        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],

        'DAYS_CREDIT_UPDATE': ['mean'],

        'CREDIT_DAY_OVERDUE': ['max', 'mean'],

        'AMT_CREDIT_MAX_OVERDUE': ['mean'],

        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],

        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],

        'AMT_CREDIT_SUM_OVERDUE': ['mean'],

        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],

        'AMT_ANNUITY': ['max', 'mean'],

        'CNT_CREDIT_PROLONG': ['sum'],

        'MONTHS_BALANCE_MIN': ['min'],

        'MONTHS_BALANCE_MAX': ['max'],

        'MONTHS_BALANCE_SIZE': ['mean', 'sum']

    }

    # Bureau and bureau_balance非数量特征处理

    cat_aggregations = {}

    for cat in bureau_cat: cat_aggregations[cat] = ['mean']

    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    #bureau根据CREDIT_ACTIVE_Active特征分为有效卡和无效卡，两种卡分别处理。

    #bureau中Active credits的卡，相同SK_ID_CURR所有行合并

    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]

    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)

    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])

    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    del active, active_agg

    gc.collect()

    # bureau中 Closed credits,相同SK_ID_CURR所有行合并

    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]

    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)

    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])

    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    del closed, closed_agg, bureau

    gc.collect()

    return bureau_agg



# 处理previous_applications.csv

def previous_applications(num_rows = None, nan_as_category = True):

    prev = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv', nrows = num_rows)

    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)

    # 关于日期的明显错误值替换为空值

    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)

    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)

    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)

    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)

    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    #创造新特征: value ask / value received percentage

    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications的数量特征合并方法

    num_aggregations = {

        'AMT_ANNUITY': ['min', 'max', 'mean'],

        'AMT_APPLICATION': ['min', 'max', 'mean'],

        'AMT_CREDIT': ['min', 'max', 'mean'],

        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],

        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],

        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],

        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],

        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],

        'DAYS_DECISION': ['min', 'max', 'mean'],

        'CNT_PAYMENT': ['mean', 'sum'],

    }

    # Previous applications非数量特征

    cat_aggregations = {}

    for cat in cat_cols:

        cat_aggregations[cat] = ['mean']

    

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Application按照NAME_CONTRACT_STATUS_Approved分为通过和拒绝两类，分别处理

    # Previous Applications中通过申请的数据，对SK_ID_CURR相同的行合并

    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]

    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)

    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])

    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications中拒绝申请的数据，对SK_ID_CURR相同的行合并

    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]

    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)

    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])

    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    del refused, refused_agg, approved, approved_agg, prev

    gc.collect()

    return prev_agg



# 处理 POS_CASH_balance.csv

def pos_cash(num_rows = None, nan_as_category = True):

    pos = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv', nrows = num_rows)

    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)

    # POS_CASH_balance特征合并方法

    aggregations = {

        'MONTHS_BALANCE': ['max', 'mean', 'size'],

        'SK_DPD': ['max', 'mean'],

        'SK_DPD_DEF': ['max', 'mean']

    }

    for cat in cat_cols:

        aggregations[cat] = ['mean']

    

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)

    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # 计算客户pos cash账户总数

    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    del pos

    gc.collect()

    return pos_agg

    

# 处理installments_payments.csv

def installments_payments(num_rows = None, nan_as_category = True):

    ins = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', nrows = num_rows)

    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)

    # Percentage and difference paid in each installment (amount paid and installment value)

    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']

    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # 创造新特征：逾期支付和按时支付

    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']

    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']

    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)

    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    #installments_payments特征的合并方式

    aggregations = {

        'NUM_INSTALMENT_VERSION': ['nunique'],

        'DPD': ['max', 'mean', 'sum'],

        'DBD': ['max', 'mean', 'sum'],

        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],

        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],

        'AMT_INSTALMENT': ['max', 'mean', 'sum'],

        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],

        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']

    }

    for cat in cat_cols:

        aggregations[cat] = ['mean']

    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)

    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # 计算installments account总数

    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    del ins

    gc.collect()

    return ins_agg



# 处理credit_card_balance.csv

def credit_card_balance(num_rows = None, nan_as_category = True):

    cc = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv', nrows = num_rows)

    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    # 合并后丢弃SK_ID_PREV特征

    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)

    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])

    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # 计算credit card行数

    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc

    gc.collect()

    return cc_agg



# LightGBM GBDT with KFold or Stratified KFold

def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data

    train_df = df[df['TARGET'].notnull()]

    test_df = df[df['TARGET'].isnull()]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    del df

    gc.collect()

    # 交叉验证模型

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    # 创建储存模型运行结果的数组和表

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]



        # 贝叶斯优化得到的LightGBM超参数

        clf = LGBMClassifier(

            nthread=4,

            n_estimators=10000,

            learning_rate=0.02,

            num_leaves=34,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.041545473,

            reg_lambda=0.0735294,

            min_split_gain=0.0222415,

            min_child_weight=39.3259775,

            random_state=100,

            silent=-1,

            verbose=-1, )

        #模型拟合

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)



        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        #根据模型运行结果，储存特征重要性数据

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()



    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    # 保存测试集结果并画出特征重要性图标

    if not debug:

        test_df['TARGET'] = sub_preds

        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

    display_importances(feature_importance_df)

    return feature_importance_df



# 特征重要性作图

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))

    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))

    plt.title('LightGBM Features (avg over folds)')

    plt.tight_layout()

    plt.savefig('lgbm_importances01.png')







def main(debug = False):

    num_rows = 10000 if debug else None

    df = application_train_test(num_rows)

    with timer("Process bureau and bureau_balance"):

        bureau = bureau_and_balance(num_rows)

        print("Bureau df shape:", bureau.shape)

        df = df.join(bureau, how='left', on='SK_ID_CURR')

        del bureau

        gc.collect()

    with timer("Process previous_applications"):

        prev = previous_applications(num_rows)

        print("Previous applications df shape:", prev.shape)

        df = df.join(prev, how='left', on='SK_ID_CURR')

        del prev

        gc.collect()

    with timer("Process POS-CASH balance"):

        pos = pos_cash(num_rows)

        print("Pos-cash balance df shape:", pos.shape)

        df = df.join(pos, how='left', on='SK_ID_CURR')

        del pos

        gc.collect()

    with timer("Process installments payments"):

        ins = installments_payments(num_rows)

        print("Installments payments df shape:", ins.shape)

        df = df.join(ins, how='left', on='SK_ID_CURR')

        del ins

        gc.collect()

    with timer("Process credit card balance"):

        cc = credit_card_balance(num_rows)

        print("Credit card balance df shape:", cc.shape)

        df = df.join(cc, how='left', on='SK_ID_CURR')

        del cc

        gc.collect()

    with timer("Run LightGBM with kfold"):

        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)





if __name__ == "__main__":

    submission_file_name = "submission_kernel02.csv"

    with timer("Full model run"):

        main()






