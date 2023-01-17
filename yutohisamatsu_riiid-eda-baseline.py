# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

    'content_type_id': 'int8',

    'task_container_id': 'int16',

    'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=False,

                       nrows=10**7,

                       dtype=data_types_dict, 

                      )
train_df.head(10)
train_df.info()
train_df.describe()
train_df['timestamp'].hist(bins=100);
grouped_by_user_df = train_df.groupby('user_id')
grouped_by_user_df.agg({'timestamp':'max'}).hist(bins=100);
# 講義の割合  # 平均 -1 (True)の割合

(train_df['answered_correctly'] == -1).mean()
train_questions_only_df = train_df[train_df['answered_correctly'] != -1]

train_questions_only_df['answered_correctly'].mean()
grouped_by_user_df = train_questions_only_df.groupby('user_id')
# 回答率('mean')と回答数（'count'）で分ける

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']})

user_answers_df[('answered_correctly', 'mean')].hist(bins=100); # bins = 棒の数
user_answers_df
user_answers_df[('answered_correctly', 'count')].hist(bins=100);
(user_answers_df[('answered_correctly','count')]< 50).mean()
# 初心者の正答率

user_answers_df[user_answers_df[('answered_correctly', 'count')] < 50][('answered_correctly', 'mean')].mean()
user_answers_df[user_answers_df[('answered_correctly', 'count')] < 50][('answered_correctly', 'mean')].hist(bins=100);
# アクティブユーザーの正答率

user_answers_df[user_answers_df[('answered_correctly', 'count')] >= 50][('answered_correctly', 'mean')].mean()
user_answers_df[user_answers_df[('answered_correctly', 'count')] >= 50][('answered_correctly', 'mean')].hist(bins=100);
# ヘビーユーザーの割合 500以上questionを回答しているユーザーの割合

(user_answers_df[('answered_correctly','count')] >= 500).mean()
# ヘビーユーザーの回答率の分布

user_answers_df[user_answers_df[('answered_correctly', 'count')] >= 500][('answered_correctly', 'mean')].hist(bins=100);
# ヘビーユーザーの正答率

user_answers_df[user_answers_df[('answered_correctly', 'count')] >= 500][('answered_correctly', 'mean')].mean()
plt.scatter(x = user_answers_df[('answered_correctly', 'count')], y = user_answers_df[('answered_correctly', 'mean')]);
grouped_by_content_df = train_questions_only_df.groupby('content_id')
content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count']})
content_answers_df
content_answers_df[('answered_correctly', 'count')].hist(bins=100);
content_answers_df[('answered_correctly', 'mean')].hist(bins=100);
content_answers_df[content_answers_df[('answered_correctly','count')]>50][('answered_correctly','mean')].hist(bins = 100);
questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')
questions_df
print(f"There are {len(questions_df['part'].unique())} different parts")
questions_df['tags'].values[-1] # なんで最後の行のtagを取得してるのか？
unique_tags = set().union(*[y.split() for y in questions_df['tags'].astype(str).values])



print(f"There are {len(unique_tags)} different tags")
# [question_id] content_type_idが質問(0)のとき、train/test content_id列の外部キー / [bundle_id] 質問と一緒に提供されるコード

(questions_df['question_id'] != questions_df['bundle_id']).mean()
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       low_memory=False,

                       nrows=10**7,

                       dtype=data_types_dict, 

                      )
# 900万行

features_part_df = train_df.iloc[:int( 9 / 10 * len(train_df) )]

# 100万行

train_part_df = train_df.iloc[int( 9 / 10 * len(train_df) ):]
train_questions_only_df = features_part_df[features_part_df['answered_correctly'] != -1]



grouped_by_user_df = train_questions_only_df.groupby('user_id')



user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']
user_answers_df
grouped_by_content_df = train_questions_only_df.groupby('content_id')



content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked']
content_answers_df
questions_df = questions_df.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')



# [question_id] content_id列の外部キー ・・・　content_type_idが質問(0)のとき、train/test content_id列の外部キー
questions_df
bundle_dict = questions_df['bundle_id'].value_counts().to_dict()

# value_count ・・・　ユニークな要素の値とその出現回数をpandas.Seriesで返す
# [bundle_id] 質問と一緒に提供されるコード

bundle_dict
# right_answers 正解数

questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']



questions_df['bundle_size'] = questions_df['bundle_id'].apply(lambda x: bundle_dict[x])
questions_df
grouped_by_bundle_df = questions_df.groupby('bundle_id')



bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

bundle_answers_df.columns = ['bundle_right_answers', 'bundle_questions_asked']



bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_right_answers'] / bundle_answers_df['bundle_questions_asked']



bundle_answers_df
grouped_by_part_df = questions_df.groupby('part')



part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()



part_answers_df.columns = ['part_right_answers', 'part_questions_asked']

part_answers_df['part_accuracy'] = part_answers_df['part_right_answers'] / part_answers_df['part_questions_asked']



part_answers_df
del train_df

del features_part_df

del grouped_by_user_df

del grouped_by_content_df
# python のメモリ管理

import gc

# 何も考えずにとりあえずGCを動かすには以下の通り。回収可能なオブジェクトを削除。

gc.collect()
features = [

    'timestamp','mean_user_accuracy', 'questions_answered','mean_accuracy',

    'question_asked','prior_question_elapsed_time', 'prior_question_had_explanation',

    'bundle_size', 'bundle_accuracy','part_accuracy', 'right_answers'

]



target = 'answered_correctly'
# 講義(-1)以外を抽出 train

train_part_df = train_part_df[train_part_df[target] != -1]
train_part_df.head()
# 追加した特徴量のdfをマージ



# user_answers_df

train_part_df = train_part_df.merge(user_answers_df, how='left', on='user_id')



# questions_df

train_part_df = train_part_df.merge(questions_df, how='left', left_on='content_id', right_on='question_id')



# bundle_answers_df

train_part_df = train_part_df.merge(bundle_answers_df, how='left', on='bundle_id')



# part_answers_df

train_part_df = train_part_df.merge(part_answers_df, how='left', on='part')
train_part_df.head()
# ユーザーが質問に回答した後、説明と正しい回答を確認したかどうか 欠損値をFalseと置く、 astypeでデータ型の変換(キャスト)

train_part_df['prior_question_had_explanation'] = train_part_df['prior_question_had_explanation'].fillna(value=False).astype(bool)



train_part_df.fillna(value = -1, inplace = True)
train_part_df
train_part_df.columns
train_part_df = train_part_df[features + [target]]
train_part_df
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(

    num_leaves=31, 

    max_depth= 2, 

    n_estimators = 25, 

    min_child_samples = 1000, 

    subsample=0.7, 

    subsample_freq=5,

    n_jobs= -1,

    is_higher_better = True,

    first_metric_only = True

)
lgbm.fit(train_part_df[features], train_part_df[target])
roc_auc_score(train_part_df[target].values, lgbm.predict_proba(train_part_df[features])[:,1])
import riiideducation



env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

    test_df = test_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')

    test_df = test_df.merge(part_answers_df, how = 'left', on = 'part')

    

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = -1, inplace = True)



    test_df['answered_correctly'] = lgbm.predict_proba(test_df[features])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])