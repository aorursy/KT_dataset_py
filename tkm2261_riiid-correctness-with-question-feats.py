data_types_dict = {

    'row_id': 'int64',

    'timestamp': 'int64',

    'user_id': 'int32',

    'content_id': 'int16',

#     'content_type_id': 'int8',

#     'task_container_id': 'int16',

#     'user_answer': 'int8',

    'answered_correctly': 'int8',

    'prior_question_elapsed_time': 'float16',

    'prior_question_had_explanation': 'boolean'

}
import pandas as pd

train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', 

                       nrows=10**7,

                       usecols = data_types_dict.keys(),

                       dtype=data_types_dict, 

                       index_col = 0)
grouped_by_user_df = train_df.groupby('user_id')
grouped_by_user_df.agg({'timestamp': 'max'}).hist(bins = 100)
(train_df['answered_correctly']==-1).mean()
train_questions_only_df = train_df[train_df['answered_correctly']!=-1]

train_questions_only_df['answered_correctly'].mean()
grouped_by_user_df = train_questions_only_df.groupby('user_id')
user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'] })



user_answers_df[('answered_correctly','mean')].hist(bins = 100)
user_answers_df[('answered_correctly','count')].hist(bins = 100)
(user_answers_df[('answered_correctly','count')]< 50).mean()
user_answers_df[user_answers_df[('answered_correctly','count')]< 50][('answered_correctly','mean')].mean()
user_answers_df[user_answers_df[('answered_correctly','count')]< 50][('answered_correctly','mean')].hist(bins = 100)
user_answers_df[user_answers_df[('answered_correctly','count')] >= 50][('answered_correctly','mean')].hist(bins = 100)
user_answers_df[user_answers_df[('answered_correctly','count')] >= 50][('answered_correctly','mean')].mean()
user_answers_df[user_answers_df[('answered_correctly','count')] >= 500][('answered_correctly','mean')].hist(bins = 100)
import matplotlib.pyplot as plt

plt.scatter(x = user_answers_df[('answered_correctly','count')], y=user_answers_df[ ('answered_correctly','mean')])
grouped_by_content_df = train_questions_only_df.groupby('content_id')
content_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count'] })
content_answers_df[('answered_correctly','count')].hist(bins = 100)
content_answers_df[('answered_correctly','mean')].hist(bins = 100)
content_answers_df[content_answers_df[('answered_correctly','count')]>50][('answered_correctly','mean')].hist(bins = 100)
train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

                       usecols = data_types_dict.keys(),

                       dtype=data_types_dict, 

                       index_col = 0)
features_part_df = train_df.iloc[:int(9 /10 * len(train_df))]

train_part_df = train_df.iloc[int(9 /10 * len(train_df)):]
train_questions_only_df = features_part_df[features_part_df['answered_correctly']!=-1]

grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']

# user_features_dict = user_answers_df.to_dict('index')
features_part_df = train_df.iloc[:int(9 /10 * len(train_df))]

train_part_df = train_df.iloc[int(9 /10 * len(train_df)):]
train_questions_only_df = features_part_df[features_part_df['answered_correctly']!=-1]

grouped_by_user_df = train_questions_only_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']

# user_features_dict = user_answers_df.to_dict('index')
grouped_by_content_df = train_questions_only_df.groupby('content_id')

content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked']

# user_features_dict = conten_answers_df.to_dict('index')
del train_df

del features_part_df

del grouped_by_user_df

del grouped_by_content_df
import gc

gc.collect()
import numpy as np

questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv', 

                        names=['content_id', 'q_bundle_id', 'q_correct_answer', 'q_part', 'q_tags'], header=0) 

def ppp(x):

    try:

        if len(x) == 0:

            return np.nan

        else:

            return int(x.split()[0])

    except:

        return np.nan

questions['q_tags'] = questions['q_tags'].fillna('').map(ppp)

questions.drop('q_bundle_id', axis=1, inplace=True)

questions.set_index('content_id')
features = ['timestamp','mean_user_accuracy', 'questions_answered','mean_accuracy', 'question_asked', 'prior_question_elapsed_time', 'prior_question_had_explanation',

           'q_correct_answer', 'q_part', 'q_tags']

target = 'answered_correctly'
train_part_df = train_part_df[train_part_df[target] != -1]
train_part_df = train_part_df.merge(user_answers_df, how = 'left', on = 'user_id')

train_part_df = train_part_df.merge(content_answers_df, how = 'left', on = 'content_id')

train_part_df = train_part_df.merge(questions, how = 'left', on = 'content_id')
train_part_df['prior_question_had_explanation'] = train_part_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

train_part_df.fillna(value = -1, inplace = True)
train_part_df.columns
train_part_df = train_part_df[features + [target]]
train_part_df
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(

    boosting_type='gbdt', 

    num_leaves=31, 

    max_depth=- 1, 

    n_estimators=60, 

    min_child_samples=1000, 

    subsample=0.6, 

    subsample_freq=1, 

    n_jobs= 2

)
lgbm.fit(train_part_df[features], train_part_df[target])
roc_auc_score(train_part_df[target].values, lgbm.predict_proba(train_part_df[features])[:,1])
import riiideducation



env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(content_answers_df, how = 'left', on = 'content_id')

    test_df = test_df.merge(questions, how = 'left', on = 'content_id')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = -1, inplace = True)



    test_df['answered_correctly'] = lgbm.predict_proba(test_df[features])[:,1]

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])