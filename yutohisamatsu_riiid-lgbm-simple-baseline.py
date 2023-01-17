import lightgbm as lgb

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder



import riiideducation
env = riiideducation.make_env()
train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', low_memory=False, nrows=10**7,

                      dtype={

                          'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',

                              'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 

                             'prior_question_had_explanation': 'boolean',

                      })



# train_df = train_df.query('answered_correctly != -1').reset_index(drop=True)

# train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(float) 後でやる
train_df
# 900万行・・・特徴量作成

features_part_df = train_df.iloc[:int( 9 / 10 * len(train_df) )]

# 100万行・・・最新の100万件 

train_part_df = train_df.iloc[int( 9 / 10 * len(train_df) ):]
train_part_df
# 追加するデータ1 user_answers_df

train_questions_only_df = features_part_df[features_part_df['answered_correctly'] != -1]



grouped_by_user_df = train_questions_only_df.groupby('user_id')



user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy', 'median_user_accuracy', 'skew_user_accuracy']
user_answers_df
# 追加するデータ2 questions_df

questions_df = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')



grouped_by_content_df = train_questions_only_df.groupby('content_id')



content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew'] }).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked', 'std_accuracy', 'median_accuracy', 'skew_accuracy']



questions_df = questions_df.merge(content_answers_df, left_on = 'question_id', right_on = 'content_id', how = 'left')



bundle_dict = questions_df['bundle_id'].value_counts().to_dict()



# right_answers 正解数

questions_df['right_answers'] = questions_df['mean_accuracy'] * questions_df['question_asked']



questions_df['bundle_size'] = questions_df['bundle_id'].apply(lambda x: bundle_dict[x])
questions_df
# 追加するデータ3 bundle_answers_df

grouped_by_bundle_df = questions_df.groupby('bundle_id')



bundle_answers_df = grouped_by_bundle_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()

bundle_answers_df.columns = ['bundle_right_answers', 'bundle_questions_asked']



bundle_answers_df['bundle_accuracy'] = bundle_answers_df['bundle_right_answers'] / bundle_answers_df['bundle_questions_asked']
bundle_answers_df
# 追加するデータ4 part_answers_df

grouped_by_part_df = questions_df.groupby('part')



part_answers_df = grouped_by_part_df.agg({'right_answers': 'sum', 'question_asked': 'sum'}).copy()



part_answers_df.columns = ['part_right_answers', 'part_questions_asked']

part_answers_df['part_accuracy'] = part_answers_df['part_right_answers'] / part_answers_df['part_questions_asked']
part_answers_df
# 旧

# features = [

#     'timestamp','mean_user_accuracy', 'questions_answered','mean_accuracy',

#     'question_asked','prior_question_elapsed_time', 'prior_question_had_explanation',

#     'bundle_size', 'bundle_accuracy','part_accuracy', 'right_answers'

# ]



# 新

features = [

    'timestamp','prior_question_elapsed_time', 'prior_question_had_explanation',

    'mean_user_accuracy', 'questions_answered', 'std_user_accuracy',

    'median_user_accuracy', 'skew_user_accuracy','mean_accuracy',

    'question_asked', 'std_accuracy', 'median_accuracy', 'skew_accuracy',

    'bundle_size','bundle_accuracy', 'part_accuracy'

]



target = 'answered_correctly'
# 講義(-1)以外を抽出 train

train_part_df = train_part_df[train_part_df[target] != -1]
# user_answers_df

train_part_df = train_part_df.merge(user_answers_df, how='left', on='user_id')



# questions_df

train_part_df = train_part_df.merge(questions_df, how='left', left_on='content_id', right_on='question_id')



# bundle_answers_df

train_part_df = train_part_df.merge(bundle_answers_df, how='left', on='bundle_id')



# part_answers_df

train_part_df = train_part_df.merge(part_answers_df, how='left', on='part')
# ユーザーが質問に回答した後、説明と正しい回答を確認したかどうか 欠損値をFalseと置く、 astypeでデータ型の変換(キャスト)

train_part_df['prior_question_had_explanation'] = train_part_df['prior_question_had_explanation'].fillna(value=False).astype(bool)



train_part_df.fillna(value = -1, inplace = True)
# ラベルエンコーディング

le = LabelEncoder()

train_part_df["prior_question_had_explanation"] = le.fit_transform(train_part_df["prior_question_had_explanation"])
train_part_df
train_part_df.columns
X_train = train_part_df[features]

y_train = train_part_df[target]
X_train
models = []

oof_train = np.zeros(len(X_train),) ### array([0., 0., 0., ..., 0., 0., 0.])

categorical_features = ['prior_question_had_explanation']



params = {

    'objective': 'binary',

    'max_bin': 300,

    'learning_rate': 0.05,

    'num_leaves': 40

}



n_tr = round(981094 * 0.9)



X_tr = X_train[:n_tr]

X_val = X_train[n_tr:]



y_tr = y_train[:n_tr]

y_val = y_train[n_tr:]



lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)



model = lgb.train(

    params,

    lgb_train,

    valid_sets=[lgb_train, lgb_eval],

    verbose_eval=10,

    num_boost_round=1000,

    early_stopping_rounds=10

)



oof_train = model.predict(X_val, num_iteration=model.best_iteration)



models.append(model)
# ROC曲線のAUCスコア・・・曲線下の面積を意味するらしい。

roc_auc_score(y_val, oof_train)
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(questions_df, how = 'left', left_on = 'content_id', right_on = 'question_id')

    test_df = test_df.merge(bundle_answers_df, how = 'left', on = 'bundle_id')

    test_df = test_df.merge(part_answers_df, how = 'left', on = 'part')

    

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df.fillna(value = -1, inplace = True)

    X_test = test_df[features]

    

    for model in models:

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        y_preds.append(y_pred)

        

    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])