import pandas as pd

from sklearn.metrics import log_loss
df_train = pd.read_csv('../input/dmia-sport-2019-fall-competition-1/train.csv')
df_train.head()
df_train.shape
checks_num = df_train.check_id.nunique()

checks_num
df_train = df_train.drop_duplicates(subset=['check_id', 'target'])
df_train.shape[0] == checks_num
target_probs = df_train.target.value_counts(normalize=True).sort_index()

target_probs
log_loss(df_train.target, [df_train.target.value_counts(normalize=True)]*df_train.shape[0])
df_test = pd.read_csv('../input/dmia-sport-2019-fall-competition-1/test.csv')
df_test.head(10)
ids = df_test.check_id.unique()
columns_names = ['target_{}'.format(i) for i in range(25)]
df_result = pd.DataFrame([target_probs.tolist()]*len(ids), columns=columns_names)

df_result.insert(0, 'check_id', ids)

df_result.head(10)
df_result.to_csv('const_submission.csv', index=False)