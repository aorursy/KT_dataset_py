import pandas as pd



def load_data(file):

    return pd.read_csv(file, index_col='ID')
train_df = load_data('../input/train.csv')

print("Train dataset has {} samples.".format(len(train_df)))

test_df = load_data('../input/test.csv')

print("Test dataset has {} samples.".format(len(test_df)))

train_df.head()
print(train_df['X0'].unique())
def mercedes_code_to_int(code):

    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 

             'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 

             'u', 'v', 'w', 'x', 'y', 'z','aa','ab','ac','ad',

            'ae','af','ag','ah','ai','aj','ak','al','am','an', 

            'ao','ap','aq','ar','as','at','au','av','aw','ax', 

            'ay','az','ba','bb','bc','bd','be','bf','bg','bh',  

    ]

    return vocab.index(code)
def extract_feature_matrix(df):

    return df['X0'].apply(mercedes_code_to_int).values.reshape(-1, 1)
import numpy as np

train_X = extract_feature_matrix(train_df)

print(np.unique(train_X))



train_y = train_df['y'].values

print(train_X.shape)

print(train_y.shape)
import xgboost as xgb
dtrain = xgb.DMatrix(data=train_X, label=train_y)

param = {'objective':'reg:linear', 'max_depth': 2, 'eta': 0.1}
from sklearn.metrics import r2_score

def kaggle_eror_eval(preds, dtrain):

    return 'r^2', r2_score(y_pred=preds, y_true=dtrain.get_label())
cv_results = xgb.cv(param, dtrain, 1000, nfold=10, verbose_eval=False, feval=kaggle_eror_eval,

                    maximize=False, early_stopping_rounds=20, seed=42, as_pandas=True)

cv_results.tail()
bst = xgb.train(param, dtrain, num_boost_round=len(cv_results))

bst
test_X = extract_feature_matrix(test_df)

print(np.unique(test_X))

print(test_X.shape)
dtest = xgb.DMatrix(data=test_X)

predictions = bst.predict(dtest)
submission = pd.DataFrame(index=test_df.index,

                          data={'y': predictions})

submission