import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')
c_feat = list(df.select_dtypes(['float64', 'int64']).columns)

c_data = df[c_feat].fillna(0)
y = c_data.SalePrice

X = c_data[c_feat[:-1]]

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, random_state=1, test_size=0.3)
re_or = GradientBoostingRegressor(random_state=1)

re_or.fit(X_tr, y_tr)

y_pred = re_or.predict(X_ts)

er_mae = mean_absolute_error(y_pred, y_ts)

print(er_mae)
model_on_full_data = GradientBoostingRegressor(random_state=1)

model_on_full_data.fit(X, y)



Xt = test_data[c_feat[:-1]]

Xt = Xt.fillna(0)



test_preds = model_on_full_data.predict(Xt)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)