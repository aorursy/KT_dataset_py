import pandas
import numpy
import xgboost
import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pandas.read_csv('../input/train.csv')
features = df.columns[2:]
X = df[features].values
y = df['target'].values
dtrain = xgboost.DMatrix(X, y, missing=0.0, nthread=-1)
def rmsle(predictions, dmat):
    labels = dmat.get_label()
    diffs = numpy.log(predictions + 1) - numpy.log(labels + 1)
    squared_diffs = numpy.square(diffs)
    avg = numpy.mean(squared_diffs)
    return ('RMSLE', numpy.sqrt(avg))
params = {
    'objective': 'reg:linear',
    'max_depth': 6,
    'learning_rate': .01
}
results = xgboost.cv(params, dtrain, 100, early_stopping_rounds=10, feval=rmsle, nfold=10, verbose_eval=True, show_stdv=False)
results[['train-RMSLE-mean', 'test-RMSLE-mean']].plot()
results[['train-rmse-mean', 'test-rmse-mean']].plot()
