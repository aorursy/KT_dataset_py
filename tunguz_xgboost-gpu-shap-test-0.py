!pip install --use-feature=2020-resolver https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/xgboost-1.3.0_SNAPSHOT%2B00b0ad1293b4fa74d6aca5da4e9ab7a9d16777f0-py3-none-manylinux2010_x86_64.whl
import xgboost

xgboost.__version__
from sklearn.datasets import load_boston, load_digits

X, y = load_boston(return_X_y=True)
dtrain = xgboost.DMatrix(X, label=y)
%%time

param = {'max_depth':2, 'eta':.1, 'objective':'reg:squarederror' , 'tree_method': 'gpu_hist', 'predictor': "gpu_predictor"}

clf = xgboost.train(param, dtrain, 2000)
%%time

clf.predict(dtrain, pred_contribs=True)
%%time

param = {'max_depth':2, 'eta':.1, 'objective':'reg:squarederror' , 'tree_method': 'hist', 'predictor': "cpu_predictor"}

clf = xgboost.train(param, dtrain, 2000)
%%time

clf.predict(dtrain, pred_contribs=True)
%%time

param = {'max_depth':2, 'eta':.1, 'objective':'reg:squarederror' , 'tree_method': 'exact', 'predictor': "cpu_predictor"}

clf = xgboost.train(param, dtrain, 2000)
%%time

clf.predict(dtrain, pred_contribs=True)