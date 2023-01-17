import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
df = pd.read_csv('../input/train.csv').drop(['id'], 1) # id é irrelvante agora
X, y = df.drop('quality', 1), df['quality']-3
estimator = XGBClassifier()
param_grid = {
    'nthread': [4],
    'n_estimators': [1000],
    'learning_rate': [0.01, 0.05, 0.2, 0.25, 0.3],
    'min_child_weight': [1, 4, 11],
    'max_depth': np.arange(6, 9),
    'objective':['multi:softmax'],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'scale_pos_weight': [1],
    'seed': [42]
}
scoring = 'f1_micro'
cv = StratifiedKFold(3, random_state=42)
# n_iter = 135
n_iter = 2 # Valor baixo para o "commit & run" do Kaggle ir mais rápido.
%%time
search = RandomizedSearchCV(estimator, param_grid, scoring=scoring, cv=cv, n_iter=n_iter, verbose=0)
search.fit(X, y)
results = pd.DataFrame(search.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.head()
def to_csv(df):
    df.to_csv('tmp.csv', index=False)
    print(open('tmp.csv').read())
    
def make_submission(model):
    test = pd.read_csv('../input/test.csv')
    prediction = model.predict(test.drop(['id'], 1))
    submission = pd.DataFrame({'id': test['id'], 'quality': prediction+3}) # (-3)+3
    to_csv(submission)
model = XGBClassifier(**search.best_params_)
model.fit(X, y)
# make_submission(model)