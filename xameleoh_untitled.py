import numpy as np
import pandas as pd
import sys

from scipy import sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, OneHotEncoder

from xgboost import XGBClassifier


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, est):
        super(TextTransformer, self).__init__()
        self.est = est
        pass
    
    def fit(self, X, y=None):
        self.est.fit(X.ravel())
        return self
    
    def transform(self, X):
        Xs = [
            self.est.transform(X[:,_])
            for _ in range(X.shape[-1])
        ]
        result = (Xs[0]>0).astype(int)
        for _ in range(len(Xs)-1):
            result += (Xs[_+1]>0).astype(int)
        return sp.hstack((
                (result == len(Xs)).astype(float), # Some kind of binary AND
                (result == 1).astype(float) # Binary XOR
            )).tocsr()

class SupervisedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, est, method):
        super(SupervisedTransformer, self).__init__()
        self.est = est
        self.method = method
        pass
    
    def fit(self, X, y=None):
        self.est.fit(X, y)
        return self
    
    def transform(self, X):
        return getattr(self.est, self.method)(X)

class EqNotEqBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super(EqNotEqBinarizer, self).__init__()
        pass
    
    def fit(self, X, y=None):
        assert X.shape[-1] == 2, 'Only two-column arrays'
        self.bin_ = LabelBinarizer(sparse_output=True)
        self.bin_.fit(X.ravel())
        return self
    
    def transform(self, X):
        z = np.zeros((X.shape[0], 3), dtype=int)
        eqmask = X[:,0] == X[:,1]
        noteqmask = X[:,0] != X[:,1]
        z[eqmask,0] = X[eqmask,0]
        z[noteqmask,1] = X[noteqmask,0]
        z[noteqmask,2] = X[noteqmask,1]
        return sp.hstack((
                self.bin_.transform(z[:,0]),
                self.bin_.transform(z[:,1]) + self.bin_.transform(z[:,2])
            ))

def get_file(mode):
    params = {}
    if mode == 'test':
        params['index_col'] = 0
    items = pd.read_csv('../input/ItemInfo_%s.csv' % mode, index_col=0)[[
            'categoryID', 'title', 'locationID', 'metroID', 'lon', 'lat']]
    items['title'] = items['title'].fillna('nan')
    items['metroID'] = items['metroID'].fillna(-1)
    parent_categories = pd.read_csv('../input/Category.csv', index_col=0)
    regions = pd.read_csv('../input/Location.csv', index_col=0)
    items = pd.merge(items, parent_categories, left_on='categoryID', right_index=True, how='inner', sort=False)
    items = pd.merge(items, regions, left_on='locationID', right_index=True, how='inner', sort=False)
    del parent_categories
    del regions

    pr = pd.read_csv('../input/ItemPairs_%s.csv' % mode, **params)
    pr = pd.merge(pr, items, left_on='itemID_1', right_index=True, how='inner', sort=False)
    pr = pd.merge(pr, items, left_on='itemID_2', right_index=True, how='inner', sort=False)
    del items

    print('Columns: ' + str(pr.columns), file=sys.stderr)

    fields = [
        'categoryID_x', 'parentCategoryID_x', 'title_x', 'title_y',
        'locationID_x', 'locationID_y', 'regionID_x', 'regionID_y', 'metroID_x', 'metroID_y',
        'lon_x', 'lon_y', 'lat_x', 'lat_y',
    ]
    if mode == 'train':
        return pr[fields + ['isDuplicate']]
    else:
        return pr[fields]


def get_balanced_train_indices(column="categoryID"):
    from sklearn.utils import shuffle
    
    prtest = pd.read_csv("../input/ItemPairs_test.csv", index_col=0)
    prtest = pd.merge(prtest,
                     pd.read_csv("../input/ItemInfo_test.csv", index_col=0),
                     left_on="itemID_1", right_index=True, how="inner", sort=False)
    catdist = prtest[column].value_counts() / len(prtest)
    del prtest
    
    prtrain = pd.read_csv("../input/ItemPairs_train.csv")
    prtrain = pd.merge(prtrain,
                       pd.read_csv("../input/ItemInfo_train.csv", index_col=0),
                       left_on="itemID_1", right_index=True, how="inner", sort=False)
    
    indices = np.array([])
    trainsize = len(prtrain)
    for cat, dist in catdist.iteritems():
        trcatdist = len(prtrain[prtrain[column] == cat])
        if trcatdist < int(1.0 * dist * trainsize):
            trainsize = int(1.0 * trainsize * trcatdist / (dist * trainsize))
    for cat, dist in catdist.iteritems():
        indices = np.hstack((indices, shuffle(prtrain[prtrain[column] == cat].index, random_state=1)[:int(dist*trainsize)]))
    
    indices = pd.Index(np.sort(indices.astype(int)))
    return indices

    
def _print_shape(X):
    print("SHAPE: ", X.shape, file=sys.stderr)
    return X
    
est = Pipeline([
        ('shape1', FunctionTransformer(_print_shape, validate=False)),
        ('feats', FeatureUnion(transformer_list=[
                    ('categories', Pipeline([
                                ('filter', FunctionTransformer(lambda X: X[:,[0]], validate=False)),
                                ('binarizer', OneHotEncoder()),
                                ('shape1', FunctionTransformer(_print_shape, validate=False)),
                            ])),
                    ('parentCategories', Pipeline([
                                ('filter', FunctionTransformer(lambda X: X[:,[1]], validate=False)),
                                ('binarizer', OneHotEncoder()),
                                ('shape1', FunctionTransformer(_print_shape, validate=False)),
                            ])),
                    ('titles', Pipeline([
                                ('filter', FunctionTransformer(lambda X: X[:,[2,3]], validate=False)),
                                ('titleswitch', TextTransformer(CountVectorizer(binary=True))),
                                ('logreg', SupervisedTransformer(LogisticRegression(C=0.01), 'predict_proba')),
                                ('selector', FunctionTransformer(lambda X: X[:,[1]])),
                                ('shape1', FunctionTransformer(_print_shape, validate=False)),
                            ])),
#                     ('locationID', Pipeline([
#                                 ('filter', FunctionTransformer(lambda X: X[:,[4,5]].astype(int), validate=False)),
#                                 ('binarizer', EqNotEqBinarizer()),
#                                 ('threshold', VarianceThreshold(0.0001)),
#                                 ('shape1', FunctionTransformer(_print_shape, validate=False)),
#                             ])),
                    ('regionID', Pipeline([
                                ('filter', FunctionTransformer(lambda X: X[:,[6,7]].astype(int), validate=False)),
                                ('binarizer', EqNotEqBinarizer()),
                                ('threshold', VarianceThreshold(0.0001)),
                                ('shape1', FunctionTransformer(_print_shape, validate=False)),
                            ])),
#                     ('metroID', Pipeline([
#                                 ('filter', FunctionTransformer(lambda X: X[:,[8,9]].astype(int), validate=False)),
#                                 ('binarizer', EqNotEqBinarizer()),
#                                 ('threshold', VarianceThreshold(0.0001)),
#                                 ('shape1', FunctionTransformer(_print_shape, validate=False)),
#                             ])),
                    ('coords', Pipeline([
                                ('filter', FunctionTransformer(lambda X: X[:,[10,11,12,13]].astype(float), validate=False)),
                                ('shape1', FunctionTransformer(_print_shape, validate=False)),
                            ])),
                ])),
        ('shape2', FunctionTransformer(_print_shape, validate=False)),
        ('est', XGBClassifier()),
    ])

pr = get_file('train')
print('Columns: ' + str(pr.columns), file=sys.stderr)
print('FITTING...', file=sys.stderr)
est.fit(pr.drop('isDuplicate', axis=1).values, pr['isDuplicate'].values)
print('FITTED', file=sys.stderr)

del pr

pr = get_file('test')
print('Columns: ' + str(pr.columns), file=sys.stderr)

pr['probability'] = est.predict_proba(pr.values)[:,1]

pr[['probability']].to_csv('submission.csv')
