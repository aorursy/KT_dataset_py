# general python imports and jupyter configuration
import numpy as np
import pandas as pd
import scipy.sparse as sps
import seaborn as sns
import implicit
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split
from implicit.evaluation import train_test_split, mean_average_precision_at_k, precision_at_k
from lightfm.evaluation import precision_at_k as precision_at_k_light, recall_at_k as recall_at_k_light, auc_score, reciprocal_rank
from lightfm.cross_validation import random_train_test_split
from lightfm import LightFM
from sklearn.model_selection import ParameterGrid
from skopt import Optimizer, forest_minimize, gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed
%matplotlib inline
df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None, names=['userId', 'game', 'action', 'hours', 'other'])
df.head()
df.other.unique()
del df['other']
len(df.game.unique()), len(df.userId.unique())
purchase_play = pd.concat([df.assign(purchases=1).groupby('game').purchases.sum(), df.groupby('game').hours.mean()], axis=1)
sns.regplot('purchases', 'hours', data=purchase_play)
sns.regplot('purchases', 'hours', data=purchase_play[(purchase_play.hours < 600) & (purchase_play.purchases < 1000)])
purchase_play.corr()
df.groupby('userId').hours.mean().hist(bins=50)
df.groupby('userId').hours.mean().pipe(lambda h: h[h< 400]).hist(bins=50)
actionmap = { 'purchase': 1, 'play': -1 }
playonly = df.replace('purchase', 1).replace('play', -1).groupby(['userId', 'game']).action.sum()
playonly.hist()
# remove multiple purchase/plays for the same user-game-action tuple
# possible caused by multiple products mapping to the same game string
dupeactionmerge = df.groupby(['userId','game','action']).hours.sum().reset_index()
len(df), len(dupeactionmerge)
# check for further duplicates
len(dupeactionmerge.sort_values(['userId','game','action']).pipe(lambda d: d[d.duplicated()]))
# check for games a user has played but not purchased
actionmap = { 'purchase': 1, 'play': -1 }
playonly = dupeactionmerge.replace('purchase', 1).replace('play', -1).groupby(['userId', 'game']).action.sum()
playonly[playonly == -1]
# weight purchases by a scalar value (e.g 1/10 hr) relative to gameplay hours and merge play/purchase rows
purchase_weight = 10
weighted_purchases = dupeactionmerge.copy()
weighted_purchases['hours'] += purchase_weight
merged = weighted_purchases.groupby(['userId', 'game']).hours.sum().reset_index()
# convert game, userId to categories in preparation for sparse matrix creation
# also add column of ones for unary ratings
renumbered = merged.assign(
    game=merged.game.astype('category'), 
    userId=merged.userId.astype('category'), 
    rating=1
)
cleaned = renumbered
game_map = cleaned.game.cat.categories
user_map = cleaned.userId.cat.categories
len(game_map)
# make Bag-of-words sparse matrix of game name unigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
game_bow = vectorizer.fit_transform(game_map.astype('str'))

# create a sparse matrix with our games encoded in the rows and users encoded in the columns
unary_matrix = sps.coo_matrix(
    (cleaned.rating, (cleaned.game.cat.codes.copy(), cleaned.userId.cat.codes.copy())),
    shape=(len(game_map), len(user_map))
).tocsr()

hours_matrix = sps.coo_matrix(
    (cleaned.hours, (cleaned.game.cat.codes.copy(), cleaned.userId.cat.codes.copy())),
    shape=(len(game_map), len(user_map))
).tocsr()
class EvaluatableModel(object):
    """Base class for models that can be used in our evaluation framework"""
    def evaluate(self):
        pass
train_matrix_unary, test_matrix_unary = train_test_split(unary_matrix, train_percentage=0.8)
train_matrix_hours, test_matrix_hours = train_test_split(hours_matrix, train_percentage=0.8)

class FitRecModel(EvaluatableModel):
    """Base class for models that are evaluated using 'fit' and 'recommend' methods, 
    designed to work with the implicit library's evaluation methods via duck type conformance """
    
    def __init__(self, use_playtime = False):
        self.use_playtime = use_playtime
    def fit(self, train_matrix):
        pass
    def recommend(self, userid, user_items, N):
        pass
    def evaluate(self):   
        train_matrix = train_matrix_hours if self.use_playtime else train_matrix_unary
        test_matrix = test_matrix_hours if self.use_playtime else test_matrix_unary
        self.fit(train_matrix)        
        pak = precision_at_k(self, train_matrix.tocsr(), test_matrix.tocsr(), K=5, show_progress=False)
        return { 'model': model, 'P@k': pak }
class RandomModel(FitRecModel):
    """This baseline predictor predicts random purchases for every user."""
    def fit(self, train_matrix):
        pass
    def recommend(self, userid, user_items, N):
        return [(i, 0,) for i in np.random.randint(0, user_items.shape[1], N)]
class TopModel(FitRecModel):
    """This baseline predictor predicts the most popular purchases for every user."""
    def __init__(self, **params):
        self.max_N = params.pop('max_N', 30)
        super(TopModel, self).__init__(**params)
        
    def fit(self, train_matrix):
        self.tops = np.argpartition(train_matrix.sum(axis=1), -self.max_N, axis=0)[-self.max_N:]
    def recommend(self, userid, user_items, N):
        return [(i, 0,) for i in self.tops[-N:]]
class ImplicitModel(EvaluatableModel):
    """Wrapper class for models from the implicit library.
    Creates a model from the 'ModelClass' init parameter
    and passes all other parameters to it"""
    def __init__(self, **params):
        ModelClass = params.pop('model_class')
        self.use_playtime = params.pop('use_playtime', False)
        self.model = ModelClass(**params)
        
    def evaluate(self):    
        train_matrix = train_matrix_hours if self.use_playtime else train_matrix_unary
        test_matrix = test_matrix_hours if self.use_playtime else test_matrix_unary
        self.model.fit(train_matrix)        
        pak = precision_at_k(self.model, train_matrix.tocsr(), test_matrix.tocsr(), K=5, show_progress=False)
        return { 'model': model, 'P@k': pak }
user_item_train_unary, user_item_test_unary = random_train_test_split(unary_matrix.T, test_percentage=0.2)
user_item_train_unary_csr, user_item_test_unary_csr = user_item_train_unary.tocsr(), user_item_test_unary.tocsr()
user_item_train_hours = user_item_train_unary_csr.multiply(hours_matrix.T).tocoo()
user_item_test_hours = user_item_test_unary_csr.multiply(hours_matrix.T).tocoo()
user_item_train_unary, user_item_test_unary = user_item_train_unary_csr.tocoo(), user_item_test_unary_csr.tocoo()
class LightFMModel(EvaluatableModel):
    """Wrapper class for models from the LightFM library. 
    Takes a loss parameter and optional item_features matrix. """
    def __init__(self, **params):
        self.epochs = params.pop('epochs', 30)
        self.item_features = params.pop('item_features', False)
        self.use_playtime = params.pop('use_playtime', False)
        self.params = params
    
    def fit(self, train_matrix):
        self.model = LightFM(**self.params)
        self.model.fit(
            train_matrix,   
            item_features=self.item_features, 
            sample_weight=user_item_train_hours,
            epochs=self.epochs, 
            num_threads=2
        )
    
    def evaluate(self):
        train_matrix = user_item_train_hours
        test_matrix = user_item_test_hours
        self.fit(train_matrix)
        metric_args = { 
            'model': self.model, 
            'train_interactions': train_matrix, 
            'test_interactions': test_matrix,
            'item_features': self.item_features
        }
        pak = precision_at_k_light(k=10, **metric_args)
        rak = recall_at_k_light(k=10, **metric_args)
        auc = auc_score(**metric_args).mean()
        
        recip_rank = reciprocal_rank(**metric_args)
        
        return {'model': model, 
                'P@k': pak.mean(), 
                'R@k': rak.mean(), 
                'auc': auc.mean(), 
                'recip_rank': recip_rank.mean(), 
                'raw_metrics': {'P@k': pak, 'R@k': rak, 'auc': auc, 'recip_rank': recip_rank,}
               }
# Create initial parameter search grid

lightfm_grid = { 
    'loss': ['warp', 'logistic'],
    'item_features': [None, game_bow], 
    'use_playtime': [True, False], 
    'no_components': [30],
    'epochs': [30]
}
    
implicit_grid = {
    'model_class': [implicit.als.AlternatingLeastSquares, implicit.bpr.BayesianPersonalizedRanking],
    'use_playtime': [False, True], 
    'factors': [30]
}
models = [({ 'type': 'random' }, RandomModel(),), ({ 'type': 'top' }, TopModel(),)]
models += [((params), ImplicitModel(**params),) for params in  list(ParameterGrid(implicit_grid))]
models += [((params), LightFMModel(**params),) for params in  ParameterGrid(lightfm_grid)]
# Evaluate models in search grid
trained = []
for (name, model,) in models: 
    evaluation = model.evaluate()
    print((name, evaluation,))
    trained += [(name, evaluation,)]
trained_df = (pd.DataFrame(trained, columns=['parameters', 'evaluation'])
.pipe(lambda d: pd.concat([pd.DataFrame(list(d.parameters)), pd.DataFrame(list(d.evaluation))], axis=1)))
trained_df.sort_values(['P@k'], ascending=False)
trained_df[['P@k', 'R@k', 'auc', 'recip_rank']].plot()
# Search grid
# reasonable ranges borrowed from https://blog.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
lightfm_search_grid = { 
    'loss': ['warp', 'logistic'],
    'use_item_features': [False, True], 
    'use_playtime': [False, True], 
    'no_components': (20, 200, 'uniform'),
    'learning_rate': (1e-4, 1e-1, 'log-uniform'),
    'alpha': (1e-6, 1e-3, 'log-uniform'),
    'epochs': [100], 
}

# final parameters from ~20 min of search on 12cpu machine
lightfm_final_params = ['warp', False, True, 200.0, 0.04647945835997648, 0.001, 100]
trained = []
def objective(param_list, return_model = False):
    params = dict(zip(lightfm_search_grid.keys(), param_list))
    params['user_alpha'] = params['item_alpha'] = params.pop('alpha')
    params['item_features'] = game_bow if params.pop('use_item_features') else None
    params['epochs'] = int(params['epochs'])
    params['no_components'] = int(params['no_components'])
    use_playtime = params.pop('use_playtime', False)
    model = LightFMModel(**params)
    train_matrix = user_item_train_unary
    test_matrix = user_item_test_unary
    
    model.fit(train_matrix)
    metric_args = { 
        'model': model.model, 
        'train_interactions': train_matrix, 
        'test_interactions': test_matrix,
        'item_features': params['item_features'],
        'k': 10,
        'preserve_rows': False,
    }
    
    rak = recall_at_k_light(**metric_args)
    print("Mean R@K:", rak.mean())
    
    return (rak,model) if return_model else np.sqrt(np.mean(((1-rak) ** 2)))
optimizer = Optimizer(
    dimensions=(lightfm_search_grid).values(),
    random_state=1
)
do_search = False # Set to true to search the grid rather than using predefined final values

best_params = None
if do_search:
    for i in range(10): 
        x = optimizer.ask(n_points=(5))
        y = Parallel(n_jobs=5 , verbose=50)(delayed(objective)(v) for v in x)
        print(list(zip(x,y)))
        optimizer.tell(x, y)
    best = np.argmin(optimizer.yi)
    best_params = optimizer.Xi[best]
    print(best_params, optimizer.yi[best])
else:
    best_params = lightfm_final_params
    
print("Fitting final model with ", list(zip(lightfm_search_grid.keys(), best_params)))
rak, final_model = objective(best_params, True)
plt.title("Per user Recall@K")
plt.hist(rak, bins=50)
