import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train_df =pd.read_feather('../input/shelter/train_df')
test_df = pd.read_feather('../input/shelter/test_df')
train_df.head()
test_df.head()
X = train_df.drop(['Outcome1', 'Outcome2'], axis = 1)
y = train_df['Outcome1']
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.43, random_state = 42)
print('training shape: {}'.format(X_train.shape))
print('validation shape: {}'.format(X_val.shape))
print('test shape: {}'.format(test_df.shape))
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(n_estimators=100, n_jobs= -1)
%time rf1.fit(X_train, y_train)
def print_score(model, X_t, y_t, X_v, y_v, oob = False):
    print('Training Score: {}'.format(model.score(X_t, y_t)))
    print('Validation Score: {}'.format(model.score(X_v, y_v)))
    if oob:
        if hasattr(model, 'oob_score_'):
            print("OOB Score:{}".format(model.oob_score_))
print_score(rf1, X_train, y_train, X_val, y_val)
def get_subset(df, train_percent=.6, validate_percent=.2, copy = True, seed=None):
    if copy:
        df_copy = df.copy()
    perm = np.random.RandomState(seed).permutation(df_copy.index)
    length = len(df_copy.index)
    train_end = int(train_percent * length)
    validate_end = int(validate_percent * length) + train_end
    train = df_copy.iloc[perm[:train_end]]
    validate = df_copy.iloc[perm[train_end:validate_end]]
    test = df_copy.iloc[perm[validate_end:]]
    
    return train, validate, test
train_speed, val_speed, test_speed = get_subset(train_df, 0.35, 0.35, seed = 42)
train_speed.head()
train_speed.shape
rf_speed = RandomForestClassifier(n_estimators=100, n_jobs=-1)
X_train_speed = train_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_train_speed = train_speed['Outcome1']
X_val_speed = val_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_val_speed = val_speed['Outcome1']
%time rf_speed.fit(X_train_speed, y_train_speed)
print_score(rf_speed, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
rf_1tree = RandomForestClassifier(n_estimators=1, max_depth=3, bootstrap=False, random_state=23,  n_jobs=-1)
rf_1tree.fit(X_train_speed, y_train_speed)
print_score(rf_1tree, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
from sklearn.tree import export_graphviz
estimator = rf_1tree.estimators_[0]
export_graphviz(estimator, out_file = 'tree.dot', 
                feature_names = X_train_speed.columns, 
                class_names = rf_1tree.classes_,
                rounded = True,
                filled = True,
                precision = 2,
                rotate = True,
                node_ids = True)
import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
from IPython.display import Image
Image(filename = 'tree.png')
train_speed, val_speed, test_speed = get_subset(train_df, 0.5, 0.35)
X_train_speed = train_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_train_speed = train_speed['Outcome1']
X_val_speed = val_speed.drop(['Outcome1', 'Outcome2'], axis = 1)
y_val_speed = val_speed['Outcome1']
rf_deeptree = RandomForestClassifier(n_estimators=1, n_jobs=-1)
rf_deeptree.fit(X_train_speed, y_train_speed)
print_score(rf_deeptree, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
rf_deeptree = RandomForestClassifier(n_estimators=100, n_jobs=-1)
%time rf_deeptree.fit(X_train_speed, y_train_speed)
print_score(rf_deeptree, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
from sklearn.ensemble import ExtraTreesClassifier
etc_deeptree =ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
%time etc_deeptree.fit(X_train_speed, y_train_speed)
print_score(etc_deeptree, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
preds = np.stack([i.predict(X_val_speed) for i in rf_deeptree.estimators_])
print(preds.shape)
print(preds)
def convert_outcome1(col):
    if col == 'Adoption':
        return 0
    if col == 'Died':
        return 1
    if col == 'Euthanasia':
        return 2
    if col == 'Return_to_owner':
        return 3
    if col == 'Transfer':
        return 4

y_val_speed_convert = y_val_speed.apply(convert_outcome1)
import scipy.stats
from sklearn import metrics
plt.plot([metrics.accuracy_score(y_val_speed_convert, np.round(scipy.stats.mode(preds[0:i+1],axis = 0)[0][0])) for i in range(100)])
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
model = RandomForestClassifier(n_estimators=10, n_jobs=-1)
model.fit(X_train_speed, y_train_speed)
print_score(model, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
model.fit(X_train_speed, y_train_speed)
print_score(model, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train_speed, y_train_speed)
print_score(model, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model.fit(X_train_speed, y_train_speed)
print_score(model, X_train_speed, y_train_speed, X_val_speed, y_val_speed)
model = RandomForestClassifier(n_estimators=50, oob_score= True, n_jobs=-1)
model.fit(X_train_speed, y_train_speed)
print_score(model, X_train_speed, y_train_speed, X_val_speed, y_val_speed, oob=True)
rf_all = RandomForestClassifier(n_estimators=50, n_jobs=-1, oob_score=True, random_state=42)
rf_all.fit(X_train, y_train)
print_score(rf_all, X_train, y_train, X_val, y_val, oob=True)
rf_minleaf = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, n_jobs=-1, oob_score=True, random_state=42)
rf_minleaf.fit(X_train, y_train)
print_score(rf_minleaf, X_train, y_train, X_val, y_val, oob=True)
rf_maxfeat = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, max_features=0.3, n_jobs=-1, oob_score=True, random_state=42)
rf_maxfeat.fit(X_train, y_train)
print_score(rf_maxfeat, X_train, y_train, X_val, y_val, oob=True)
rf_maxfeat = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, max_features='sqrt', n_jobs=-1, oob_score=True, random_state=42)
rf_maxfeat.fit(X_train, y_train)
print_score(rf_maxfeat, X_train, y_train, X_val, y_val, oob=True)
rf_maxfeat = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, max_features='log2', n_jobs=-1, oob_score=True, random_state=42)
rf_maxfeat.fit(X_train, y_train)
print_score(rf_maxfeat, X_train, y_train, X_val, y_val, oob=True)
n_estimators = [int(x) for x in range(1,100,5)]
max_features = [float(x) for x in np.linspace(0.1,1,9)]
max_features.append('log2')
max_features.append('sqrt')
min_samples_split = [2,5,8,10,20,25]
min_samples_leaf = [2,5,8,10,20,25]
bootstrap = [True, False]
randomCV_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
rf_random = RandomForestClassifier(n_jobs=-1)
from sklearn.model_selection import RandomizedSearchCV
rf_randomGrid = RandomizedSearchCV(rf_random, param_distributions=randomCV_grid, n_iter = 100, cv = 3, random_state=42,  verbose=1)
rf_randomGrid.fit(X_train, y_train);
rf_randomGrid.best_params_
best_randomCV_tree = rf_randomGrid.best_estimator_
print_score(best_randomCV_tree, X_train, y_train, X_val, y_val)
from sklearn.model_selection import GridSearchCV
CV_grid = {'n_estimators': [60, 65, 70, 75], 
           'max_features': [0.2, 0.3, 0.4, 0.5], 
           'min_samples_split': [16, 18, 20, 22], 
           'min_samples_leaf': [6, 7, 8, 9], 
           'bootstrap': [False]}
rf_grid = RandomForestClassifier(n_jobs=-1)
rf_randomGrid = GridSearchCV(rf_grid, param_grid=CV_grid, cv = 3, verbose = 1)
rf_randomGrid.fit(X_train, y_train)
rf_randomGrid.best_params_
print_score(rf_randomGrid, X_train, y_train, X_val, y_val)
