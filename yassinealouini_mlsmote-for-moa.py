import numpy as np

import pandas as pd

import random

from sklearn.datasets import make_classification

from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTENC
TRAIN_FEATURES_PATH = "../input/lish-moa/train_features.csv"

TRAIN_TARGETS_PATH = "../input/lish-moa/train_targets_scored.csv"

DOSE_MAPPING = {"D1": 0, "D2": 1}

TIME_MAPPING = {24: 0, 48: 2, 72: 3}
train_targets_df = pd.read_csv(TRAIN_TARGETS_PATH)

train_features_df = pd.read_csv(TRAIN_FEATURES_PATH)

train = train_features_df.merge(train_targets_df, on="sig_id")

train = (

    train.loc[lambda df: df["cp_type"] == "trt_cp"]

    .reset_index(drop=True)

    .drop(["cp_type", "sig_id"], axis=1)

)
train["cp_dose"] = train["cp_dose"].map(DOSE_MAPPING)

train["cp_time"] = train["cp_time"].map(TIME_MAPPING)
FEATURES = sorted(train_features_df.drop(["cp_type", "sig_id"], axis=1).columns.tolist())

TARGETS = sorted(train_targets_df.drop("sig_id", axis=1).columns.tolist())
print(len(TARGETS))

print(FEATURES[:5])

print(TARGETS[:5])
def get_tail_label(df: pd.DataFrame, ql=[0.05, 1.]) -> list:

    """

    Find the underrepresented targets.

    Underrepresented targets are those which are observed less than the median occurance.

    Targets beyond a quantile limit are filtered.

    """

    irlbl = df.sum(axis=0)

    irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & ((irlbl < irlbl.quantile(ql[1])))]  # Filtering

    irlbl = irlbl.max() / irlbl

    threshold_irlbl = irlbl.median()

    tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()

    return tail_label



def get_minority_samples(X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):

    """

    return

    X_sub: pandas.DataFrame, the feature vector minority dataframe

    y_sub: pandas.DataFrame, the target vector minority dataframe

    """

    tail_labels = get_tail_label(y, ql=ql)

    index = y[y[tail_labels].apply(lambda x: (x == 1).any(), axis=1)].index.tolist()

    

    X_sub = X[X.index.isin(index)].reset_index(drop = True)

    y_sub = y[y.index.isin(index)].reset_index(drop = True)

    return X_sub, y_sub



def nearest_neighbour(X: pd.DataFrame, neigh) -> list:

    """

    Give index of 10 nearest neighbor of all the instance

    

    args

    X: np.array, array whose nearest neighbor has to find

    

    return

    indices: list of list, index of 5 NN of each element in X

    """

    nbs = NearestNeighbors(n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)

    euclidean, indices = nbs.kneighbors(X)

    return indices



def MLSMOTE(X, y, n_sample, neigh=5):

    """

    Give the augmented data using MLSMOTE algorithm

    

    args

    X: pandas.DataFrame, input vector DataFrame

    y: pandas.DataFrame, feature vector dataframe

    n_sample: int, number of newly generated sample

    

    return

    new_X: pandas.DataFrame, augmented feature vector data

    target: pandas.DataFrame, augmented target vector data

    """

    indices2 = nearest_neighbour(X, neigh=5)

    n = len(indices2)

    new_X = np.zeros((n_sample, X.shape[1]))

    target = np.zeros((n_sample, y.shape[1]))

    for i in range(n_sample):

        reference = random.randint(0, n-1)

        neighbor = random.choice(indices2[reference, 1:])

        all_point = indices2[reference]

        nn_df = y[y.index.isin(all_point)]

        ser = nn_df.sum(axis = 0, skipna = True)

        target[i] = np.array([1 if val > 0 else 0 for val in ser])

        ratio = random.random()

        gap = X.loc[reference,:] - X.loc[neighbor,:]

        new_X[i] = np.array(X.loc[reference,:] + ratio * gap)

    new_X = pd.DataFrame(new_X, columns=X.columns)

    target = pd.DataFrame(target, columns=y.columns)

    return new_X, target





# TODO: Adapt this to MLSMOTE?

# smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)

# X_resampled, y_resampled = smote_nc.fit_resample(X, y)
# Should be a DataFrame

X = train.loc[:, FEATURES]

y = train.loc[:, TARGETS]
X.shape
N_SAMPELS = 1000

N_NEIGHBORS = 5
X_sub, y_sub = get_minority_samples(X, y)  # Getting minority samples of that datframe

X_res, y_res = MLSMOTE(X_sub, y_sub, N_SAMPELS, N_NEIGHBORS)  # Applying MLSMOTE to augment the dataframe
y_res.head()
X_res.head()
X_res["cp_time"].value_counts()
X_res["cp_dose"].value_counts()
y_res.sum()
X_res.mean()
X.mean()
pd.concat([X, X_res]).to_csv("augmented_train_features.csv", index=False)
pd.concat([y, y_res]).to_csv("augmented_train_targets.csv", index=False)