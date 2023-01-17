# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# Load Dataset
data = pd.read_csv('../input/data.csv')
data.dtypes
# Convert some columns to category type
data.set_index('shot_id', inplace=True)
data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')
data.head(2)
data.dtypes
data.shape
# Take a look at numerical columns statistics
data.describe(include=["number"])
# Then for the categorical columns
data.describe(include=['object','category'])
# Virtualize shot distributation
ax = plt.axes()
sns.countplot(x='shot_made_flag', data=data, ax=ax)
ax.set_title("Shot Made Distribuation")
plt.show()
# Describle analysisi based on different features

f, axarr = plt.subplots(8, figsize=(15, 25))

sns.countplot(x="combined_shot_type", hue="shot_made_flag", data=data, ax=axarr[0])
sns.countplot(x="season", hue="shot_made_flag", data=data, ax=axarr[1])
sns.countplot(x="period", hue="shot_made_flag", data=data, ax=axarr[2])
sns.countplot(x="playoffs", hue="shot_made_flag", data=data, ax=axarr[3])
sns.countplot(x="shot_type", hue="shot_made_flag", data=data, ax=axarr[4])
sns.countplot(x="shot_zone_area", hue="shot_made_flag", data=data, ax=axarr[5])
sns.countplot(x="shot_zone_basic", hue="shot_made_flag", data=data, ax=axarr[6])
sns.countplot(x="shot_zone_range", hue="shot_made_flag", data=data, ax=axarr[7])

axarr[0].set_title("Combined shot type")
axarr[1].set_title("Season")
axarr[2].set_title("Period")
axarr[3].set_title("Playoffs")
axarr[4].set_title("Shot type")
axarr[5].set_title("Shot Zone Area")
axarr[6].set_title("Shot Zone Basic")
axarr[7].set_title("Shot Zone Range")

plt.tight_layout()
plt.show()
# Store Shot made flag is null
unknown_mask = data["shot_made_flag"].isnull()

# Make a copy of data
data_cp = data.copy()
target = data_cp["shot_made_flag"].copy()

# Remove unrelated columns

data_cp.drop("team_id", axis=1, inplace=True) # Only one number
data_cp.drop("lat", axis=1, inplace=True) # Correlated to loc_x
data_cp.drop("lon", axis=1, inplace=True) # Correlated to loc_y
data_cp.drop("game_id", axis=1, inplace=True) # Independent 
data_cp.drop("game_event_id", axis=1, inplace=True) # Independent
data_cp.drop("team_name", axis=1, inplace=True) # Only LA lakers
data_cp.drop("shot_made_flag", axis=1, inplace=True) # For predict
data_cp.dtypes
# Function to Remove outliers
def detect_outliers(series, whis=1.5):
    q75, q25 = np.percentile(series, [75,25])
    iqr = q75 - q25
    return ~((series - series.median()).abs() <= (whis * iqr))
data_cp["seconds_from_period_end"] = 60 * data_cp["minutes_remaining"] + data_cp["seconds_remaining"]
data_cp["last_5_sec_in_period"] = data_cp["seconds_from_period_end"] < 5

data_cp.drop("minutes_remaining", axis=1, inplace=True)
data_cp.drop("seconds_remaining", axis=1, inplace=True)
data_cp.drop("seconds_from_period_end", axis=1, inplace=True)

data_cp["home_play"] = data_cp["matchup"].str.contains('vs').astype('int')
data_cp.drop('matchup', axis=1, inplace=True)

data_cp['game_date'] = pd.to_datetime(data_cp['game_date'])
data_cp['game_year'] = data_cp['game_date'].dt.year
data_cp['game_month'] = data_cp['game_date'].dt.month
data_cp.drop('game_date', axis=1, inplace=True)
data_cp['loc_x'] = pd.cut(data_cp['loc_x'], 25)
data_cp['loc_y'] = pd.cut(data_cp['loc_y'], 25)

rare_action_types = data_cp['action_type'].value_counts().sort_values().index.values[:20]
data_cp.loc[data_cp['action_type'].isin(rare_action_types), 'action_type'] = 'Other'
categorial_cols = [
    'action_type', 'combined_shot_type', 'period', 'season', 'shot_type',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'game_year',
    'game_month', 'opponent', 'loc_x', 'loc_y'
]
# Function to get dummies and append to dataset
for cc in categorial_cols:
    dummies = pd.get_dummies(data_cp[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cp.drop(cc, axis=1, inplace=True)
    data_cp = data_cp.join(dummies)
data_submit = data_cp[unknown_mask]

X = data_cp[~unknown_mask]
Y = target[~unknown_mask]
from sklearn.feature_selection import VarianceThreshold, RFE, SelectKBest, chi2
threshold = 0.9
vt = VarianceThreshold().fit(X)
# Select features from Data
feature_var_threshold = data_cp.columns[vt.variances_ > threshold * (1-threshold)]
feature_var_threshold
# Using Random Forest Classifier model
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
model = RandomForestClassifier()
model.fit(X,Y)
# Find important features
feature_imp = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
feature_imp_20 = feature_imp.sort_values("importance", ascending=False).head(20).index
feature_imp_20
# Prepare features for chi2 test
from sklearn.preprocessing import MinMaxScaler

X_minmax = MinMaxScaler(feature_range=(0,1)).fit_transform(X)
X_scored = SelectKBest(score_func=chi2, k="all").fit(X_minmax, Y)

feature_scoring = pd.DataFrame({
    'feature': X.columns,
    'score': X_scored.scores_
})
feature_scored_20 = feature_scoring.sort_values('score', ascending=False).head(20)['feature'].values
feature_scored_20
# Prepare features for Recursive Features Elimination(RFE)
from sklearn.linear_model import LogisticRegression
rfe = RFE(LogisticRegression(), 20)
rfe.fit(X,Y)
feature_rfe_scoring = pd.DataFrame({
    'feature': X.columns,
    'score': rfe.ranking_
})

feature_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] ==1]['feature'].values
feature_rfe_20
# Select final features by merged together
features = np.hstack([
    feature_var_threshold,
    feature_imp_20,
    feature_scored_20,
    feature_rfe_20
])
# Remove duplicate features
features = np.unique(features)
print('Final features set:\n')
for f in features:
    print("\t-{}".format(f))
# Prepare dataset and check the info of the dataset
data_cp = data_cp.ix[:, features]
X = X.ix[:, features]
print('Clean Dataset shapes: {}'.format(data_cp.shape))
print('Subbmitable dataset shape:{}'.format(data_submit.shape))
print('Train features shape:{}'.format(X.shape))
print('Target label shape:{}'.format(Y.shape))
# PCA Visualization
from sklearn.decomposition import PCA, KernelPCA
components = 8
pca = PCA(n_components=components).fit(X)
pca_variance_explained_df = pd.DataFrame({
    "component": np.arange(1, components+1),
    "variance_explained": pca.explained_variance_ratio_
})

ax = sns.barplot(x='component', y='variance_explained', data=pca_variance_explained_df)
ax.set_title("PCA - Variance explained")
plt.show()
X_pca = pd.DataFrame(pca.transform(X)[:,:2])
X_pca['target'] = Y.values
X_pca.columns = ["x", "y", "target"]

sns.lmplot('x','y',
          data=X_pca,
          hue="target",
          fit_reg=False,
          markers=["o","x"],
          palette="Set1",
          size=7,
          scatter_kws={"alpha": .2})
plt.show()
# Prepare and evaluate models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
seed = 7
processors = 1
num_folds = 3
num_instances = len(X)
scoring = 'log_loss'
kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

models =[]
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('K-NN', KNeighborsClassifier(n_neighbors=5)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
# Evaluate each model with Cross_val_score function
from sklearn.cross_validation import KFold, cross_val_score
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=processors)
    results.append(cv_results)
    names.append(name)
    print("{0}:({1:.3f}) +/ ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
    