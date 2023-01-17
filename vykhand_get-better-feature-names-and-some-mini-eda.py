import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")
test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
#generating names
def get_names(n_names):
    epithets = ["adoring","amazing","angry",
                "blissful","boring","brave",
                "clever","cocky","compassionate"]
    heroes = pd.read_csv("../input/superhero-set/heroes_information.csv")
    heronames = heroes.name.str.lower().str.replace("\s+", "_").str.replace("-+","_").unique()
    namelist = [epi + "_" + nm for epi in epithets for nm in heronames]
    return namelist[:n_names]
new_cols =  get_names(len(train.columns) - 2)
col_map = {c:v for c, v in zip(test.columns.tolist()[1:], new_cols  )}
train.columns = ["ID", "target"] + new_cols
test.columns = ["ID"] + new_cols
colmap_df = pd.DataFrame(pd.Series(col_map))
colmap_df.to_csv("col_map.csv",index=True)
colmap_df.head()
train_desc = train.describe().T
train_desc.head()
test_desc = test.describe().T
test_desc.head()
import seaborn as sns
from matplotlib import pyplot as plt
def plot_dist(varname):
    plt.figure(figsize=(12,8))
    plt.subplot(2,1,1)
    plt.title(f"Distribution of variable {varname} in train")
    sns.distplot(train_desc[varname])
    plt.title(f"Distribution of {varname}  in test")
    plt.subplot(2,1,2)
    sns.distplot(test_desc[varname])
for varname in train_desc.columns[1:]:
    plot_dist(varname)

sns.distplot(train.target)
train.dtypes
train["adoring_adam_strange"][train["adoring_adam_strange"] > 0]
train["clever_venom"][train["clever_venom"] > 0]
sns.distplot(np.log1p(train.target))