import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from pathlib import Path



print(os.listdir("../input"))



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
DATA_ROOT = Path("../input")



test_path = DATA_ROOT / "test_stage_1.tsv"

test_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"

dev_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv"

val_path = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv"
devdf = pd.read_csv(dev_path, delimiter="\t")

devdf.shape

devdf.head()
valdf = pd.read_table(val_path, delimiter="\t")

valdf.shape

valdf.head()
testdf = pd.read_table(test_path, delimiter="\t")

testdf.shape

testdf.head()
df = pd.concat([devdf, valdf, testdf])

df.shape

df.head()
filt1 = (df["A-coref"] == True) & (df["B-coref"] == True)

filt2 = (df["A-coref"] == True) | (df["B-coref"] == True)



df["label"] = "NEITHER"

df.loc[df["A-coref"] == True, "label"] = "A"

df.loc[df["B-coref"] == True, "label"] = "B"
print("Cases where both A and B are correct:", filt1.sum())
print("Cases where either A or B is correct:", filt2.sum())
print("Cases where neither A nor B is correct:", df[~filt2].shape[0])
df.label.value_counts()
df["textlen"] = df.Text.str.len()

df["textwords"] = df.Text.str.split().str.len()
df.textlen.describe()



ax = df.textlen.plot.hist(bins=15, figsize=(10, 5))

_ = ax.set_title("Text length distribution")
df[df.textlen < 70].T.to_dict()
df[df.textlen > 1340].T.to_dict()
df.textwords.describe()



ax = df.textwords.plot.hist(bins=15, figsize=(10, 5))

_ = ax.set_title("Text words histogram")
df[df.textwords < 13].T.to_dict()
df[df.textwords > 220].T.to_dict()
ax = df[["textlen", "textwords"]].plot.scatter(x="textlen", y="textwords", figsize=(10, 5))

_ = ax.set_title("text words vs text length")
male_pro = ["his", "he", "He", "him", "His"]

female_pro = ["her", "she", "She", "her", "hers"]



df["gender"] = df.Pronoun.apply(lambda x: "male" if x.lower() in male_pro else "female")
df.gender.value_counts()
df.Pronoun.str.lower().value_counts()
df[df.Pronoun == "hers"]
ax = df[["textlen", "Pronoun-offset"]].plot.scatter(x="Pronoun-offset", y="textlen", figsize=(10, 5))

_ = ax.set_title("text len vs pronoun offset")
(df.textlen / df["Pronoun-offset"]).describe()
df[(df.textlen / df["Pronoun-offset"])>3].shape
((df.textlen / df["Pronoun-offset"])<=2).sum() / df.shape[0]
filt = (df.textlen / df["Pronoun-offset"])<=3

temp = df.loc[filt, ["textlen", "Pronoun-offset"]]

ax = (temp.textlen / temp["Pronoun-offset"]).plot.hist(bins=15, figsize=(10, 5))

_ = ax.set_title("pronoun position to text length ratio distribution")
filt = (df["A-coref"] == True) | (df["B-coref"] == True)



df["label_offset"] = pd.np.nan

df.loc[df["A-coref"] == True, "label_offset"] = df.loc[df["A-coref"] == True, "A-offset"]

df.loc[df["B-coref"] == True, "label_offset"] = df.loc[df["B-coref"] == True, "B-offset"]
df[filt].shape
temp = df.loc[filt, ["Pronoun-offset", "label_offset"]]

temp["label_pronoun_gap"] = temp["label_offset"] - temp["Pronoun-offset"]



temp["label_pronoun_gap"].describe()



ax = temp["label_pronoun_gap"].plot.hist(bins=15, figsize=(10, 5))
(temp.label_pronoun_gap > 0).value_counts()

(temp.label_pronoun_gap > 0).value_counts()*100/temp.shape[0]
ax = temp.plot.scatter(x="Pronoun-offset", y="label_pronoun_gap", figsize=(10, 5))
ax = temp.plot.scatter(x="label_offset", y="label_pronoun_gap", figsize=(10, 5))