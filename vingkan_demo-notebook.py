import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pickle
# Load module from another directory

import shutil

shutil.copyfile(src="../input/redcarpet.py", dst="../working/redcarpet.py")

from redcarpet import mat_to_sets
item_file = "../input/talent.pkl"

item_records, COLUMN_LABELS, READABLE_LABELS, ATTRIBUTES = pickle.load(open(item_file, "rb"))

item_df = pd.DataFrame(item_records)[ATTRIBUTES + COLUMN_LABELS].fillna(value=0)

ITEM_NAMES = item_df["name"].values

ITEM_IDS = item_df["id"].values

item_df.head()
s_items = mat_to_sets(item_df[COLUMN_LABELS].values)

print("Items", len(s_items))

csr_train, csr_test, csr_input, csr_hidden = pickle.load(open("../input/train_test_mat.pkl", "rb"))

m_split = [np.array(csr.todense()) for csr in [csr_train, csr_test, csr_input, csr_hidden]]

m_train, m_test, m_input, m_hidden = m_split

print("Matrices", len(m_train), len(m_test), len(m_input), len(m_hidden))

s_train, s_test, s_input, s_hidden = pickle.load(open("../input/train_test_set.pkl", "rb"))

print("Sets", len(s_train), len(s_test), len(s_input), len(s_hidden))
like_df = pd.DataFrame(m_train, columns=ITEM_NAMES)

like_df.head()
from redcarpet import mapk_score, uhr_score
help(mapk_score)
help(uhr_score)
from redcarpet import jaccard_sim, cosine_sim
help(jaccard_sim)
help(cosine_sim)
from redcarpet import collaborative_filter, content_filter, weighted_hybrid
help(collaborative_filter)
help(content_filter)
help(weighted_hybrid)
from redcarpet import get_recs
help(get_recs)
n_pred = 100 # len(s_input)

k_top = 10

j_neighbors = 30

s_input_sample = s_input[0:n_pred]

s_hidden_sample = s_hidden[0:n_pred]
print("Strategy: Collaborative")

print("Similarity: Jaccard")

collab_jac = collaborative_filter(s_train, s_input_sample, sim_fn=jaccard_sim, j=j_neighbors)

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(collab_jac), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(collab_jac), k=k_top)))
print("Strategy: Collaborative")

print("Similarity: Cosine")

collab_cos = collaborative_filter(s_train, s_input_sample, sim_fn=cosine_sim, j=j_neighbors)

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(collab_cos), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(collab_cos), k=k_top)))
print("Strategy: Collaborative")

print("Similarity: Hybrid (0.8 * Jaccard + 0.2 * Cosine)")

collab_hybrid = weighted_hybrid([

    (collab_jac, 0.8),

    (collab_cos, 0.2)

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(collab_hybrid), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(collab_hybrid), k=k_top)))
print("Strategy: Content-Based")

print("Similarity: Jaccard")

cont_jac = content_filter(s_items, s_input_sample, sim_fn=jaccard_sim)

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(cont_jac), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(cont_jac), k=k_top)))
print("Strategy: Content-Based")

print("Similarity: Cosine")

cont_cos = content_filter(s_items, s_input_sample, sim_fn=cosine_sim)

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(cont_cos), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(cont_cos), k=k_top)))
print("Strategy: Content-Based")

print("Similarity: Hybrid (0.8 * Jaccard + 0.2 * Cosine)")

cont_hybrid = weighted_hybrid([

    (cont_jac, 0.8),

    (cont_cos, 0.2)

])

print("MAP = {0:.3f}".format(mapk_score(s_hidden_sample, get_recs(cont_hybrid), k=k_top)))

print("UHR = {0:.3f}".format(uhr_score(s_hidden_sample, get_recs(cont_hybrid), k=k_top)))
uid = 3

all_recs = collab_jac

s_pred = get_recs(all_recs)

print("Model: Collaborative Filtering with Jaccard Similarity (j=30)")

print("User: {}".format(uid))

print()

print("Given:       {}".format(sorted(s_input[uid])))

print("Recommended: {}".format(sorted(s_pred[uid])))

print("Actual:      {}".format(sorted(s_hidden[uid])))

set_intersect = set(s_pred[uid]).intersection(set(s_hidden[uid]))

n_intersect = len(set_intersect)

n_union = len(set(s_pred[uid]).union(set(s_hidden[uid])))

apk = mapk_score([s_hidden[uid]], [s_pred[uid]], k_top)

jacc = jaccard_sim(set(s_pred[uid]), set(s_hidden[uid]))

print()

print("Recommendation Hits = {}".format(n_intersect))

print("Average Precision   = {0:.3f}".format(apk))

print("Jaccard Similarity  = {0:.3f}".format(jacc))

print()

print("Successful Recommendations:")

for item_id in set_intersect:

    print("- {} ({})".format(ITEM_NAMES[item_id], "cameo.com/" + ITEM_IDS[item_id]))

print()

print("All Recommendation Scores:")

for i, (item_id, score) in enumerate(all_recs[uid]):

    hit = "Y" if item_id in s_hidden[uid] else " "

    print("{0}. [{3}] ({2:.3f}) {1}".format(str(i + 1).zfill(2), ITEM_NAMES[item_id], score, hit))
from redcarpet import write_kaggle_recs
help(write_kaggle_recs)
# Load hold out set

s_hold_input = pickle.load(open("../input/hold_set.pkl", "rb"))

print("Hold Out Set: N = {}".format(len(s_hold_input)))

s_all_input = s_input + s_hold_input

print("All Input:    N = {}".format(len(s_all_input)))
print("Final Model")

print("Strategy: Collaborative")

print("Similarity: Jaccard")

# Be sure to use the entire s_input

final_scores = collaborative_filter(s_train, s_all_input, sim_fn=jaccard_sim, j=30)

final_recs = get_recs(final_scores)
outfile = "kaggle_submission_collab_jaccard_j30.csv"

n_lines = write_kaggle_recs(final_recs, outfile)

print("Wrote predictions for {} users to {}.".format(n_lines, outfile))