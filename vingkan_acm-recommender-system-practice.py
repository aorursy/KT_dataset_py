import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from ast import literal_eval

from ml_metrics import mapk as mapk_score

import pickle
# This is a Kaggle-specific quirk, this will import the item metadata

with open("../input/item_strings.py", "r") as script_infile:

    with open("item_strings.py", "w") as script_outfile:

        script_outfile.write(script_infile.read())

from item_strings import COLUMN_LABELS, READABLE_LABELS, ATTRIBUTES
item_df = pd.read_csv("../input/talent.csv")

item_df["categories"] = item_df["categories"].apply(lambda s: literal_eval(s))

ITEM_NAMES = item_df["name"].values

ITEM_IDS = item_df["id"].values

item_df.head()
csr_train, csr_test, csr_input, csr_hidden = pickle.load(open("../input/train_test_mat.pkl", "rb"))

m_split = [csr.todense() for csr in [csr_train, csr_test, csr_input, csr_hidden]]

m_train, m_test, m_input, m_hidden = m_split

print(len(m_train), len(m_test), len(m_input), len(m_hidden))

pd.DataFrame(m_train, columns=ITEM_NAMES).head()
s_train, s_test, s_input, s_hidden = pickle.load(open("../input/train_test_set.pkl", "rb"))

print(len(s_train), len(s_test), len(s_input), len(s_hidden))
def collaborate_jaccard(recs_train, recs_input, n=10, k=3):

    """

    Collaborative filtering recommender system using Jaccard set similarity.

    params:

        recs_train: list of sets of liked item indices for train data

        recs_input: list of sets of liked item indices for input data

        n: number of items to recommend for each user

        k: number of similar users to base recommendations on

    returns:

        recs_pred: list of lists of recommended item indices,

        with each list sorted in order of decreasing relevance

    """

    recs_pred = []

    for src in recs_input:

        users = []

        for vec in recs_train:

            sim = len(vec.intersection(src)) / (len(vec.union(src)) + 1e-5)

            if sim > 0:

                users.append((sim, vec))

        k_users = min(len(users), k)

        if k_users > 0:

            top_users = sorted(users, key=lambda p: p[0], reverse=True)

            vecs = [vec for (sim, vec) in top_users[0:k_users]]

            opts = dict()

            for user_set in vecs:

                for item in user_set:

                    if item not in src:

                        if item not in opts:

                            opts[item] = 0

                        opts[item] += 1

            ranks = [(opts[i], i) for i in opts]

            top_ranks = sorted(ranks, reverse=True)

            n_recs = min(len(top_ranks), n)

            recs = [i for (s, i) in top_ranks[0:n_recs]]

            recs_pred.append(recs)

        else:

            recs_pred.append([])

    return recs_pred
def uhr_score(recs_true, recs_pred, t=10):

    """

    Computes the User Hit Rate (UHR) score of recommendations.

    UHR = the fraction of users whose top list included at

    least one item also in their hidden set.

    params:

        recs_true: list of sets of hidden items for each user

        recs_pred: list of lists of recommended items, with each list

        t: number of recommendations to use in top set

    """

    if len(recs_true) != len(recs_pred):

        note = "Length of true list {} does not match length of recommended list {}."

        raise ValueError(note.format(len(recs_true), len(recs_pred)))

    scores = []

    for r_true, r_pred_orig in zip(recs_true, recs_pred):

        r_pred = list(r_pred_orig)[0:t]

        intersect = set(r_true).intersection(set(r_pred))

        scores.append(1 if len(intersect) > 0 else 0)

    return np.mean(scores)
n_pred = len(s_input)

n_top = 10

s_pred = collaborate_jaccard(s_train, s_input[0:n_pred], n=n_top, k=30)

uhr = uhr_score(s_hidden[0:n_pred], s_pred, t=n_top)

mapk = mapk_score(s_hidden[0:n_pred], s_pred, n_top)

print("UHR  = {0:.3f}".format(uhr))

print("MAP  = {0:.3f}".format(mapk))
uid = 13

print("For User {}:".format(uid))

print()

print("Given:       {}".format(sorted(s_input[uid])))

print("Recommended: {}".format(sorted(s_pred[uid])))

print("Actual:      {}".format(sorted(s_hidden[uid])))

set_intersect = set(s_pred[uid]).intersection(set(s_hidden[uid]))

n_intersect = len(set_intersect)

n_union = len(set(s_pred[uid]).union(set(s_hidden[uid])))

apk = mapk_score([s_hidden[uid]], [s_pred[uid]], n_top)

jacc = n_intersect / (n_union + 1e-5)

print()

print("Recommendation Hits = {}".format(n_intersect))

print("Average Precision   = {0:.3f}".format(apk))

print("Jaccard Similarity  = {0:.3f}".format(jacc))

print()

print("Successful Recommendations:")

for item_id in set_intersect:

    print("- {} ({})".format(ITEM_NAMES[item_id], "cameo.com/" + ITEM_IDS[item_id]))