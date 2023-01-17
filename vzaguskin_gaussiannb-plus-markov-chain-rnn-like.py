import pandas as pd

import numpy as np

from sklearn.metrics import f1_score, classification_report, accuracy_score

from sklearn.naive_bayes import GaussianNB

from tqdm.notebook import tqdm

import os
#from https://www.kaggle.com/miklgr500/ghost-drift-and-outliers

TRAIN_DIR = "/kaggle/input/ion-ghost-drift-removed/"
dftrain = pd.read_csv(os.path.join(TRAIN_DIR, "train_clean_kalman.csv"))

dftrain.head()
gnb = GaussianNB()

gnb.fit(dftrain[["signal"]], dftrain.open_channels)

proba = gnb.predict_proba(dftrain[["signal"]])

pred = np.argmax(proba, axis=1)

f1_score(dftrain.open_channels, pred, average="macro")

MC = np.zeros((11,11))

prev_c = None

for c in tqdm(dftrain.open_channels.values):

    if prev_c is not None:

        MC[c, prev_c] += 1

    prev_c = c

MC_normed = (MC / MC.sum(axis=0))
class RNN():

    def __init__(self, weights):

        self.weights = weights

        self.hidden = None

    def forward(self, x):

        ret = x.copy()

        if self.hidden is not None:

            ret = self.hidden.dot(self.weights) * ret

        self.hidden = ret / ret.sum()

        return ret

    def reset(self):

        self.hidden = None
class BiDirectionalPredict:

    def __init__(self, weights, silent=False):

        self.forwardDirectionRNN = RNN(weights)

        self.backwardDirectionRNN = RNN(weights.T)

        self.silent = silent

    

    def predict_proba(self, samples):

        self.forwardDirectionRNN.reset()

        self.backwardDirectionRNN.reset()

        pred_fwd = [self.forwardDirectionRNN.forward(x) for x in tqdm(samples, disable=self.silent)]

        bred_bkwd = [self.backwardDirectionRNN.forward(x) for x in tqdm(samples[::-1], disable=self.silent)]

        

        bidirect_pred = np.array(pred_fwd) * np.array(bred_bkwd[::-1])

        return bidirect_pred

    

    def predict(self, samples):

        proba = self.predict_proba(samples)

        return proba.argmax(axis=-1)
clf = BiDirectionalPredict(MC_normed, silent=True)
pred_mc = []

for grp in tqdm(np.array_split(proba, 10)):

    pred_grp = clf.predict(grp)

    pred_mc.extend(pred_grp)
print(classification_report(dftrain.open_channels, pred_mc))
f1_score(dftrain.open_channels, pred_mc, average="macro")
dftest = pd.read_csv(os.path.join(TRAIN_DIR, "test_clean_kalman.csv"))
proba_test = gnb.predict_proba(dftest[["signal"]])
pred_test = clf.predict(proba_test)
sub = pd.read_csv(os.path.join("/kaggle/input/liverpool-ion-switching/", "sample_submission.csv"), dtype={"time":object})
sub.open_channels = pred_test
sub.open_channels.value_counts()
sub.to_csv("submission.csv", index=False)