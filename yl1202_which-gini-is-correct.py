# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numba import jit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def gini(y, pred):

    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)

    g = g[np.lexsort((g[:,2], -1*g[:,1]))]

    gs = g[:,0].cumsum().sum() / g[:,0].sum()

    gs -= (len(y) + 1) / 2.

    return gs / len(y)



def gini_xgb(pred, y):

    y = y.get_label()

    return 'gini', gini(y, pred) / gini(y, y)



def gini_lgb(preds, dtrain):

    y = list(dtrain.get_label())

    score = gini(y, preds) / gini(y, y)

    return 'gini', score, True



# -------------------------------------------------------------

# for sklearn api

@jit

def eval_gini(y_true, y_prob):

    """

    Original author CPMP : https://www.kaggle.com/cpmpml

    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini



def gini_xgb_sklearn_api(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = eval_gini(labels, preds)

    return [('gini', gini_score)]
pred1 = [0.7, 0.2, 0.1, 0.2, 0.1, 0.2, 0.89, 0.33]

lbl1 = [1, 1, 1, 0, 0, 0, 1, 0]
print(gini(lbl1, pred1) / gini(lbl1, lbl1))

print(eval_gini(lbl1, pred1))
pred2 = [0.7, 0.9, 0.1, 0.2, 0.1, 0.2, 0.89, 0.53]

lbl2 = [1, 1, 1, 0, 0, 0, 1, 0]
print(gini(lbl2, pred2) / gini(lbl2, lbl2))

print(eval_gini(lbl2, pred2))