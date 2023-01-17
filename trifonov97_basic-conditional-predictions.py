import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
MAX_DAY = 1099

DAYS_IN_WEEK = 7

DAYS_IN_MONTH = 31
df = pd.read_csv("../input/train.csv")

df.visits = df.visits.apply(lambda x: np.fromiter((int(i) for i in x.split()), dtype=int))
visits = df.visits.values
all_visits_by_week_day = [x % DAYS_IN_WEEK for v in visits for x in v]
plt.hist(all_visits_by_week_day)
def get_week_visits(visits):

    res = np.zeros((visits.shape[0], MAX_DAY // DAYS_IN_WEEK, DAYS_IN_WEEK))

    for i, id_visits in enumerate(visits):

        for visit in id_visits:

            week, week_day = divmod(visit - 1, DAYS_IN_WEEK)

            res[i][week][week_day] = 1

    

    return res
week_visits = get_week_visits(visits)
week_day_probs = np.sum(week_visits, axis=1) / week_visits.shape[1]
preds = []



for probs in week_day_probs:

    next_visit_prob = [0] * DAYS_IN_WEEK

    for i in range(DAYS_IN_WEEK):

        i_prob = probs[i]

        for j in range(i):

            i_prob *= (1 - probs[j])

        next_visit_prob[i] = i_prob

    preds.append(np.argmax(next_visit_prob) + 1)
with open("subm.csv", "w") as f:

    print("id,nextvisit", file=f)

    for i, p in enumerate(preds):

        print("{}, {}".format(i+1, p), file=f)