import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score, fowlkes_mallows_score
df = pd.read_table('../input/shuttle.tst', sep=' ', header=None)

df.head()
X = df.iloc[:, :-1]

X.head()
y = df[9]
y.hist()
outliers_mask = df[9].isin([2,3,6,7])
len(df[outliers_mask]) / len(df)
kmeans = KMeans(n_clusters=3, random_state=0)
ground_truth = df[9]
y_pred_with_outliers = kmeans.fit_predict(X)
y_pred_without_outliers = kmeans.fit_predict(X[~outliers_mask])
v_measure_score(y_pred_with_outliers[~outliers_mask], y_pred_without_outliers)
v_measure_score(ground_truth[~outliers_mask], y_pred_without_outliers)
# lets try again with different k in range 1..7
iterations = []
for i in range(100):
    scores = []
    for k in range(1,8):
        kmeans = KMeans(n_clusters=k) # without seeding the random state this time
        y_pred_with_outliers = kmeans.fit_predict(X)
        y_pred_without_outliers = kmeans.fit_predict(X[~outliers_mask])
        v_score = v_measure_score(y_pred_with_outliers[~outliers_mask], y_pred_without_outliers)
        adj_r_score = adjusted_rand_score(y_pred_with_outliers[~outliers_mask], y_pred_without_outliers)
        fm_score = fowlkes_mallows_score(y_pred_with_outliers[~outliers_mask], y_pred_without_outliers)
        scores.append([v_score, adj_r_score, fm_score])
    iterations.append(scores)
# pd.DataFrame(iterations).describe()
# TODO check if a figure would give any more insights here
# plt.figure(figsize=(10,4))
# plt.plot(np.array(iterations)[:, 6])

iterations_as_ndarr = np.array(iterations)
fig, axs = plt.subplots(7, 1, figsize=(8,12), sharex=True, constrained_layout=True)
for i in range(7):
    axs[i].plot(iterations_as_ndarr[:, i, :])
    axs[i].set_title('k = {0}'.format(i+1))
    axs[i].set_ylabel('score')
    axs[i].set_xlabel('iterations')
    axs[i].legend(['v score', 'adj', 'fmi'])
plt.plot(range(1,8), iterations_as_ndarr.mean(axis=0))
plt.legend(['v score', 'adj', 'fmi'])