%pylab inline



import pandas as pd

import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau, shapiro, pointbiserialr

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/glass.csv')

print(data.shape)
data.sample(5)
col_names, cn = data.columns, 0

fig, ax = plt.subplots(3, 3, sharey=True, figsize=(9,9))

for row in ax:

    for cell in row:

        cell.hist(data[col_names[cn]], label=col_names[cn], bins=15)

        norm = shapiro(data[col_names[cn]])[1]>0.05 # Shapiro-Wilk test for normality

        cell.set_title('normal dist: ' + str(norm))

        cell.legend()

        cn+=1
col_names, cn = data.columns, 0

fig, ax = plt.subplots(3, 3, figsize=(9,9))

for row in ax:

    for cell in row:

        sv = data[col_names[cn]].sort_values().tolist()

        cell.plot(sv, label=col_names[cn])

        cell.legend()

        cn+=1
plt.figure(figsize=(9,4))

plt.title("Pearson's correlation")

sns.heatmap(abs(data.drop('Type', axis=1).corr(method='pearson')), cmap='Blues', annot=True, cbar=False)
plt.figure(figsize=(9,4))

plt.title("Spearman's correlation")

sns.heatmap(abs(data.drop('Type', axis=1).corr(method='spearman')), cmap='Blues', annot=True, cbar=False)
plt.figure(figsize=(9,4))

plt.title("Kendall's tau correlation")

sns.heatmap(abs(data.drop('Type', axis=1).corr(method='kendall')), cmap='Blues', annot=True, cbar=False)
def plot_colored(ax, X, Y, target):

    labels = target.unique()

    for c, l in zip("rgbcmykw",labels):

        ax.scatter(X[target == l], Y[target == l], c=c, alpha=0.6, s=100)
fig, ax = plt.subplots(2,2, figsize=(9,9))



plot_colored(ax[0,0], data.RI, data.Ca, data.Type)

ax[0,0].set_title(

    'P, S, K: ' + str(('%.2f' % pearsonr(data.RI, data.Ca)[0], 

                       '%.2f' % spearmanr(data.RI, data.Ca)[0],

                       '%.2f' % kendalltau(data.RI, data.Ca)[0]))

    )



a, b = 12, 12

A, B = data.RI[data.Ca < a], data.Ca[data.Ca < b]

plot_colored(ax[0,1], A, B, data.Type)

ax[0,1].set_title(

    'P, S, K: ' + str(('%.2f' % pearsonr(A,B)[0], 

                       '%.2f' % spearmanr(A,B)[0], 

                       '%.2f' % kendalltau(A,B)[0]))

    )





a, b = 7.5, 10.5

A = data.RI[(data.Ca > a) & (data.Ca < b)]

B = data.Ca[(data.Ca > a) & (data.Ca < b)]

plot_colored(ax[1,0], A, B, data.Type)

ax[1,0].set_title(

    'P, S, K: ' + str(('%.2f' % pearsonr(A,B)[0], 

                       '%.2f' % spearmanr(A,B)[0], 

                       '%.2f' % kendalltau(A,B)[0]))

    )





a, b = 7.6, 9.5

A = data.RI[(data.Ca > a) & (data.Ca < b) & (data.RI >1.515) & (data.RI < 1.52)]

B = data.Ca[(data.Ca > a) & (data.Ca < b) & (data.RI >1.515) & (data.RI < 1.52)]

plot_colored(ax[1,1], A, B, data.Type)

ax[1,1].set_title(

    'P, S, K: ' + str(('%.2f' % pearsonr(A,B)[0], 

                       '%.2f' % spearmanr(A,B)[0], 

                       '%.2f' % kendalltau(A,B)[0]))

    )



print( 'There is left %.2f percent of data on the last scatter.' % (100.*A.shape[0]/data.shape[0]))
def print_biserialr(X, y, annot=True):

    # X - DataFrame

    # y - dihatom Series

    labels = list(set(y.values))

    shape = list(X.shape)

    shape[0] = len(labels)

    d = np.ndarray(tuple(shape))

    for i, label in enumerate(labels):

        for j, col in enumerate(X.columns):

            d[i, j] = pointbiserialr(y == label, X[col])[0]



    pbr = pd.DataFrame(d, index=labels, columns=X.columns)

    plt.figure(figsize=(9,4))

    sns.heatmap(pbr.abs(), cmap='Blues', annot=annot, cbar=False)

    return pbr
X = data.ix[:,:-1]

y = data.ix[:,-1]
pbr = print_biserialr(X,y)
def get_bootstrap_samples(data, n_samples):

    indices = np.random.randint(0, len(data), n_samples)

    samples = data.ix[indices]

    samples = samples.set_index(np.arange(samples.shape[0]))

    return samples
def balance(data, target):

    # data - pandas.DataFrame

    # target - string for target column

    counts = data[target].value_counts()

    m = counts.max()

    balanced = data.copy()

    for i in counts.index:

        new_index = data[data[target]==i].shape[0]

        df = data[data[target]==i].set_index(np.arange(new_index))

        balanced = balanced.append(get_bootstrap_samples(df, m - counts[i]))

    balanced = balanced.set_index(np.arange(balanced.shape[0]))

    return balanced
balanced = balance(data, 'Type')
fig, ax = plt.subplots(1,2, figsize=(7,3))

ax[0].hist(data.Type)

ax[0].set_title('unbalanced data')

ax[1].hist(balanced.Type)

ax[1].set_title('balanced data')

plt.title
Xb = balanced.ix[:,:-1]

yb = balanced.ix[:,-1]
pbr = print_biserialr(X,y)

plt.title('Biserial coefficient for non balanced data')
pbr = print_biserialr(Xb,yb)

plt.title('Biserial coefficient for balanced data')