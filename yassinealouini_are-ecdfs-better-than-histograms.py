import pandas as pd

import matplotlib.pylab as plt

%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

N_FEATURES = 200
def ecdf(s):

    """ An ECDF computation function using pandas methods."""

    value_counts_s = s.value_counts()

    return value_counts_s.sort_index().cumsum().div(len(s))





def optimal_fd_bins(s):

    """ 

    Optimal number of bins using the FD rule of thumb: 

    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule

    """

    # Computeing the interquartile range: 

    # https://en.wikipedia.org/wiki/Interquartile_range

    q1 = s.quantile(0.25)

    q3 = s.quantile(0.75)

    iqr = q3 - q1

    width = 2 * iqr / (len(s) ** 0.33)

    return int((s.max() - s.min()) / width)
for i in range(N_FEATURES):

    col = 'var_' + str(i)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    

    # ECDF

    ecdf(train_df.loc[lambda df: df.target == 0, col]).plot(ax=ax[0], label="0")

    ecdf(train_df.loc[lambda df: df.target == 1, col]).plot(ax=ax[0], label="1")

    ax[0].set_title(f"ECDF for {col}")

    ax[0].legend()

    

    # Histogram

    bins = optimal_fd_bins(train_df[col])

    train_df.loc[lambda df: df.target == 0, col].plot(kind="hist", bins=bins, ax=ax[1], 

                                                      label="0")

    train_df.loc[lambda df: df.target == 1, col].plot(kind="hist", bins=bins, ax=ax[1], 

                                                      label="1")

    ax[1].set_title(f"Freedman–Diaconis histogram for {col}")

    ax[1].legend()      

    

    plt.show()

    fig.clf()