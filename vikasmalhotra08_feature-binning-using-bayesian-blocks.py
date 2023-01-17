# Define our test distributions: a mix of Cauchy-distributed variables

import numpy as np

from scipy import stats



np.random.seed(0)



x = np.concatenate([stats.cauchy(-5, 1.8).rvs(500),

                    stats.cauchy(-4, 0.8).rvs(2000),

                    stats.cauchy(-1, 0.3).rvs(500),

                    stats.cauchy(2, 0.8).rvs(1000),

                    stats.cauchy(4, 1.5).rvs(500)])



# Truncate values to a reasonable range:

x = x[(x > -15) & (x < 15)]
import pylab as pl

pl.hist(x, normed=True)
pl.hist(x, bins=100, normed=True)
def bayesian_blocks(t):

    """Bayesian Blocks Implementation



    By Jake Vanderplas.  License: BSD

    Based on algorithm outlined in http://adsabs.harvard.edu/abs/2012arXiv1207.5578S



    Parameters

    ----------

    t : ndarray, length N

        data to be histogrammed



    Returns

    -------

    bins : ndarray

        array containing the (N+1) bin edges



    Notes

    -----

    This is an incomplete implementation: it may fail for some

    datasets.  Alternate fitness functions and prior forms can

    be found in the paper listed above.

    """

    # copy and sort the array

    t = np.sort(t)

    N = t.size



    # create length-(N + 1) array of cell edges

    edges = np.concatenate([t[:1],

                            0.5 * (t[1:] + t[:-1]),

                            t[-1:]])

    block_length = t[-1] - edges



    # arrays needed for the iteration

    nn_vec = np.ones(N)

    best = np.zeros(N, dtype=float)

    last = np.zeros(N, dtype=int)



    #-----------------------------------------------------------------

    # Start with first data cell; add one cell at each iteration

    #-----------------------------------------------------------------

    for K in range(N):

        # Compute the width and count of the final bin for all possible

        # locations of the K^th changepoint

        width = block_length[:K + 1] - block_length[K + 1]

        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]



        # evaluate fitness function for these possibilities

        fit_vec = count_vec * (np.log(count_vec) - np.log(width))

        fit_vec -= 4  # 4 comes from the prior on the number of changepoints

        fit_vec[1:] += best[:K]



        # find the max of the fitness: this is the K^th changepoint

        i_max = np.argmax(fit_vec)

        last[K] = i_max

        best[K] = fit_vec[i_max]



    #-----------------------------------------------------------------

    # Recover changepoints by iteratively peeling off the last block

    #-----------------------------------------------------------------

    change_points =  np.zeros(N, dtype=int)

    i_cp = N

    ind = N

    while True:

        i_cp -= 1

        change_points[i_cp] = ind

        if ind == 0:

            break

        ind = last[ind - 1]

    change_points = change_points[i_cp:]



    return edges[change_points]
from matplotlib import pyplot



# plot a standard histogram in the background, with alpha transparency

H1 = pyplot.hist(x, bins=200, histtype='stepfilled',

          alpha=0.2, normed=True)

# plot an adaptive-width histogram on top

H2 = pyplot.hist(x, bins=bayesian_blocks(x), color='black',

          histtype='step', normed=True)
from astropy import stats

import multiprocessing as mp

from functools import reduce



def variable_to_bin(var, df_train):

    

    # Lets calculate bin values for a particular column in the dataframe passed to this function

    bin_values = stats.bayesian_blocks(df_train[var],

                                      fitness='events',

                                      p0=0.01)

    

    # Lets create labels for bin values so as to use these labels in dataframe 

    labels = []

    for i, x in enumerate(bin_values):

        labels.append(i)

    

    # delete the last bin label 

    del labels[-1]



    # create a new dataframe to 

    df = pd.DataFrame(index=df_train.index)



    df["ID_code"] = df_train["ID_code"]

    df['new' + var] = pd.cut(df_train[var], 

                               bins = bin_values, 

                               labels = labels)

    

    df.set_index('ID_code')

    

    # Lets delete the bin values and labels to some some space.

    del bin_values, labels

    

    return df
def get_new_feature_train():

    features = [c for c in df_train.columns if c not in ["ID_code", "target"]]



    # Use below line to test whether the binning works or not. 

    #features = ('var_2', 'var_3')

    

    new_df = pd.DataFrame()

    

    # Lets create a multi processing pool but N - 4 CPU's = 4 less CPU's then what your machine has.

    # My machine has 16 CPU's, but I wanted to use 12 of them for calculating bins. 

    pool = mp.Pool(mp.cpu_count() - 4)

    

    # Lets map each CPU to each variable coming out of features list. This line helps in parallel computation

    # of bayesian block bins. 

    results = pool.map(variable_to_bin, features)

    

    pool.close()

    pool.join()

    

    # Lets reduce the series coming out of variable_to_bin function and create a new dataframe.

    results_df = reduce(lambda x, y: pd.merge(x, y, on = 'ID_code'), results)



    return results_df
df_train.set_index('ID_code')

train_df = pd.DataFrame(index=df_train.index)

train_df = get_new_feature_train()
from sklearn.preprocessing import LabelEncoder 



features = [c for c in train_df.columns if c not in ["ID_code", "target"]]



lbl_enc = LabelEncoder()

lbl_enc.fit(train_df[features])



df_Train_cat = lbl_enc.transform(train_df[features])