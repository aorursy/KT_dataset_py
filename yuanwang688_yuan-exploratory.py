# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import kagglegym

import numpy as np

from matplotlib import pyplot as plt

plt.plot()
env = kagglegym.make()

observations = env.reset()

df = observations.train
#compute autocorrelation for each id, with a lag of 1 to 20



autocorr_mat = []



for id in df['id'].unique():

    y_df = df['y'][df['id']==id]

    autocorr_vec = []

    for j in range(1, 20):

        autocorr_vec += [y_df.autocorr(lag=j)]

    autocorr_mat += [autocorr_vec]

        
autocorr_mat = np.array(autocorr_mat)
#find the largest autocorrelation

max_i = -1

max_j = -1

curr_abs_max = 0



for i in range(len(autocorr_mat)):

    for j in range(len(autocorr_mat[i])):

        if abs(autocorr_mat[i][j])> abs(curr_abs_max):

            max_i = i

            max_j = j

            curr_abs_max = autocorr_mat[i][j]



print('({},{})=>{}'.format(df['id'].unique()[max_i], max_j, curr_abs_max))
sorted_idx_max = sorted(range(len(autocorr_mat)), key=lambda i: max(autocorr_mat[i]))

id_sorted_max = df['id'].unique()[sorted_idx_max]

autocorr_sorted_max = np.amax(autocorr_mat[sorted_idx_max], axis=1)

plt.plot(id_sorted_max, autocorr_sorted_max)
#distribution of maximum positive autocorrelation (number of id)

autocorr_sorted_max=np.nan_to_num(autocorr_sorted_max)

plt.hist(autocorr_sorted_max)
sorted_idx_min = sorted(range(len(autocorr_mat)), key=lambda i: min(autocorr_mat[i]))

id_sorted_min = df['id'].unique()[sorted_idx_min]

autocorr_sorted_min = np.amin(autocorr_mat[sorted_idx_min], axis=1)

plt.plot(id_sorted_min, autocorr_sorted_min)
#distribution of minimum positive autocorrelation (number of id)

autocorr_sorted_min=np.nan_to_num(autocorr_sorted_min)

plt.hist(autocorr_sorted_min)
corr_1276=df[df['id']==111].corr()

print(corr_1276)