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
df = pd.read_csv("../input/data.csv")
df.describe()
df.head()
import matplotlib.pyplot as plt



# the histogram of the data

n, bins, patches = plt.hist(df["radius_mean"], 50, normed=1, alpha=0.7)



plt.xlabel('Radius')

plt.ylabel('Probability')

plt.title('Mean Radius of Cell Nucleus')

plt.grid(True)

plt.show()





# Dividing the data into two categories by diagnosis

malignant = df[df['diagnosis'] == 'M']

benign = df[df['diagnosis'] == 'B']



print('Malignant Group')

print(malignant['radius_mean'].describe())



print('\nBenign Group')

print(benign['radius_mean'].describe())



import matplotlib.mlab as mlab



# Visualising the data

n, bins, patches = plt.hist([malignant['radius_mean'], benign['radius_mean']], 

                            label=['Malignant', 'Benign'], bins=50, range=(0, 30), 

                            normed=1, histtype='stepfilled', alpha=0.7)



plt.xlabel('Radius')

plt.ylabel('Probability')

plt.title('Mean Radius of Cell Nucleus')

plt.grid(True)

plt.legend()

plt.show()
# Performing the t-test

from scipy.stats import ttest_ind



t, prob = ttest_ind(malignant['radius_mean'], benign['radius_mean'], equal_var=False)

print("t-score = %f, p = %f" % (t, prob))