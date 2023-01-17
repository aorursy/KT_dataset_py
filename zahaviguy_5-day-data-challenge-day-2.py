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




