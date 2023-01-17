# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.preprocessing import scale



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



def is_number(s):

    try:

        float(s)

        return True

    except ValueError:

        return False



df = pd.read_csv("../input/complete.csv", error_bad_lines=False, warn_bad_lines=False)



data = []



for index, row in df.iterrows():

    lat = row['latitude']

    lon = row['longitude']

    if is_number(lat) and is_number(lon):

        data.append([float(lon), float(lat)])

        

data = np.array(data)

        

model = KMeans(n_clusters=6).fit(scale(data))



plt.scatter(data[:, 0], data[:, 1], c=model.labels_.astype(float))

plt.show()

# Any results you write to the current directory are saved as output.