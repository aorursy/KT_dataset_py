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
df = pd.read_csv("../input/movie_metadata.csv")

print("data has {} samples with {} features".format(*df.shape))
df.info()
df.head()
for feature in df.keys():

    printdf[df[feature] == 0.0 or df[feature].isnull]



df1.head()
df1.keys()
def outlier(data):

    for feature in df1.key():

        Q1 = np.percentile(data[feature],25)

        Q3 = np.percentile(data[feature],75)

        step = 1.5 * (Q3 - Q1)

    

        print("Data points considered outliers for the feature '{}':".format(feature))

        outlier_data = log_data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]

        display(outlier_data)

        

outlier(df1)

outliers  = []

good_data = data.drop(data.index[outliers]).reset_index(drop = True)