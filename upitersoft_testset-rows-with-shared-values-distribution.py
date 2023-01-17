import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data_path = "../input/test-common-rows/"
print(os.listdir("../input"))
print(os.listdir(data_path))


# Any results you write to the current directory are saved as output.
from six.moves import cPickle as pickle
import bz2
import numpy as np
import os

def loadPickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickle(pickle_file, data):
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return

def loadPickleBZ(pickle_file):
    with bz2.BZ2File(pickle_file, 'r') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickleBZ(pickle_file, data):
    with bz2.BZ2File(pickle_file, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return
import matplotlib
import matplotlib.pyplot as plt

rowNumMatches = loadPickleBZ(data_path+'rowNumMatches.pbz')

matches30 = rowNumMatches[:, 4]
matches50 = rowNumMatches[:, 2]
matches80 = rowNumMatches[:, 0]
print('Percent of common values between rows:\t\t\t   30% common\t   50% common\t   80% common')
for lim in [1, 3, 10]:
    print('Rows sharing same values count <= {} :  > {}  '.format(lim, lim), '\t '
          '\t', (matches30 <= lim).sum(), ':', (matches30 > lim).sum(),
          '\t', (matches50 <= lim).sum(), ':', (matches50 > lim).sum(),
          '\t', (matches80 <= lim).sum(), ':', (matches80 > lim).sum(),
          )

fig = plt.figure(figsize=(18, 8))
ax = plt.subplot(111)
ax.set_yscale('log')
plt.plot(rowNumMatches)
plt.show()