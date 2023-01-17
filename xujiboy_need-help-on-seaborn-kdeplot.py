import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.
data0 = np.random.random((100,2))
data1 = data0*10000 # data range [0-20,000]
data2 = data0*20000 # data range [0-30,000]
data3 = data0*31000 # data range [0-31,000]
data_list = [data1, data2, data3]
ax = [None for i in range(len(data_list))]

fig = plt.figure(figsize=(30,10))
for i,data in enumerate(data_list):
    ax[i] = fig.add_subplot(1,3,i+1)
    ax[i].annotate('largest value in data = {:,.2f}'.format(data.max()), xy=(0.5,1.05), xycoords='axes fraction', ha='center', size=20)
    sns.kdeplot(data[:,0], data[:,1], shade=True, ax=ax[i])
# plt.scatter(data[:,0], data[:,1], marker='x', color='k', axes=ax)
