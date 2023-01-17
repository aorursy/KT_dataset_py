# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
x1 = np.array(np.arange(0,2.5,0.5))
x2 = np.array(np.arange(0,2.5,0.5))
net1=x1-1
net2=-x1+2
net3=x2
net4=-x2+3
o1=np.sign(net1)
o2=np.sign(net2)
o3=np.sign(net3)
o4=np.sign(net4)
out=o1+o2+o3+o4-3.5
o5=np.sign(out)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize = (8,5))
ax = plt.axes(projection = "3d")
dx = np.array([1 for i in range(5)])
dy = np.array([1 for i in range(5)])
dz = []
for i in range(5):
    if o5[i] == 1:
        dz.append(-2.5)
    else:
        dz.append(0)
ax.bar3d(x1, x2, o5, dx, dy, dz)
ax.set_zlim3d(-1,1.5)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel("Output")
