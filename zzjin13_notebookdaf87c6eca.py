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
import sys



sys.setdefaultencoding('utf-8')



import matplotlib.pyplot as plt



plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

x=[0,1,2,3]

y1=[0.645,0.677,0.676,0.674]

y2=[0.307,0.311,0.311,0.399]

y3=[0.222,0.2,0.216,0.24]

y4=[0.634,0.631,0.627,0.682]

y5=[0.588,0.576,0.593,0.621]

y6=[0.656,0.689,0.682,0.742]

plt.figure(figsize=(10, 6))

plt.plot(x,y1,'',label="Timbre")

plt.plot(x,y2,'',label="Pitch")

plt.plot(x,y3,'',label="Beat")

plt.plot(x,y4,'',label="Acoustic")

plt.plot(x,y5,'',label="NN_PCA")

plt.plot(x,y6,'',label="Total")

plt.title('')

plt.legend(loc='upper right')

plt.xticks((0,1,2,3),('LR','SVM','NN','GBDT'))

plt.xlabel('Classifier')

plt.ylabel('Accuracy')

plt.grid(x)

plt.show()
x=[0,1,2,3,4,5]

y1=[0.573,0.576,0.573,0.581,0.585,0.583]

y2=[0.583,0.585,0.588,0.593,0.593,0.592]

y3=[0.589,0.593,0.598,0.603,0.601,0.605]

y4=[0.595,0.597,0.601,0.605,0.604,0.606]

plt.figure(figsize=(10, 6))

plt.plot(x,y1,'',label="n_filters=32")

plt.plot(x,y2,'',label="n_filters=64")

plt.plot(x,y3,'',label="n_filters=128")

plt.plot(x,y4,'',label="n_filters=256")

plt.title('')

plt.legend(loc='upper right')

plt.xticks((0,1,2,3,4,5),('5','7','9','11','13','15'))

plt.xlabel('f_width')

plt.ylabel('Accuracy')

plt.grid(x)

plt.show()
x=[0,1,2,3,4]

y3=[0.603,0.605,0.601,0.602,0.603]

plt.figure(figsize=(10, 6))

plt.plot(x,y3)

plt.title('')

plt.legend(loc='upper right')

plt.xticks((0,1,2,3,4),('1','2','3','4','5'))

plt.xlabel('f_length')

plt.ylabel('Accuracy')

plt.grid(x)

plt.show()