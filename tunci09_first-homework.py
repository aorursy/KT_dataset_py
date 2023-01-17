# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")
data.info()
# Scatter Plot 
# x = Age, y = Purchase
data.plot(kind='scatter', x='Product_Category_1', y='Purchase',alpha = 0.5,color = 'red')
plt.xlabel('Product_Category_1')              # label = name of label
plt.ylabel('Purchase')
plt.title('Product_Category_1 - Purchase')  
plt.show()# title = title of plot
data.plot(kind='scatter', x='Product_Category_2', y='Purchase',alpha = 0.5,color = 'red')
plt.xlabel('Product_Category_2')              # label = name of label
plt.ylabel('Purchase')
plt.title('Product_Category_2 - Purchase')
data.plot(kind='scatter', x='Product_Category_3', y='Purchase',alpha = 0.5,color = 'red')
plt.xlabel('Product_Category_3')              # label = name of label
plt.ylabel('Purchase')
plt.title('Product_Category_3 - Purchase')
data.Purchase.plot(kind='hist',bins = 2500,figsize = (12,12) )
plt.show()

