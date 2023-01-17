# This Python 3 environment comes with many helpful analytics libraries installed

# 这是个python3环境，自带了很多分析库

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# 这是由kaggle/python docker镜像所构建的：https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# 例如，接下来就是为你导入了有用的一堆包



import numpy as np # linear algebra 线代

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 数据处理，csv测试的输入输出 例如pd.read_csv



# Input data files are available in the "../input/" directory.

# 可以将存放在../input/文件夹内的文件输入为数据文件

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# 例如，运行这个（Shift+Enter）将会列出输入文件夹里有什么东东



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# 任何结果你写入这个当前文件夹的内容，都会输出在这个结果内