import numpy as np 
import pandas as pd

import os
for root,dirs,files in os.walk('/kaggle/input'):
    for name in files:
        print(os.path.basename(name))
        print(os.path.join(root,name))
print("你好 python")
df=pd.read_excel("/kaggle/input/preclass/data.xlsx") #调用excel文件定义为df
print(type(df))
df
df2=df[1:] #从1开始提取
print(df2)
df3=df["Age"] #只获取年龄
print(df3)
df.Age