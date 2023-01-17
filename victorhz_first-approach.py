import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import clear_output
train_df = pd.read_csv('../input/train.csv')
train_df_sorted = train_df.sort_values('Target')
train_df_sorted.head()
plt.plot(train_df_sorted.Target.tolist())
plt.show()
train_df_sorted.Target.tolist().count(4)/len(train_df_sorted.Target.tolist())
plt.plot(train_df_sorted.v2a1.tolist())
plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.hacdor.tolist())*2)
plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.rooms.tolist())/10)
plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.meaneduc.tolist())/20)
plt.plot(train_df_sorted.Target.tolist())
plt.plot(np.array(train_df_sorted.hhsize.tolist())/10)
test_df = pd.read_csv('../input/test.csv')
test_id = test_df.Id.tolist()
pre = []
for i in range(len(test_id)):
    pre.append(4)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre4.csv',index=False)
pre = []
for i in range(len(test_id)):
    pre.append(3)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre3.csv',index=False)
pre = []
for i in range(len(test_id)):
    pre.append(2)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre2.csv',index=False)
pre = []
for i in range(len(test_id)):
    pre.append(1)
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('pre1.csv',index=False)
pre = []
for i in range(len(test_id)):
    pre.append(np.random.randint(1,5))
df = pd.DataFrame({'Id':test_id,'Target':pre})
df.to_csv('prernd.csv',index=False)