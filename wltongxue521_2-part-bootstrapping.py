import pandas as pd
import random
from random import randint
 
oldf=open('/kaggle/input/expedia-hotel-recommendations/test.csv','r',encoding='UTF-8')
newf=open('new_choose.csv','w',encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(1,75342),6000)
lines=oldf.readlines()
newf.write(lines[0])
for i in resultList:
    newf.write(lines[i])
    
oldf.close()
newf.close()
meta_data=pd.read_csv('new_choose.csv')
meta_data.head()
meta_data.info()
meta_data.groupby('is_mobile').count()
meta_data.groupby('is_mobile')['is_package'].mean()
# Creating an list with bootstrapped means for each AB-group
boot_1d = []
for i in range(1000):
    boot_mean = meta_data.sample(frac = 1,replace = True).groupby('is_mobile')['is_package'].mean()
    boot_1d.append(boot_mean)
    
# Transforming the list to a DataFrame
boot_1d = pd.DataFrame(boot_1d)
    
# A Kernel Density Estimate plot of the bootstrap distributions
boot_1d.plot(kind='density')
boot_1d.info()
# Adding a column with the % difference between the two AB-groups
boot_1d['diff'] = (boot_1d[0] - boot_1d[1])/boot_1d[1]*100

# Ploting the bootstrap % difference
ax = boot_1d['diff'].plot(kind='density')
ax.set_title('% difference in is_package between the two AB-groups')

# Calculating the probability that 1-day retention is greater when the gate is at level 30
print('Probability that click/booking is worse when use mobile connection:',(boot_1d['diff'] > 0).mean())