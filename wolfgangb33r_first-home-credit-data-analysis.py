import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
application = pd.read_csv('../input/application_train.csv')
print("Overall total applicants: %d" % application.size)

print(application.dtypes)
print(application.groupby(['TARGET']).TARGET.count())
print(application['NAME_EDUCATION_TYPE'].head(10))
print(application['FLAG_MOBIL'].head(10)) 
import matplotlib.pyplot as plt

ed = application.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
u_ed = application.NAME_EDUCATION_TYPE.unique()
plt.figure(figsize=(15, 3))
plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
#plt.bar(ages, counts['F'], bottom=counts['M'], color='pink', label='F')
plt.legend()
plt.xlabel('Education level')
plt.plot()

import matplotlib.pyplot as plt

ed = application.groupby('CNT_CHILDREN').CNT_CHILDREN.count()
u_ch = application.CNT_CHILDREN.unique()
plt.figure(figsize=(10, 6))
plt.bar(u_ch, ed, bottom=None, color='green', label='Children count')
plt.legend()
plt.xlabel('Children count')
plt.plot()
ed = application.groupby('NAME_CONTRACT_TYPE').NAME_CONTRACT_TYPE.count()
u_ct = application.NAME_CONTRACT_TYPE.unique()
plt.figure(figsize=(5, 3))
plt.bar(u_ct, ed, bottom=None, color='grey', label='Contract type count')
plt.legend()
plt.xlabel('Contract type count')
plt.plot()

ed = application.groupby(['TARGET', 'CNT_CHILDREN'])['TARGET'].count().unstack('TARGET').fillna(0)
ed.plot(kind='bar', stacked=True)
print(ed)
application['income_bins'] = pd.cut(application['AMT_INCOME_TOTAL'], range(0, 1000000, 10000))

ed = application.groupby(['TARGET', 'income_bins'])['TARGET'].count().unstack('TARGET').fillna(0)
ed.plot(kind='bar', stacked=True, figsize=(50, 15))
print(ed)
