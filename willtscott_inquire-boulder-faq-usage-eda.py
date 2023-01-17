import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/crm_faq_usage_2017.csv')
df.info()
df.describe()
df.head()
sns.pairplot(df)
df = df.sort_values(by='Made Service Request', ascending=False)
plt.figure(figsize=(8, 4))
sns.barplot(data = df.head(20), y = 'Topic', x='Made Service Request')