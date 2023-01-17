import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from matplotlib.patches import Shadow

from matplotlib import cm

from scipy import stats

import seaborn as sb

%matplotlib inline

df = pd.read_csv('../input/all-rows-5-from-50/allrows5from50.csv',index_col=0)
df.head()
df.info()
# Binomial coefficient, n über k



def binomial(n, k):

    ''' Binomial coefficient, n über k 

        n! / (k! * (n - k)!)

        $\binom{n}{k}$

    '''

    p = 1    

    for i in range(1, min(k, n - k) + 1):

        p *= n

        p //= i

        n -= 1

    return p
# Permutationen anzeigen

binomial(50,5)
# 1,2

# 1,3

# 1,4

# 1,5

# 1,6

# 1,7

# 1,8

# 1,9

# 1,10

# 2,3

# 2,4

# ...

# 9,10

binomial(10,2)
df3_8 = df[(df['z1'] == 3) & (df['z2'] == 8) |

   (df['z1'] == 3) & (df['z3'] == 8) |

   (df['z1'] == 3) & (df['z4'] == 8) |

   (df['z1'] == 3) & (df['z5'] == 8) |

   (df['z2'] == 3) & (df['z3'] == 8) |

   (df['z2'] == 3) & (df['z4'] == 8) |

   (df['z2'] == 3) & (df['z5'] == 8) |

   (df['z3'] == 3) & (df['z4'] == 8) |

   (df['z3'] == 3) & (df['z5'] == 8) |

   (df['z4'] == 3) & (df['z5'] == 8) ] 

df3_8.head(30)
df3_8.info()
df3_8.to_csv('AllRows5f50_3_8.csv', sep=';')
df38 = pd.read_csv('AllRows5f50_3_8.csv',index_col=0,sep=';')

df38.head()
plt.figure(figsize=(19,6))

plt.xlim(-5, 56)

plt.xticks(np.arange(50))

sb.distplot(df38,bins=50)
plt.figure(figsize=(19,6))

df38['z1'].hist()
plt.figure(figsize=(19,6))

df38['z2'].hist()
plt.figure(figsize=(19,6))

df38['z3'].hist()
plt.figure(figsize=(19,6))

df38['z4'].hist()
plt.figure(figsize=(19,6))

df38['z5'].hist()
df38.plot.area(alpha=.4)
#df38[['z1','z2']].plot.bar(stacked=True,figsize=(19,6))
df38[['z1','z2']].plot.density(figsize=(19,6))
sb.pairplot(df38)
plt.figure(figsize=(19,6))

sb.boxplot(data=df38,palette='rainbow')
df38.corr()
plt.figure(figsize=(19,15))

sb.heatmap(df38.corr(),annot=True)
df3_8_10 = df[(df['z1'] == 3) & (df['z2'] == 8) & (df['z3']==10) |

   (df['z1'] == 3) & (df['z2'] == 8) & (df['z4']==10) |

   (df['z1'] == 3) & (df['z2'] == 8) & (df['z5']==10) |

   (df['z1'] == 3) & (df['z3'] == 8) & (df['z4']==10) |

   (df['z1'] == 3) & (df['z3'] == 8) & (df['z5']==10) |

   (df['z1'] == 3) & (df['z4'] == 8) & (df['z5']==10) |

   (df['z2'] == 3) & (df['z3'] == 8) & (df['z4']==10) |

   (df['z2'] == 3) & (df['z3'] == 8) & (df['z5']==10) |

   (df['z2'] == 3) & (df['z4'] == 8) & (df['z5']==10) |

   (df['z3'] == 3) & (df['z4'] == 8) & (df['z5']==10) ]
df3_8_10.head(30)
df3_8_10.info()
df3_8_10_27 = df[(df['z1'] == 3) & (df['z2'] == 8) & (df['z3']==10) & (df['z4']==27) | 

   (df['z1'] == 3) & (df['z2'] == 8) & (df['z3']==10) & (df['z5']==27) | 

   (df['z1'] == 3) & (df['z2'] == 8) & (df['z4']==10) & (df['z5']==27) | 

   (df['z1'] == 3) & (df['z3'] == 8) & (df['z4']==10) & (df['z5']==27) | 

   (df['z2'] == 3) & (df['z3'] == 8) & (df['z4']==10) & (df['z5']==27) ]
df3_8_10_27.head()
df3_8_10_27.info()