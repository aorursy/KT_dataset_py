import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



train = pd.read_csv('../input/liverpool-ion-switching/train.csv')



tr2 = train.iloc[3643990:3644009]



print(tr2)
plt.plot(tr2['signal'])



plt.plot((tr2['open_channels']+1)*1.27)



plt.show()