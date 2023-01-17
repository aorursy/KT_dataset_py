import pandas as pd

import numpy as np

import matplotlib.pyplot as plt







logins     = pd.read_csv('../input/logins.csv',parse_dates=['Logged'])

grouped    = logins.groupby(by="Logged").sum()["Purchase"]





plt.plot(grouped.index, grouped.values,marker="D",linewidth=2.0,color="y")



plt.xlabel('Date')

plt.ylabel('Purchases')

plt.title('Purchases by Date')

plt.show()
