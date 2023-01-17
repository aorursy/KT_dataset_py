# Following:

# http://mailchi.mp/0ed493c9f68b/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576417
import pandas as pd

import matplotlib.pyplot as plt



from scipy.stats import ttest_ind

from scipy.stats import probplot
cereals = pd.read_csv('../input/cereal.csv')
cereals.head(10)
cereals.describe(include="all").transpose()
probplot(cereals["sodium"], dist="norm", plot=plt)
sodium_c = cereals["sodium"][cereals["type"] == "C"]



sodium_h = cereals["sodium"][cereals["type"] == "H"]

ttest_ind(sodium_c, sodium_h, equal_var=False)
sodium_c.hist()
sodium_h.hist()
plt.hist(sodium_c, alpha=0.5, label='cold')

plt.hist(sodium_h, alpha=0.5, label='hot')

plt.legend(loc='upper right')

plt.title("Sodium Content by Cereal Type")