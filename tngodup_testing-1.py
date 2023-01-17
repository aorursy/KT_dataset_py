# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

%matplotlib inline
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
import pandas as pd
import matplotlib as plt
loans = pd.read_csv("../input/loan.csv", index_col = 0, low_memory=False)

df = pd.DataFrame(loans)
df["b"] = df["term"].astype('category')
df['b'].unique()



import matplotlib.pyplot as plt
#plt.plot(loans.loan_amnt,loans.funded_amnt)
plt.plot(loans.funded_amnt,loans.funded_amnt)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(loans[['loan_amnt','funded_amnt']].as_matrix(), loans[['int_rate']].as_matrix())
loans[['loan_amnt','funded_amnt']]
type(loans)
from sklearn import datasets
iris = datasets.load_iris()
type(loans)
loans_np = loans[['loan_amnt','funded_amnt']].as_matrix()
loans_np.size()
loans[['int_rate']].as_matrix().s