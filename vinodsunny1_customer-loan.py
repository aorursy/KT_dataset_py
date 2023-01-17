
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn  as sns
sns.set_style('darkgrid')
%matplotlib inline
loan = pd.read_excel("/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx")
loan.columns 