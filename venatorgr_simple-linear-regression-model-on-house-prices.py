# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.formula.api as sm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
model = sm.ols(formula="SalePrice ~ YearBuilt + MSSubClass + LotArea + Street + Neighborhood + Condition2 + BldgType + HouseStyle + OverallQual + RoofMatl + ExterQual + GrLivArea", data=train_data).fit()
results = model.predict(test_data)
result_df=pd.DataFrame(test_data['Id'],columns=["Id"])

result_df['SalePrice'] = results

result_df.to_csv('output.csv',header=True,index_label="Id")