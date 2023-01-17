# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## List of predictors to add for ML model

# refund_description > refund of an a priori contracted service (i.e. price formally established before purchase?)

# political_party > number of party members

# political_party > age of party

# political_party > party ideology

# political_party > party position in spectrum
df = pd.read_csv('../input/dirty_deputies_v2.csv')
##Examine variable levels##

print(df.head())
##Visualize spread in spending per deputy using boxplots##

##Warning: informative, but crash kernel sometimes

#dfRefund = pd.concat((df['deputy_name'], df['refund_value']), axis=1)

#dfRefund.boxplot(vert=False, by=['deputy_name'], figsize = (20,100))







##Visualizing which companies pay what

companyRefunds = pd.crosstab(df['company_name'], df['refund_description'])

companyRefunds2 = companyRefunds.iloc[0:(round(companyRefunds.shape[0]*.4)),0:(round(companyRefunds.shape[1]*.4))]

companyRefundsMap = sb.heatmap(companyRefunds2)

companyRefundsMap
##Visualizing which companies pay to which parties
##Visaulizing which companies pay to which politicians
##Visualizing which politicians ask for what refunds
##Visualizing which parties ask for what refunds

partyRefunds = pd.crosstab(df['company_name'], df['refund_description'])

partyRefundsMap = sb.heatmap(companyRefunds)

partyRefundsMap
