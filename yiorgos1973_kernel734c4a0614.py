# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/fuelconsumption/kaggle.csv", index_col=0)

df.head()
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')
X = df.drop(columns='Fuel')

y = df[['Fuel']]
plt.figure(figsize=(12,6))

sns.heatmap(df.corr().apply(abs), annot=True, fmt='1.2f', cmap='coolwarm')

plt.title("Correlation Heatmap");
sns.set_palette('rainbow')

f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(16,4), sharey=True)

for col, ax in zip(['Payload', 'Reliability', 'LoadValue'], (ax1,ax2,ax3)):

    sns.regplot(col, 'Fuel', data=df, marker='.', ax=ax)

plt.suptitle('Univariate Regression Analysis', fontsize='x-large', fontweight='bold');
f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,4), sharey=True)

sns.violinplot(x='Net', y='Fuel', hue='TransmissionType', data=df, ax=ax1)

sns.violinplot(x='Season', y='Fuel', hue='TransmissionType', data=df, ax=ax2)

sns.violinplot(x='Net', y='Fuel', hue='Season', data=df, ax=ax3);
import statsmodels.formula.api as smf
data = df.copy()

data['TransmissionType'] = data.TransmissionType.map({'manual':0, 'automatic':1})
fml = "Fuel ~ Payload + Reliability + C(Season) + Net + LoadValue + C(TransmissionType)" # Net can be handled as  a discrete quantitative variable
model = smf.ols(fml, data).fit()
model.summary()
from statsmodels.graphics.gofplots import qqplot

rsdplot = qqplot(model.resid, line='r')

plt.title('Residual QQ-Plot');