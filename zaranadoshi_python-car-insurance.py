import pandas as pd
import os
import csv
import seaborn as sn
from statsmodels.formula.api import ols
from statsmodels.stats.anova import _get_covariance,anova_lm
import matplotlib.pyplot as plt
from scipy.stats import f_oneway 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
car = pd.read_excel('ANOVA Data.xlsx', sheet_name = 'Car Insurance')
car
car.describe()
car.info()

car.rename(columns={'New York': 'NewYork'}, inplace=True)
car
sn.distplot(car['Atlanta'],label='AT')
sn.distplot(car['Chicago'],label='CH')
sn.distplot(car['Houston'],label='HO')
sn.distplot(car['NewYork'],label='NY')
sn.distplot(car['Memphis'],label='ME')
sn.distplot(car['Philadelphia'],label='PH')
plt.legend();
car = pd.DataFrame(car.stack())
car = car.reset_index()
car = car.drop(columns="level_0")
car.columns = ['City','Interest']
car.head()
formula = 'Interest ~ C(City)'
model = ols(formula, car).fit()
aov_table = anova_lm(model)
print(aov_table)
mc = MultiComparison(car['Interest'], car['City'])
mc_results = mc.tukeyhsd()
print(mc_results)
SSW=aov_table['sum_sq'][0]
SSB=aov_table['sum_sq'][1]
SST= SSW+SSB
print('Sum of Squared Total',SST)
