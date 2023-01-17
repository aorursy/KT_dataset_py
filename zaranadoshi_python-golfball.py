import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import _get_covariance, anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

%matplotlib inline
golf = pd.read_excel('ANOVA Data.xlsx', sheet_name = 'Golfball')
golf
golf.head()
golf.shape
golf.info()
golf.describe()
golf['Design'].value_counts()
formula = 'Distance ~ C(Design)'
model = ols(formula, golf).fit()
aov_table = anova_lm(model)
print(aov_table)
sns.pointplot(x='Design', y='Distance', data=golf);
