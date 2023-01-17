import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import _get_covariance, anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

%matplotlib inline
diet = pd.read_excel('ANOVA Data.xlsx', sheet_name = 'Diet')
diet
diet.head()
diet.shape
diet.info()
diet.describe()
diet.isnull().sum()
diet['Diet'].value_counts()
# Question
sns.catplot(x="Diet", y="weight", hue='gender', kind="box", data=diet)
plt.show()

# Havent converted the Diet from object variable to categorical variable.. the end result ie graph is same
sns.catplot(x="Diet", y="weight6weeks", hue='gender', kind="box", data=diet)
plt.show()

# Havent converted the Diet from object variable to categorical variable.. the end result ie graph is same
# Question
formula = 'weight6weeks ~ C(gender)'
model = ols(formula, diet).fit()
aov_table = anova_lm(model)
print(aov_table)
formula = 'weight6weeks ~ C(Diet) '
model = ols(formula, diet).fit()
aov_table = anova_lm(model)
print(aov_table)
formula = 'weight6weeks ~ C(gender) + C(Diet)'
model = ols(formula, diet).fit()
aov_table = anova_lm(model)
print(aov_table)
sns.pointplot(x='Diet', y='weight6weeks', data=diet, hue='gender',ci=None);
