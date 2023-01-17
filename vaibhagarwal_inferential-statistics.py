import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import scipy.stats

from scipy.stats import t

from scipy.special import stdtr

from numpy.random import seed

import seaborn as sns



%matplotlib inline

from matplotlib import rcParams

sns.set_style("whitegrid")

sns.set_context("poster")
matplotlib.rcParams['figure.figsize'] = (8.0, 5.0)
file_1 = pd.read_csv('../input/churn-prediction-of-bank-customers/Churn_Modelling.csv')
df = pd.DataFrame(file_1)
df.head()
df_0 = df[df.Exited == 0]

df_1 = df[df.Exited == 1]
sns.distplot(df_0.Age, color='blue', label='Still with Bank')

sns.distplot(df_1.Age, color='green', label='Left the Bank')

plt.legend()
df_0.Age.mean() , df_0.Age.std()
df_1.Age.mean() , df_1.Age.std()
t_1,p_1 = scipy.stats.ttest_ind(df_0.Age, df_1.Age, equal_var=False)

t_1, p_1
def bs_choice(data, func, size):

    bs_s = np.empty(size)

    for i in range(size):

        bs_abc = np.random.choice(data, size=len(data))

        bs_s[i] = func(bs_abc)

    return bs_s
diff_means = np.mean(df_1.Age) - np.mean(df_0.Age)

mean_age = np.mean(df.Age)

age_shifted_0 = df_0.Age + mean_age - np.mean(df_0.Age)

age_shifted_1 = df_1.Age + mean_age - np.mean(df_1.Age)
bs_n_0 = bs_choice(age_shifted_0, np.std, 10000)

bs_n_1 = bs_choice(age_shifted_1, np.std, 10000)

bs_mean = bs_n_1 - bs_n_0
p = np.sum(bs_mean >= diff_means) / len(bs_mean)

p
sns.distplot(df_0.CreditScore, color='blue', label='Still with bank')

sns.distplot(df_1.CreditScore, color='green', label='Left the bank')

plt.legend()
t_2,p_2 = scipy.stats.ttest_ind(df_0.CreditScore, df_1.CreditScore, equal_var=False)

t_2, p_2
sns.distplot(df_0.Balance, color='blue', label='Still with bank')

sns.distplot(df_1.Balance, color='green', label='Left the bank')

plt.legend()
t_3,p_3 = scipy.stats.ttest_ind(df_0.Balance, df_1.Balance, equal_var=False)

t_3, p_3
sns.distplot(df_0[df_0.Balance != 0].Balance, color='blue', label='Still with bank')

sns.distplot(df_1[df_1.Balance != 0].Balance, color='green', label='Left with bank')

plt.legend()
t_3,p_3 = scipy.stats.ttest_ind(df_0[df_0.Balance != 0].Balance, df_1[df_1.Balance != 0].Balance, equal_var=False)

t_3, p_3
sns.distplot(df_0.EstimatedSalary, color='blue', label='Still with bank')

sns.distplot(df_1.EstimatedSalary, color='green', label='Left with bank')

plt.legend()
t_3,p_3 = scipy.stats.ttest_ind(df_0.EstimatedSalary, df_1.EstimatedSalary, equal_var=False)

t_3, p_3
diff_means = np.mean(df_1.EstimatedSalary) - np.mean(df_0.EstimatedSalary)

mean_salary = np.mean(df.EstimatedSalary)

salary_shifted_0 = df_0.EstimatedSalary + mean_salary - np.mean(df_0.EstimatedSalary)

salary_shifted_1 = df_1.EstimatedSalary + mean_salary - np.mean(df_1.EstimatedSalary)
bs_n_0 = bs_choice(salary_shifted_0, np.mean, 10000)

bs_n_1 = bs_choice(salary_shifted_1, np.mean, 10000)

bs_mean = bs_n_1 - bs_n_0
p = np.sum(bs_mean >= diff_means) / len(bs_mean)

p