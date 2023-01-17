import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
file_1 = pd.read_csv('../input/churn-prediction-of-bank-customers/Churn_Modelling.csv')
df = pd.DataFrame(file_1)
df.shape
df.set_index('CustomerId', inplace=True)

df.head()
df.describe()
#Parting the data into two dataframes containing exited and non exited customers.

df_e0 = df.loc[df.Exited == 0, :]

df_e1 = df.loc[df.Exited == 1, :]
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.distplot(df_e0['Age'])

plt.ylabel('PDF')

plt.title('Still with Bank')

plt.subplot(2,2,2)

sns.distplot(df_e1['Age'])

plt.ylabel('PDF')

plt.title('Left the Bank')

plt.subplot(2,2,3)

sns.boxplot(x= df.Exited, y= df.Age, hue=df.Exited)

plt.subplot(2,2,4)

sns.boxplot(x= df.Gender, y= df.Age, hue=df.Exited)
df.Age.median(), df_e1.Age.median()
sns.countplot(df.Gender, hue=df.Exited)
perc_f_e1 = np.sum(df_e1.Gender == 'Female')*100 / np.sum(df.Gender == 'Female')

perc_m_e1 = np.sum(df_e1.Gender == 'Male')*100 / np.sum(df.Gender == 'Male')

perc_f_e1, perc_m_e1
sns.countplot(df.IsActiveMember, hue=df.Exited)
perc_a0_e1 = np.sum(df_e1.IsActiveMember == 0)*100 / np.sum(df.IsActiveMember == 0)

perc_a1_e1 = np.sum(df_e1.IsActiveMember == 1)*100 / np.sum(df.IsActiveMember == 1)

perc_a0_e1, perc_a1_e1
sns.countplot(df.Geography, hue=df.Exited)
perc_gf_e1 = np.sum(df_e1.Geography == 'France')*100 / np.sum(df.Geography == 'France')

perc_gs_e1 = np.sum(df_e1.Geography == 'Spain')*100 / np.sum(df.Geography == 'Spain')

perc_gg_e1 = np.sum(df_e1.Geography == 'Germany')*100 / np.sum(df.Geography == 'Germany')

perc_gf_e1, perc_gs_e1, perc_gg_e1
df.groupby(['Geography','Gender']).mean()
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.boxplot(x= df.Exited, y= df.Balance, hue=df.Exited)

plt.subplot(2,2,2)

sns.boxplot(x= df.Gender, y= df.Balance, hue=df.Exited)

plt.subplot(2,2,3)

sns.boxplot(x= df.Geography, y= df.Balance, hue=df.Exited)

plt.subplot(2,2,4)

sns.boxplot(x= df.Geography, y= df.Balance, hue=df.Gender)
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.boxplot(x= df.Exited, y= df.Tenure, hue=df.Exited)

plt.subplot(2,2,2)

sns.boxplot(x= df.Gender, y= df.Tenure, hue=df.Exited)

plt.subplot(2,2,3)

sns.boxplot(x= df.Geography, y= df.Tenure, hue=df.Exited)

plt.subplot(2,2,4)

sns.boxplot(x= df.Geography, y= df.Tenure, hue=df.Gender)
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.boxplot(x= df.Exited, y= df.NumOfProducts, hue=df.Exited)

plt.subplot(2,2,2)

sns.boxplot(x= df.Gender, y= df.NumOfProducts, hue=df.Exited)

plt.subplot(2,2,3)

sns.boxplot(x= df.Geography, y= df.NumOfProducts, hue=df.Exited)

plt.subplot(2,2,4)

sns.boxplot(x= df.Geography, y= df.NumOfProducts, hue=df.Gender)
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.boxplot(x= df.Exited, y= df.EstimatedSalary, hue=df.Exited)

plt.subplot(2,2,2)

sns.boxplot(x= df.Gender, y= df.EstimatedSalary, hue=df.Exited)

plt.subplot(2,2,3)

sns.boxplot(x= df.Geography, y= df.EstimatedSalary, hue=df.Exited)

plt.subplot(2,2,4)

sns.boxplot(x= df.Geography, y= df.EstimatedSalary, hue=df.Gender)
df.head()
plt.figure(figsize=(14,12))

plt.subplot(2,2,1)

sns.regplot(x=df.Age, y=df.CreditScore)

plt.subplot(2,2,2)

sns.regplot(x=df.Age, y=df.Balance)

plt.subplot(2,2,3)

sns.regplot(x=df.Age, y=df.EstimatedSalary)

plt.subplot(2,2,4)

sns.boxplot(y=df.Age, x=df.HasCrCard, hue=df.HasCrCard)
df[df.Balance == 0].Exited.value_counts()