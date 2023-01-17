import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ld = pd.read_csv('../input/Loan payments data.csv')
ld.head()
ld.info()
ld['effective_date'] = pd.to_datetime(ld['effective_date'])
ld['due_date'] = pd.to_datetime(ld['due_date'])
ld['paid_off_time'] = pd.to_datetime(ld['paid_off_time'])
ld.info()
ld.describe()
ld.hist(bins=30, figsize=(20,15))
ld.groupby('loan_status').count()
fig, axs = plt.subplots(2, 2, figsize=(20,15))

sns.countplot(x = "loan_status",hue = "Principal", data=ld,ax=axs[0][0])
axs[0][0].set_title("Distribution of Principal Loan Amount by Loan Status")

sns.countplot(x = "loan_status",hue = "terms", data=ld,ax=axs[0][1])
axs[0][1].set_title("Distribution of Loan Payoff Terms by Loan Status")

sns.countplot(x = "loan_status",hue = "Gender", data=ld,ax=axs[1][0])
axs[1][0].set_title("Distribution of Gender by Loan Status")

sns.countplot(x = "loan_status",hue = "education", data=ld, ax=axs[1][1])
axs[1][1].set_title("Distribution of Education by Loan Status")
plt.figure(figsize=(12,8))
plt.title('Distribution of Past Due Days by Loan Status')
sns.countplot(x='past_due_days',hue='loan_status', data=ld)
plt.figure(figsize=(12,8))
plt.title('Distribution of Age by Loan Status')
sns.countplot(x='age',hue='loan_status', data=ld)
plt.figure(figsize=(20,8))
plt.title('Distribution of Effective Date by Loan Status')
plt.xticks(rotation=45)
sns.countplot(x='effective_date',hue='loan_status', data=ld)
plt.figure(figsize=(20,8))
plt.title('Distribution of Due Date by Loan Status')
plt.xticks(rotation=45)
sns.countplot(x='due_date',hue='loan_status', data=ld)