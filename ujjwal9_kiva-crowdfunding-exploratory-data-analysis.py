import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
# Any results you write to the current directory are saved as output.
loans = pd.read_csv('../input/kiva_loans.csv')
loans['date'] = pd.to_datetime(loans['date'])
loans['year'] = pd.DatetimeIndex(loans['date']).year
loans['month'] = pd.DatetimeIndex(loans['date']).month
loans.head()
loans.columns
gender_list = []
for gender in loans["borrower_genders"].values:
    if str(gender) != "nan":
        gender_list.extend( [lst.strip() for lst in gender.split(",")] )
temp_data = pd.Series(gender_list).value_counts()

labels = (np.array(temp_data.index))
sizes = (np.array((temp_data / temp_data.sum())*100))
plt.figure(figsize=(15,8))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(title='Borrower Gender')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")
plt.figure(figsize=(15,8))
count = loans['activity'].value_counts().head(30)
sns.barplot(count.values, count.index)
for i, v in enumerate(count.values):
    plt.text(0.8,i,v,color='k',fontsize=12)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Activity name?', fontsize=12)
plt.title("Top Loan Activity type", fontsize=16)
plt.figure(figsize=(8,6))
plt.scatter(range(loans.shape[0]), np.sort(loans.loan_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(range(loans.shape[0]), np.sort(loans.funded_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Funded Amount Distribution")
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
sec_loans = loans.groupby('sector')['loan_amount'].sum()
plt.figure(figsize=(20,8))
threshold = np.mean(sec_loans.values)
values = sec_loans.values

# split it up
above_threshold = np.maximum(values - threshold, 0)
below_threshold = np.minimum(values, threshold)

# and plot it
plt.bar(sec_loans.index, below_threshold, 0.35, color="g")
plt.bar(sec_loans.index, above_threshold, 0.35, color="r",
        bottom=below_threshold)
plt.xticks(sec_loans.index, sec_loans.index, rotation='vertical')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=10)
# horizontal line indicating the threshold
mean_amount = plt.plot([sec_loans.index[0], sec_loans.index[-1]], [threshold, threshold], "k--", label='Mean Loan Amount')
plt.legend(prop={'size': 10})
plt.xlabel('Sectors')
plt.ylabel('Loan Amount in (Hundered Million)')
plt.title('Sector wise loans')
plt.legend()
plt.show()
country_loans = loans.groupby('country')['loan_amount'].sum()
plt.figure(figsize=(50,15))
threshold = np.mean(country_loans.values)
values = country_loans.values

# split it up
above_threshold = np.maximum(values - threshold, 0)
below_threshold = np.minimum(values, threshold)

# and plot it
plt.bar(country_loans.index, below_threshold, 0.35, color="g")
plt.bar(country_loans.index, above_threshold, 0.35, color="r",
        bottom=below_threshold)
plt.xticks(country_loans.index, country_loans.index, rotation='vertical')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=10)
# horizontal line indicating the threshold
plt.plot([country_loans.index[0], country_loans.index[-1]], [threshold, threshold], "k--", label='Mean Loan Amount')
plt.legend(prop={'size': 24})
plt.xlabel('Countries', fontsize=18)
plt.ylabel('Loan Amount in (10 Millions)', fontsize=18)
plt.title('country wise loans', fontsize=28)
# plt.bar(country_loans.index, country_loans.values)
# plt.plot(np.mean(country_loans.values))
plt.show()
c_top4_loans = loans.groupby(['country','sector'])['loan_amount'].sum()
c_top4_loans = c_top4_loans[['Philippines', 'Kenya', 'United States', 'Peru']]
plt.figure(figsize=(20,8))
sns.barplot(c_top4_loans['Philippines'].index, c_top4_loans['Philippines'].values)
plt.xlabel('Sectors')
plt.ylabel('Loan Amount in (10 Millions)')
plt.title('Philippines',fontsize=20)
plt.show()
plt.figure(figsize=(20,8))
sns.barplot(c_top4_loans['Kenya'].index, c_top4_loans['Kenya'].values)
plt.xlabel('Sectors')
plt.ylabel('Loan Amount in (10 Millions)')
plt.title('Kenya',fontsize=20)
plt.show()
plt.figure(figsize=(20,8))
sns.barplot(c_top4_loans['United States'].index, c_top4_loans['United States'].values)
plt.xlabel('Sectors')
plt.ylabel('Loan Amount in (10 Millions)')
plt.title('United States', fontsize=20)
plt.show()
plt.figure(figsize=(20,8))
sns.barplot(c_top4_loans['Peru'].index, c_top4_loans['Peru'].values)
plt.xlabel('Sectors')
plt.ylabel('Loan Amount in (10 Millions)')
plt.title('Peru', fontsize=20)
plt.show()
c_top4_loans = loans.loc[loans['country'].isin(['Philippines', 'Kenya', 'United States', 'Peru'])]
sns.factorplot(x="sector", y="loan_amount", hue="country", data=c_top4_loans, kind="bar", size=18)
plt.title('Top 4 Countries', fontsize=15)
ind_loans = loans[loans['country']=='India']

sec_ind_loans_sum = ind_loans.groupby('sector')['loan_amount'].sum()
sec_ind_loans_count = ind_loans.groupby('sector')['loan_amount'].count()
plt.figure(figsize=(20,8))
sns.barplot(sec_ind_loans_sum.index, sec_ind_loans_sum.values)
plt.xlabel('Sectors')
plt.ylabel('Loan Amount')
plt.title('India Sector wise loan amounts', fontsize=20)
plt.show()
plt.figure(figsize=(20,8))
sns.barplot(sec_ind_loans_count.index, sec_ind_loans_count.values)
plt.xlabel('Sectors')
plt.ylabel('Loan Counts')
plt.title('India Sector wise loan counts', fontsize=20)
plt.show()
agri_ind_loans = ind_loans[ind_loans['sector']=='Agriculture']
agri_ind_loans.drop(['sector', 'country', 'currency'], axis = 1, inplace = True)
agri_ind_loans_activity_sum = agri_ind_loans.groupby('activity')['loan_amount'].sum()
agri_ind_loans_activity_count = agri_ind_loans.groupby('activity')['loan_amount'].count()
plt.figure(figsize=(20,8))
sns.barplot(agri_ind_loans_activity_sum.index, agri_ind_loans_activity_sum.values)
plt.xlabel('Activity')
plt.ylabel('loan amount')
plt.title('Agricultural Activity Analysis - SUM of Loan Amount')
plt.show()

plt.figure(figsize=(20,8))
sns.barplot(agri_ind_loans_activity_count.index, agri_ind_loans_activity_count.values)
plt.xlabel('Activity')
plt.ylabel('loan count')
plt.title('Agricultural Activity Analysis - Count of Loans')
plt.show()
loan_themes_region = pd.read_csv('../input/loan_themes_by_region.csv')
ind_loan_theme_region = loan_themes_region[loan_themes_region['country']=='India']
loan_theme_id = pd.read_csv('../input/loan_theme_ids.csv')
loans = pd.merge(loans,loan_theme_id, on='id')
ind_monthly_agri_loans_sum = agri_ind_loans.groupby(['year','month'])['loan_amount'].sum()
ind_monthly_agri_loans_count = agri_ind_loans.groupby(['year','month'])['loan_amount'].count()
ind_monthly_agri_loans_sum = ind_monthly_agri_loans_sum.to_frame()
ind_monthly_agri_loans_count = ind_monthly_agri_loans_count.to_frame()
ind_monthly_agri_loans_sum.unstack(level=0).plot(kind='bar', figsize=(20,8))
plt.xlabel('Month')
plt.ylabel('Loan_Amount_Sum')
plt.show()
ind_monthly_agri_loans_count.unstack(level=0).plot(kind='bar', figsize=(20,8))
plt.xlabel('Month')
plt.ylabel('Loan_count')
plt.show()


