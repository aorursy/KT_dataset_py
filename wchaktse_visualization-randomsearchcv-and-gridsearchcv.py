import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import sklearn as sk
from sklearn import preprocessing
import eli5

!pip install pycountry_convert
import pycountry_convert as pc

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 100)

%matplotlib inline 
raw = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
df = raw.copy()
df.info()
df[df.duplicated()].head(100)
corr = df.corr().apply(round, ndigits=2)
fig0, ax0 = plt.subplots(figsize=(16,12))

ax0 = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)
ax0.set_xticklabels(
    ax0.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
df.drop(columns=['arrival_date_year', 'arrival_date_day_of_month',
                 'company', 'reservation_status', 'reservation_status_date'],
        axis=0, inplace=True)
df['hotel'].value_counts()
pd.crosstab(df['hotel'], df['is_canceled'])
pd.crosstab(df['hotel'], df['is_canceled'], normalize='index')
sns.boxplot(data=df, x='is_canceled', y='lead_time')
# Dividing arrival month by "is_canceled"
# c = canceled; n = not canceled
month_c = pd.DataFrame(df[df['is_canceled'] == 1]['arrival_date_month'].value_counts())
month_n = pd.DataFrame(df[df['is_canceled'] == 0]['arrival_date_month'].value_counts())

month_sorter = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']

month_c = month_c.loc[month_sorter]
month_n = month_n.loc[month_sorter]


# Dividing arrival week number by "is_canceled"
weeknumber_c = pd.DataFrame(df[df['is_canceled'] == 1]['arrival_date_week_number'].value_counts())
weeknumber_n = pd.DataFrame(df[df['is_canceled'] == 0]['arrival_date_week_number'].value_counts())

weeknumber_c.sort_index(inplace=True)
weeknumber_n.sort_index(inplace=True)
# Stacked Barchart

fig1, ax1 = plt.subplots(2, figsize=(16,6))
ax1[0].bar(weeknumber_c.index, weeknumber_c['arrival_date_week_number'], width=0.5, label='Canceled', color='#d63031')
ax1[0].bar(weeknumber_n.index, weeknumber_c['arrival_date_week_number'], width=0.5,
       bottom=weeknumber_c['arrival_date_week_number'], label='Uncanceled', color='#00b894')

ax1[0].set_ylabel('Frequency')
ax1[0].set_title('Arrival Week Number')
ax1[0].legend()


ax1[1].bar(month_c.index, month_c['arrival_date_month'], width=0.5, label='Canceled', color='#d63031')
ax1[1].bar(month_n.index, month_n['arrival_date_month'], width=0.5,
       bottom=month_c['arrival_date_month'], label='Uncanceled', color='#00b894')

ax1[1].set_ylabel('Frequency')
ax1[1].set_title('Arrival Month')
ax1[1].legend()

plt.show()
# Converts the frequency data in month into proportions
month_c['total'] = month_n['total'] = pd.DataFrame([(month_c.loc[month] + month_n.loc[month]) for month in month_sorter])
month_c['proportion'] = month_c['arrival_date_month'] / month_c['total']
month_n['proportion'] = month_n['arrival_date_month'] / month_n['total']

# Converts the freqency data  in week number into proportions
weeknumber_c['total'] = weeknumber_n['total'] = pd.DataFrame([(weeknumber_c.loc[weeknumber] + weeknumber_n.loc[weeknumber]) for weeknumber in range(1, 54)])
weeknumber_c['proportion'] = weeknumber_c['arrival_date_week_number'] / weeknumber_c['total']
weeknumber_n['proportion'] = weeknumber_n['arrival_date_week_number'] / weeknumber_n['total']
fig2, ax2 = plt.subplots(2, figsize=(16,6))
ax2[0].bar(weeknumber_c.index, weeknumber_c['proportion'], width=0.5, label='Canceled', color='#d63031')
ax2[0].bar(weeknumber_n.index, weeknumber_n['proportion'], width=0.5,
        bottom=weeknumber_c['proportion'], label='Uncanceled', color='#00b894')

ax2[0].set_ylim(0,1)
ax2[0].set_ylabel('Proportion')
ax2[0].set_title('Arrival Week Number')
ax2[0].legend()


ax2[1].bar(month_c.index, month_c['proportion'], width=0.5, label='Canceled', color='#d63031')
ax2[1].bar(month_n.index, month_n['proportion'], width=0.5,
       bottom=month_c['proportion'], label='Uncanceled', color='#00b894')

ax2[1].set_ylim(0,1)
ax2[1].set_ylabel('Proportion')
ax2[1].set_title('Arrival Month')
ax2[1].legend()

plt.show()
df.drop(columns=['arrival_date_month'], axis=0, inplace=True)
fig3, ax3 = plt.subplots(1, 2, figsize=(12,4))

sns.boxplot(data=df, x='is_canceled', y='stays_in_weekend_nights', ax=ax3[0])
ax3[0].set_title('Weekend Stay Length vs Cancel Status')

sns.boxplot(data=df, x='is_canceled', y='stays_in_week_nights', ax=ax3[1])
ax3[1].set_title('Week Stay Length vs Cancel Status')
df[df['stays_in_week_nights'] > 25][['stays_in_week_nights', 'stays_in_weekend_nights']]
fig8, ax8 = plt.subplots(1,3, figsize=(16,4))

sns.distplot(df['adults'], ax=ax8[0], kde=False)
ax8[0].set_title('Distribution of adults')

sns.distplot(df['children'], ax=ax8[1], color='orange', kde=False)
ax8[1].set_title('Distribution of children')

sns.distplot(df['babies'], ax=ax8[2], color='green', kde=False)
ax8[2].set_title('Distribution of babies')
fig4, ax4 = plt.subplots(1,3, figsize=(16,4))

sns.boxplot(data=df, x='is_canceled', y='adults', ax=ax4[0])
ax4[0].set_title('Adults vs Cancel Status')

sns.boxplot(data=df, x='is_canceled', y='children', ax=ax4[1])
ax4[1].set_title('Children vs Cancel Status')

sns.boxplot(data=df, x='is_canceled', y='babies', ax=ax4[2])
ax4[2].set_title('Babies vs Cancel Status')
df[(df['adults'] > 20) | (df['children'] > 5) | (df['babies'] > 5)]
df[df['children'].isna()]
df['children'].fillna(df['children'].median(), inplace = True)
df['meal'].value_counts()
len(df['country'].unique())
country_c = pd.DataFrame(df[df['is_canceled'] == 1]['country'].value_counts())
country_n = pd.DataFrame(df[df['is_canceled'] == 0]['country'].value_counts())

total_c = country_c['country'].sum()
total_n = country_n['country'].sum()

country_c['percentage'] = round(country_c["country"] / total_c * 100, 2)
country_n['percentage'] = round(country_n["country"] / total_n * 100, 2)
fig5 = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]]) # plotly function

fig5.add_trace(go.Pie(values=country_c['country'],
                      labels=country_c.index,
                      title="Home country of canceled guests",
                      textposition="inside",
                      textinfo="value+percent+label"),
               row=1, col=1)

fig5.add_trace(go.Pie(values=country_n['country'],
                      labels=country_n.index,
                      title="Home country of non-canceled guests",
                      textposition="inside",
                      textinfo="value+percent+label"),
               row=1, col=2)
df.dropna(subset=['country'], inplace=True)
# Fix problematic values
df['country'] = df['country'].apply(lambda country: 'CHN' if country == 'CN' else country)
df['country'] = df['country'].apply(lambda country: 'TLS' if country == 'TMP' else country)

# pycountry does not provide a method directly transforming alpha3 country code to continents
# Needs to convert into alpha2 country code first
df['continent'] = df['country'].apply(pc.country_alpha3_to_country_alpha2)

# Edit and credit to pycountry-convert's method to transform alpha2 country code to continents
# Copying and editing due to the missing 'TL' country in the method
COUNTRY_ALPHA2_TO_CONTINENT = {
    'AB': 'Asia',
    'AD': 'Europe',
    'AE': 'Asia',
    'AF': 'Asia',
    'AG': 'North America',
    'AI': 'North America',
    'AL': 'Europe',
    'AM': 'Asia',
    'AO': 'Africa',
    'AR': 'South America',
    'AS': 'Oceania',
    'AT': 'Europe',
    'AU': 'Oceania',
    'AW': 'North America',
    'AX': 'Europe',
    'AZ': 'Asia',
    'BA': 'Europe',
    'BB': 'North America',
    'BD': 'Asia',
    'BE': 'Europe',
    'BF': 'Africa',
    'BG': 'Europe',
    'BH': 'Asia',
    'BI': 'Africa',
    'BJ': 'Africa',
    'BL': 'North America',
    'BM': 'North America',
    'BN': 'Asia',
    'BO': 'South America',
    'BQ': 'North America',
    'BR': 'South America',
    'BS': 'North America',
    'BT': 'Asia',
    'BV': 'Antarctica',
    'BW': 'Africa',
    'BY': 'Europe',
    'BZ': 'North America',
    'CA': 'North America',
    'CC': 'Asia',
    'CD': 'Africa',
    'CF': 'Africa',
    'CG': 'Africa',
    'CH': 'Europe',
    'CI': 'Africa',
    'CK': 'Oceania',
    'CL': 'South America',
    'CM': 'Africa',
    'CN': 'Asia',
    'CO': 'South America',
    'CR': 'North America',
    'CU': 'North America',
    'CV': 'Africa',
    'CW': 'North America',
    'CX': 'Asia',
    'CY': 'Asia',
    'CZ': 'Europe',
    'DE': 'Europe',
    'DJ': 'Africa',
    'DK': 'Europe',
    'DM': 'North America',
    'DO': 'North America',
    'DZ': 'Africa',
    'EC': 'South America',
    'EE': 'Europe',
    'EG': 'Africa',
    'ER': 'Africa',
    'ES': 'Europe',
    'ET': 'Africa',
    'FI': 'Europe',
    'FJ': 'Oceania',
    'FK': 'South America',
    'FM': 'Oceania',
    'FO': 'Europe',
    'FR': 'Europe',
    'GA': 'Africa',
    'GB': 'Europe',
    'GD': 'North America',
    'GE': 'Asia',
    'GF': 'South America',
    'GG': 'Europe',
    'GH': 'Africa',
    'GI': 'Europe',
    'GL': 'North America',
    'GM': 'Africa',
    'GN': 'Africa',
    'GP': 'North America',
    'GQ': 'Africa',
    'GR': 'Europe',
    'GS': 'South America',
    'GT': 'North America',
    'GU': 'Oceania',
    'GW': 'Africa',
    'GY': 'South America',
    'HK': 'Asia',
    'HM': 'Antarctica',
    'HN': 'North America',
    'HR': 'Europe',
    'HT': 'North America',
    'HU': 'Europe',
    'ID': 'Asia',
    'IE': 'Europe',
    'IL': 'Asia',
    'IM': 'Europe',
    'IN': 'Asia',
    'IO': 'Asia',
    'IQ': 'Asia',
    'IR': 'Asia',
    'IS': 'Europe',
    'IT': 'Europe',
    'JE': 'Europe',
    'JM': 'North America',
    'JO': 'Asia',
    'JP': 'Asia',
    'KE': 'Africa',
    'KG': 'Asia',
    'KH': 'Asia',
    'KI': 'Oceania',
    'KM': 'Africa',
    'KN': 'North America',
    'KP': 'Asia',
    'KR': 'Asia',
    'KW': 'Asia',
    'KY': 'North America',
    'KZ': 'Asia',
    'LA': 'Asia',
    'LB': 'Asia',
    'LC': 'North America',
    'LI': 'Europe',
    'LK': 'Asia',
    'LR': 'Africa',
    'LS': 'Africa',
    'LT': 'Europe',
    'LU': 'Europe',
    'LV': 'Europe',
    'LY': 'Africa',
    'MA': 'Africa',
    'MC': 'Europe',
    'MD': 'Europe',
    'ME': 'Europe',
    'MF': 'North America',
    'MG': 'Africa',
    'MH': 'Oceania',
    'MK': 'Europe',
    'ML': 'Africa',
    'MM': 'Asia',
    'MN': 'Asia',
    'MO': 'Asia',
    'MP': 'Oceania',
    'MQ': 'North America',
    'MR': 'Africa',
    'MS': 'North America',
    'MT': 'Europe',
    'MU': 'Africa',
    'MV': 'Asia',
    'MW': 'Africa',
    'MX': 'North America',
    'MY': 'Asia',
    'MZ': 'Africa',
    'NA': 'Africa',
    'NC': 'Oceania',
    'NE': 'Africa',
    'NF': 'Oceania',
    'NG': 'Africa',
    'NI': 'North America',
    'NL': 'Europe',
    'NO': 'Europe',
    'NP': 'Asia',
    'NR': 'Oceania',
    'NU': 'Oceania',
    'NZ': 'Oceania',
    'OM': 'Asia',
    'OS': 'Asia',
    'PA': 'North America',
    'PE': 'South America',
    'PF': 'Oceania',
    'PG': 'Oceania',
    'PH': 'Asia',
    'PK': 'Asia',
    'PL': 'Europe',
    'PM': 'North America',
    'PR': 'North America',
    'PS': 'Asia',
    'PT': 'Europe',
    'PW': 'Oceania',
    'PY': 'South America',
    'QA': 'Asia',
    'RE': 'Africa',
    'RO': 'Europe',
    'RS': 'Europe',
    'RU': 'Europe',
    'RW': 'Africa',
    'SA': 'Asia',
    'SB': 'Oceania',
    'SC': 'Africa',
    'SD': 'Africa',
    'SE': 'Europe',
    'SG': 'Asia',
    'SH': 'Africa',
    'SI': 'Europe',
    'SJ': 'Europe',
    'SK': 'Europe',
    'SL': 'Africa',
    'SM': 'Europe',
    'SN': 'Africa',
    'SO': 'Africa',
    'SR': 'South America',
    'SS': 'Africa',
    'ST': 'Africa',
    'SV': 'North America',
    'SY': 'Asia',
    'SZ': 'Africa',
    'TC': 'North America',
    'TD': 'Africa',
    'TG': 'Africa',
    'TH': 'Asia',
    'TJ': 'Asia',
    'TK': 'Oceania',
    'TL': 'Asia',
    'TM': 'Asia',
    'TN': 'Africa',
    'TO': 'Oceania',
    'TP': 'Asia',
    'TR': 'Asia',
    'TT': 'North America',
    'TV': 'Oceania',
    'TW': 'Asia',
    'TZ': 'Africa',
    'UA': 'Europe',
    'UG': 'Africa',
    'US': 'North America',
    'UY': 'South America',
    'UZ': 'Asia',
    'VC': 'North America',
    'VE': 'South America',
    'VG': 'North America',
    'VI': 'North America',
    'VN': 'Asia',
    'VU': 'Oceania',
    'WF': 'Oceania',
    'WS': 'Oceania',
    'XK': 'Europe',
    'YE': 'Asia',
    'YT': 'Africa',
    'ZA': 'Africa',
    'ZM': 'Africa',
    'ZW': 'Africa',
}

df['continent'] = df['continent'].map(COUNTRY_ALPHA2_TO_CONTINENT)
df[df['continent'].isna()]['country']
df.at[68227, 'continent'] = 'North America'
df.at[99598, 'continent'] = 'Antartica'
df.at[104762, 'continent'] = 'Antartica'
df.at[115334, 'continent'] = 'Antartica'
df.drop(columns=['country'], axis=0, inplace=True)
print(df['continent'].value_counts())
df['continent'].hist(figsize=(16,6))
df[df['continent'].isna()]
df['market_segment'].value_counts()
df['distribution_channel'].value_counts()
df['is_repeated_guest'].value_counts()
print(df['previous_cancellations'].describe())
sns.boxplot(data=df, x='is_canceled', y='previous_cancellations')
print(df['previous_bookings_not_canceled'].describe())
sns.boxplot(data=df, x='is_canceled', y='previous_bookings_not_canceled')
fig6, ax6 = plt.subplots(1, 2, figsize=(16,6))

sns.countplot(data=df, x='reserved_room_type', hue='is_canceled', ax=ax6[0])
ax6[0].set_title('Reserved Room Type vs Cancellation Status')
ax6[0].set_ylim(0, 55000)

sns.countplot(data=df, x='assigned_room_type', hue='is_canceled', ax=ax6[1])
ax6[1].set_title('Assigned Room Type vs Cancellation Status')
ax6[1].set_ylim(0, 55000)
df['reserved_equals_assigned'] = np.where(df['reserved_room_type'] == df['assigned_room_type'], 1, 0)
print('For canceled:')
print(df[df['is_canceled'] == 1]['reserved_equals_assigned'].value_counts())
print('\nFor non-canceled:')
print(df[df['is_canceled'] == 0]['reserved_equals_assigned'].value_counts())
pd.crosstab(df['is_canceled'], df['reserved_equals_assigned'])
sns.boxplot(data=df, x='is_canceled', y='booking_changes')
df['deposit_type'].value_counts()
df['agent'].value_counts()
df['agent'].fillna(0, inplace=True)
df.loc[df['agent'] != 0, 'agent'] = 1
df.rename({'agent': 'has_agent'}, axis=1, inplace=True)
df['has_agent'].value_counts()
print(df['days_in_waiting_list'].describe())
sns.boxplot(data=df, x='is_canceled', y='days_in_waiting_list')
print(df['customer_type'].value_counts())
df['customer_type'].hist()
print(pd.crosstab(df['is_canceled'], df['customer_type']))
pd.crosstab(df['is_canceled'], df['customer_type']).plot(kind='pie', subplots=True, figsize=(16,4),
                                                         title='Customer Types vs Cancellation Status')
df['adr'].describe()
adr = df['adr'].apply(lambda num: None if num==0 else np.log(num))
fig7, ax7 = plt.subplots(figsize=(6,3))
ax7 = sns.distplot(adr.dropna(), bins=30)
ax7.set_title('Distribution of log(adr)')
ax7.set_xlim((1,7))
ax7.set_xlabel('log(adr)')
ax7.set_ylabel('Frequency')
sns.boxplot(data=df, x='is_canceled', y='adr')
df[df['adr'] > 5000]
sns.boxplot(data=df[df['adr'] < 5000], x='is_canceled', y='adr')
df['required_car_parking_spaces'].value_counts()
df.loc[df['required_car_parking_spaces'] == 8]
sns.boxplot(data=df, x='is_canceled', y='required_car_parking_spaces')
print(df['total_of_special_requests'].value_counts())
df['total_of_special_requests'].hist()
sns.boxplot(data=df, x='is_canceled', y='total_of_special_requests')
fig8, ax8 = plt.subplots(figsize=(12,6))

# Only plot data where adr < 5000 to prevent the single outlier
sns.lineplot(data=df[df['adr'] < 5000],
             x='arrival_date_week_number', y='adr', hue='hotel',
             ci='sd', size='hotel', sizes=(2,2), ax=ax8)

ax8.set_title('Average Daily Price on Each Week of Year by Hotel')
ax8.set_xlim(0,54)

ax8.set_xlabel('Week Number of Year')
ax8.set_ylabel('Average Daily Price in EUR')

plt.grid()
plt.show()
fig9, ax9 = plt.subplots(figsize=(12,6))
sns.lineplot(data=df,
             x='arrival_date_week_number', y='stays_in_week_nights', hue='hotel',
             ci='sd', size='hotel', sizes=(2,2), ax=ax9)

ax9.set_title('Average Length of Stay on Each Week of Year by Hotel')
ax9.set_xlim(0,54)

ax9.set_xlabel('Week Number of Year')
ax9.set_ylabel('Average Length of Stay')

plt.grid()
plt.show()
fig10, ax10 = plt.subplots(figsize=(12,6))
sns.lineplot(data=df[df['continent'] != 'Antartica'],
             x='arrival_date_week_number', y='adr', hue='continent',
             ci=None, size='continent', sizes=(2,2), ax=ax10)

ax10.set_title('Average Daily Price on Each Week of Year by Customer Origin')
ax10.set_xlim(0,54)
ax10.set_ylim(25,200)

ax10.set_xlabel('Week Number of Year')
ax10.set_ylabel('Average Daily Price')

plt.grid()
plt.show()
df.info()
ohe_columns = ['hotel', 'meal', 'market_segment', 'arrival_date_week_number', 'distribution_channel',
               'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'continent']

df = pd.get_dummies(df, columns=ohe_columns, drop_first=True)
df.head()
# Numerical features to be normalized
normalize_columns = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                     'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
                     'adr', 'required_car_parking_spaces', 'total_of_special_requests']
df_minmax = df.copy()
minmax_scaler = preprocessing.MinMaxScaler()

for col in normalize_columns:
  scaler = minmax_scaler.fit(df_minmax[[col]])
  df_minmax[col] = scaler.transform(df_minmax[[col]])
df_minmax.head()
df_zscore = df.copy()
zscore_scaler = preprocessing.StandardScaler()

for col in normalize_columns:
  scaler = zscore_scaler.fit(df_zscore[[col]])
  df_zscore[col] = scaler.transform(df_zscore[[col]])
df_zscore.head()
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
X = df.drop(columns='is_canceled', inplace=False)
minmax_X = df_minmax.drop(columns='is_canceled', inplace=False)
zscore_X = df_zscore.drop(columns='is_canceled', inplace=False)
y = minmax_y = zscore_y = df['is_canceled']
models = [('Decision Tree', DecisionTreeClassifier()),
          ('Random Forest', RandomForestClassifier(n_jobs=-1)),
          ('Logistic Regression', LogisticRegression(n_jobs=-1)),
          ('Gaussian Naive Bayes', GaussianNB()),
          ('Multinomial Naive Bayes', MultinomialNB()),
          ('Bernoulli Naive Bayes', BernoulliNB()),
          ('Ridge Regression', RidgeClassifier()),
          ('AdaBoost', AdaBoostClassifier())]
kfolds = 10

for name, model in models:
    print('\nNow running:', name, 'on original data...')
    try:
        orig_cv = cross_val_score(model, X, y, cv=kfolds, scoring='accuracy', n_jobs=-1)
        
        orig_min_score = round(min(orig_cv), 4)
        orig_max_score = round(max(orig_cv), 4)
        orig_mean_score = round(np.mean(orig_cv), 4)
        orig_std_dev = round(np.std(orig_cv), 4)
        
        print('Scores:', orig_cv)
        print('Minimum score of', name, 'in original data =', orig_min_score)
        print('Maximum score of', name, 'in original data =', orig_max_score)
        print('Mean score of', name, 'in original data =', orig_mean_score)
        print('SD of score of', name, 'in original data =', orig_std_dev)
        
    except KeyboardInterrupt:
        raise
    except:
        print(name, 'cannot be ran on original data.')
    
    print('\nNow running:', name, 'on scale to range data...')
    try:
        minmax_cv = cross_val_score(model, minmax_X, minmax_y, cv=kfolds, scoring='accuracy', n_jobs=-1)
        
        print('Scores:', minmax_cv)
        minmax_min_score = round(min(minmax_cv), 4)
        minmax_max_score = round(max(minmax_cv), 4)
        minmax_mean_score = round(np.mean(minmax_cv), 4)
        minmax_std_dev = round(np.std(minmax_cv), 4)
        
        print('Minimum score of', name, 'in scale to range data =', minmax_min_score)
        print('Maximum score of', name, 'in scale to range data =', minmax_max_score)
        print('Mean score of', name, 'in scale to range data =', minmax_mean_score)
        print('SD of score of', name, 'in scale to range data =', minmax_std_dev)
        
    except KeyboardInterrupt:
        raise
    except:
        print(name, 'cannot be ran on scale to range data.')
        
    print('\nNow Running:', name, 'on scale to zscore data...')
    try:
        zscore_cv = cross_val_score(model, zscore_X, zscore_y, cv=kfolds, scoring='accuracy', n_jobs=-1)
        
        zscore_min_score = round(min(zscore_cv), 4)
        zscore_max_score = round(max(zscore_cv), 4)
        zscore_mean_score = round(np.mean(zscore_cv), 4)
        zscore_std_dev = round(np.std(zscore_cv), 4)

        print('Scores:', zscore_cv)
        print('Minimum score of', name, 'in scale to zscore data =', zscore_min_score)
        print('Maximum score of', name, 'in scale to zscore data =', zscore_max_score)
        print('Mean score of', name, 'in scale to zscore data =', zscore_mean_score)
        print('SD of score of', name, 'in scale to zscore data =', zscore_std_dev)
        
    except KeyboardInterrupt:
        raise
    except:
        print(name, 'cannot be ran on zscore data.')
from sklearn.tree import DecisionTreeClassifier

dt_parameters = {
    'max_depth': np.arange(4,10),
    'max_leaf_nodes': np.arange(10,30),
    'min_samples_leaf': np.arange(10,30)
}

DT = GridSearchCV(DecisionTreeClassifier(), dt_parameters, cv=5, n_jobs=-1)
DT.fit(X, y)
print(DT.best_params_)
print(DT.best_score_)
dt = DecisionTreeClassifier(max_depth=7,
                             max_leaf_nodes=20,
                             min_samples_leaf=21,
                             random_state=0)

dt_pred = cross_val_predict(dt, X, y, cv=10, n_jobs=-1, method='predict')
dt_pred_proba = cross_val_predict(dt, X, y, cv=10, n_jobs=-1, method='predict_proba')
dt_proba_y_1 = dt_pred_proba[:,1]
lr_parameters = {
    'penalty': ['l1','l2'],
    'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
}

LR = GridSearchCV(LogisticRegression(solver='saga', max_iter=10000), lr_parameters, cv=5)
LR.fit(minmax_X, minmax_y)
print(LR.best_params_)
print(LR.best_score_)
lr = LogisticRegression(C=0.001,
                        penalty='l1',
                        solver='saga',
                        max_iter=10000)

lr_pred = cross_val_predict(lr, minmax_X, minmax_y, cv=10, n_jobs=-1, method='predict')
lr_pred_proba = cross_val_predict(lr, minmax_X, minmax_y, cv=10, n_jobs=-1, method='predict_proba')
lr_proba_y_1 = lr_pred_proba[:,1]
rr_parameters = {
    'alpha': np.arange(1, 101, 5),
    'class_weight': [None, 'balanced']
}

RR = GridSearchCV(RidgeClassifier(), rr_parameters, cv=5)
RR.fit(minmax_X, minmax_y)
print(RR.best_params_)
print(RR.best_score_)
rr = RidgeClassifier(alpha=81)

rr_pred = cross_val_predict(rr, minmax_X, minmax_y, cv=10, n_jobs=-1, method='predict')
# rr_pred_proba = cross_val_predict(rr, minmax_X, minmax_y, cv=10, n_jobs=-1, method='predict_proba')
# rr_proba_y_1 = lr_pred_proba[:,1]
abc = AdaBoostClassifier()
abc_pred = cross_val_predict(abc, X, y, cv=10, n_jobs=-1, method='predict')
abc_pred_proba = cross_val_predict(abc, X, y, cv=10, n_jobs=-1, method='predict_proba')
abc_proba_y_1 = abc_pred_proba[:,1]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
def evaluation(name, y, y_pred):
    print(f'Accuracy of {name}:', accuracy_score(y, y_pred, normalize=True))
    print(f'Classification report of {name}:')
    print(classification_report(y, y_pred))
    print(f'Confusion matrix of {name}:')
    print(confusion_matrix(y, y_pred))
    print('')
    
evaluation('Decision Tree', y, dt_pred)
evaluation('Logistic Regression', y, lr_pred)
evaluation('Ridge Regression', y, rr_pred)
evaluation('Ada Boost', y, abc_pred)
fig, ax = plt.subplots(figsize=(8,8))
ax.set_title('ROC of Different Models')
ax.set_xlim(-0.05,1.05)
ax.set_ylim(-0.05,1.05)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y, dt_proba_y_1)
plt.plot(fpr_dt, tpr_dt, label="Decision Tree's AUC: " + str(round(auc(fpr_dt, tpr_dt), 3)))

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y, lr_proba_y_1)
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression's AUC: " + str(round(auc(fpr_lr, tpr_lr), 3)))

fpr_abc, tpr_abc, thresholds_abc = roc_curve(y, abc_proba_y_1)
plt.plot(fpr_abc, tpr_abc, label="Default Ada Boost's AUC: " + str(round(auc(fpr_abc, tpr_abc), 3)))

plt.plot([0, 1],[0, 1], ls="--", c=".3")
plt.legend(loc='best')
plt.show()
weight_dt = eli5.formatters.as_dataframe.explain_weights_df(dt.fit(X, y), feature_names=list(X.columns))
weight_dt[weight_dt['weight'] > 0.000001]
weights_lr = eli5.formatters.as_dataframe.explain_weights_df(lr.fit(minmax_X, minmax_y), feature_names=list(X.columns))
weights_lr
weights_abc = eli5.formatters.as_dataframe.explain_weights_df(abc.fit(X, y), feature_names=list(X.columns))
weights_abc[weights_abc['weight'] > 0.000001]