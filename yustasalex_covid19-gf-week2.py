import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_log_error

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")





paths = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        paths.append(os.path.join(dirname, filename))

        #print(os.path.join(dirname, filename))

        

sorted(paths)
train_df = pd.read_csv(sorted(paths)[10])

test_df = pd.read_csv(sorted(paths)[9])

submission = pd.read_csv(sorted(paths)[8])

population = pd.read_csv(sorted(paths)[11])

country_stats = pd.read_csv(sorted(paths)[7])

age_by_countries = pd.read_csv(sorted(paths)[1])

health_index = pd.read_csv(sorted(paths)[3])
train_df.head()
test_df.head()
submission.head()
population.head()
country_stats.head()
age_by_countries.head()
health_index.head()
population_countries = population['Country (or dependency)'].unique()



country_stats['Country'] = country_stats['Country'].apply(lambda x: x.replace(x[-1],''))

stats_countries = country_stats['Country'].unique()



age_countries = age_by_countries['Country'].unique()

health_countries = health_index['Country'].unique()
t_cols = train_df.columns

t_cols
countries = train_df[t_cols[2]].unique()

test_countries = test_df[t_cols[2]].unique()
not_in_countries = []

for country in countries:

    if country not in population_countries:

        not_in_countries.append(country)

        

not_in_countries_stats = []

for country in countries:

    if country not in stats_countries:

        not_in_countries_stats.append(country)

        

not_in_countries_age = []

for country in countries:

    if country not in age_countries:

        not_in_countries_age.append(country)

        

not_in_countries_health = []

for country in countries:

    if country not in health_countries:

        not_in_countries_health.append(country)
#for item in health_countries:

#    if 'Ad' in item:

#        print(item)
#for i in range(len(not_in_countries_health)):

#    for item in health_countries:

#        if not_in_countries_health[i][:4] in item:

#            print(item)
stats_countries_map = {'Antigua&Barbuda': 'Antigua and Barbuda',

                       'Bahamas,The': 'Bahamas',

                       'Bosnia&Herzegovina': 'Bosnia and Herzegovina',

                       'BurkinaFaso': 'Burkina Faso',

                       'CapeVerde': 'Cabo Verde',

                       'CentralAfricanRep.': 'Central African Republic',

                       'Congo,Dem.Rep.': 'Congo (Kinshasa)',

                       'Congo,Repub.ofthe': 'Congo (Brazzaville)',

                       'CostaRica': 'Costa Rica',

                       "Coted'Ivoire": "Cote d'Ivoire",

                       'CzechRepublic': 'Czechia',

                       'DominicanRepublic': 'Dominican Republic',

                       'ElSalvador': 'El Salvador',

                       'EquatorialGuinea': 'Equatorial Guinea',

                       'Swaziland': 'Eswatini',

                       'Gambia,The': 'Gambia',

                       'Korea,South': 'Korea, South',

                       'NewZealand': 'New Zealand',

                       'Macedonia': 'North Macedonia',

                       'PapuaNewGuinea': 'Papua New Guinea',

                       'SaintKitts&Nevis': 'Saint Kitts and Nevis',

                       'SaintLucia': 'Saint Lucia',

                       'SaintVincentandtheGrenadines': 'Saint Vincent and the Grenadines',

                       'SanMarino': 'San Marino',

                       'SaudiArabia': 'Saudi Arabia',

                       'SouthAfrica': 'South Africa',

                       'SriLanka': 'Sri Lanka',

                       'Taiwan': 'Taiwan*',

                       'EastTimor': 'Timor-Leste',

                       'Trinidad&Tobago': 'Trinidad and Tobago',

                       'UnitedStates': 'US',

                       'UnitedArabEmirates': 'United Arab Emirates',

                       'UnitedKingdom': 'United Kingdom'}



map_state_rev_stat = {k: v for k, v in stats_countries_map.items()}
country_map = {'United States': 'US',

               'Czech Republic (Czechia)': 'Czechia',

               'Congo': 'Congo (Brazzaville)',

               'DR Congo': 'Congo (Kinshasa)',

               'South Korea': 'Korea, South',

               'Taiwan': 'Taiwan*',

               "Côte d'Ivoire": "Cote d'Ivoire",

               'Saint Kitts & Nevis': 'Saint Kitts and Nevis',

               'St. Vincent & Grenadines': 'Saint Vincent and the Grenadines'}



map_state_rev = {k: v for k, v in country_map.items()}
country_age_map = {'Cape Verde': 'Cabo Verde',

                   'Democratic Republic of the Congo': 'Congo (Kinshasa)',

                   'Republic of the Congo': 'Congo (Brazzaville)',

                   'Czech Republic': 'Czechia',

                   'Eswatini (Swaziland)': 'Eswatini',

                   'South Korea': 'Korea, South',

                   'Taiwan': 'Taiwan*',

                   'United States': 'US'}



map_state_rev_age = {k: v for k, v in country_age_map.items()}
health_countries_map = {'Bosnia And Herzegovina': 'Bosnia and Herzegovina',

                        'Czech Republic': 'Czechia',

                        'South Korea': 'Korea, South',

                        'Taiwan': 'Taiwan*',

                        'Trinidad And Tobago': 'Trinidad and Tobago'}



map_state_rev_health = {k: v for k, v in health_countries_map.items()}
population['New_country_name'] = population['Country (or dependency)'].apply(lambda x: country_map[x] if x in

                                                                             map_state_rev else x)



country_stats['New_country_name'] = country_stats['Country'].apply(lambda x: stats_countries_map[x] if x in

                                                                  map_state_rev_stat else x)



age_by_countries['New_country_name'] = age_by_countries['Country'].apply(lambda x: country_age_map[x] if x in

                                                                  map_state_rev_age else x)



health_index['New_country_name'] = health_index['Country'].apply(lambda x: health_countries_map[x] if x in

                                                                  map_state_rev_health else x)
health_care_idx = {}

health_care_exp_idx = {}



for item in set(health_index['New_country_name']):

    health_care_idx[item] = health_index.loc[health_index['New_country_name'] == item, 'Health Care Index'].values[0]

    health_care_exp_idx[item] = health_index.loc[health_index['New_country_name'] == item, 'Health Care Exp. Index'].values[0]
pop = {}

dens = {}

l_area = {}

age = {}

urban_pop = {}



for item in set(population['New_country_name']):

    pop[item] = population.loc[population['New_country_name'] == item, 'Population (2020)'].values[0]

    dens[item] = population.loc[population['New_country_name'] == item, 'Density (P/Km²)'].values[0]

    l_area[item] = population.loc[population['New_country_name'] == item, 'Land Area (Km²)'].values[0]

    age[item] = population.loc[population['New_country_name'] == item, 'Med. Age'].values[0]

    urban_pop[item] = population.loc[population['New_country_name'] == item, 'Urban Pop %'].values[0]
for key, val in urban_pop.items():

    if val != 'N.A.':

        urban_pop[key] = int(val.replace('%', '')) / 100

    else:

        urban_pop[key] = -1
coastline = {}

inf_mort = {}

gdp = {}

literacy = {}

phones = {}

arable = {}

crops = {}

other = {}

climate = {}

birhrate = {}

deathrate = {}

agri = {}

industry = {}

service = {}



for item in set(country_stats['New_country_name']):

    coastline[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Coastline (coast/area ratio)'].values[0]

    inf_mort[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Infant mortality (per 1000 births)'].values[0]

    gdp[item] = country_stats.loc[country_stats['New_country_name'] == item, 'GDP ($ per capita)'].values[0]

    literacy[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Literacy (%)'].values[0]

    phones[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Phones (per 1000)'].values[0]

    arable[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Arable (%)'].values[0]

    crops[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Crops (%)'].values[0]

    other[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Other (%)'].values[0]

    climate[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Climate'].values[0]

    birhrate[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Birthrate'].values[0]

    deathrate[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Deathrate'].values[0]

    agri[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Agriculture'].values[0]

    industry[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Industry'].values[0]

    service[item] = country_stats.loc[country_stats['New_country_name'] == item, 'Service'].values[0]
train_df['Population'] = train_df['Country_Region'].map(pop)

train_df['Density'] = train_df['Country_Region'].map(dens)

train_df['Land_Area'] = train_df['Country_Region'].map(l_area)

train_df['Med_Age'] = train_df['Country_Region'].map(age)

train_df['Urban_Pop'] = train_df['Country_Region'].map(urban_pop)

train_df['Coastline'] = train_df['Country_Region'].map(coastline)

train_df['Infant_mortality'] = train_df['Country_Region'].map(inf_mort)

train_df['GDP'] = train_df['Country_Region'].map(gdp)

train_df['Literacy'] = train_df['Country_Region'].map(literacy)

train_df['Phones'] = train_df['Country_Region'].map(phones)

train_df['Arable'] = train_df['Country_Region'].map(arable)

train_df['Crops'] = train_df['Country_Region'].map(crops)

train_df['Other'] = train_df['Country_Region'].map(other)

train_df['Climate'] = train_df['Country_Region'].map(climate)

train_df['Birthrate'] = train_df['Country_Region'].map(birhrate)

train_df['Deathrate'] = train_df['Country_Region'].map(deathrate)

train_df['Agriculture'] = train_df['Country_Region'].map(agri)

train_df['Industry'] = train_df['Country_Region'].map(industry)

train_df['Service'] = train_df['Country_Region'].map(service)

train_df['Health_Care_Index'] = train_df['Country_Region'].map(health_care_idx)

train_df['Health_Care_Exp_Index'] = train_df['Country_Region'].map(health_care_exp_idx)



test_df['Population'] = test_df['Country_Region'].map(pop)

test_df['Density'] = test_df['Country_Region'].map(dens)

test_df['Land_Area'] = test_df['Country_Region'].map(l_area)

test_df['Med_Age'] = test_df['Country_Region'].map(age)

test_df['Urban_Pop'] = test_df['Country_Region'].map(urban_pop)

test_df['Coastline'] = test_df['Country_Region'].map(coastline)

test_df['Infant_mortality'] = test_df['Country_Region'].map(inf_mort)

test_df['GDP'] = test_df['Country_Region'].map(gdp)

test_df['Literacy'] = test_df['Country_Region'].map(literacy)

test_df['Phones'] = test_df['Country_Region'].map(phones)

test_df['Arable'] = test_df['Country_Region'].map(arable)

test_df['Crops'] = test_df['Country_Region'].map(crops)

test_df['Other'] = test_df['Country_Region'].map(other)

test_df['Climate'] = test_df['Country_Region'].map(climate)

test_df['Birthrate'] = test_df['Country_Region'].map(birhrate)

test_df['Deathrate'] = test_df['Country_Region'].map(deathrate)

test_df['Agriculture'] = test_df['Country_Region'].map(agri)

test_df['Industry'] = test_df['Country_Region'].map(industry)

test_df['Service'] = test_df['Country_Region'].map(service)

test_df['Health_Care_Index'] = test_df['Country_Region'].map(health_care_idx)

test_df['Health_Care_Exp_Index'] = test_df['Country_Region'].map(health_care_exp_idx)
train_df['Urban_pop_num'] = train_df[['Population', 'Urban_Pop']].apply(lambda x: x[0]*x[1], axis=1)

test_df['Urban_pop_num'] = test_df[['Population', 'Urban_Pop']].apply(lambda x: x[0]*x[1], axis=1)
age_cols = ['Age 0 to 14 Years', 'Age 15 to 64 Years', 'Age above 65 Years']

for item in age_cols:

    for i in range(len(age_by_countries)):

        if type(age_by_countries[item][i]) == str:

            age_by_countries[item][i] = age_by_countries[item][i].replace('%','')

            age_by_countries[item][i] = float(age_by_countries[item][i]) / 100
age_014 = {}

age_1564 = {}

age_65plus = {}



for item in set(age_by_countries['New_country_name']):

    age_014[item] = age_by_countries.loc[age_by_countries['New_country_name'] == item, 'Age 0 to 14 Years'].values[0]

    age_1564[item] = age_by_countries.loc[age_by_countries['New_country_name'] == item, 'Age 15 to 64 Years'].values[0]

    age_65plus[item] = age_by_countries.loc[age_by_countries['New_country_name'] == item, 'Age above 65 Years'].values[0]
train_df['age_0-14'] = train_df['Country_Region'].map(age_014)

train_df['age_15-64'] = train_df['Country_Region'].map(age_1564)

train_df['age_65plus'] = train_df['Country_Region'].map(age_65plus)



test_df['age_0-14'] = test_df['Country_Region'].map(age_014)

test_df['age_15-64'] = test_df['Country_Region'].map(age_1564)

test_df['age_65plus'] = test_df['Country_Region'].map(age_65plus)
train_df = train_df.fillna(-1)

test_df = test_df.fillna(-1)
def time_feat(df, start_day):

    start_day = datetime.strptime(start_day, '%Y-%m-%d').date()

    df['Date_time'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())

    df['Time_delta'] = df['Date_time'].apply(lambda x: (x - start_day).days)

    #df['Weekday'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())

    #df['Day_of_month'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)

    df['Month'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)

    

    return df
train_df = time_feat(train_df, '2020-01-22')

test_df = time_feat(test_df, '2020-01-22')
cols = train_df.columns

cols
def to_str(x):

    if x == -1:

        return str(x)

    return x
train_df[cols[1]] = train_df[cols[1]].apply(lambda x: to_str(x))

test_df[cols[1]] = test_df[cols[1]].apply(lambda x: to_str(x))
features=['Province_State', 'Country_Region', 'Population', 'Density', 'Land_Area', 'Med_Age',

          'Urban_Pop', 'Coastline', 'Infant_mortality', 'GDP', 'Literacy',

          'Phones', 'Arable', 'Crops', 'Other', 'Climate', 'Birthrate', 'Health_Care_Index', 

          'Health_Care_Exp_Index','Deathrate', 'Agriculture', 'Industry', 'Service', 'Urban_pop_num',

          'Time_delta', 'Month', 'age_0-14', 'age_15-64', 'age_65plus']



train_X = train_df[features + ['ConfirmedCases', 'Fatalities']].copy()

train_yc = train_df['ConfirmedCases'].copy()

train_yf = train_df['Fatalities'].copy()
for i in range(len(train_X)):

    if train_X['Med_Age'][i] == 'N.A.':

            train_X['Med_Age'][i] = -1

            

for feature in features:

    for i in range(len(test_df)):

        if test_df[feature][i] == 'N.A.':

            test_df[feature][i] = -1
for item in test_df[cols[2]].unique():

    if item not in train_df[cols[2]].unique():

        print(item)
enc1 = OrdinalEncoder()

enc2 = OrdinalEncoder()

enc1.fit(train_X[cols[1]].to_numpy().reshape(-1, 1))

enc2.fit(train_X[cols[2]].to_numpy().reshape(-1, 1))

train_X[cols[1]] = enc1.transform(train_X[cols[1]].to_numpy().reshape(-1, 1))

train_X[cols[2]] = enc2.transform(train_X[cols[2]].to_numpy().reshape(-1, 1))

test_df[cols[1]] = enc1.transform(test_df[cols[1]].to_numpy().reshape(-1, 1))

test_df[cols[2]] = enc2.transform(test_df[cols[2]].to_numpy().reshape(-1, 1))
check_list = ['Coastline', 'Infant_mortality', 'GDP', 'Literacy','Phones', 'Arable', 'Crops', 'Other', 

              'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']



for item in check_list:

    for i in range(len(train_X)):

        if type(train_X[item][i]) == str:

            train_X[item][i] = train_X[item][i].replace(',','.')

            train_X[item][i] = float(train_X[item][i])

            

for item in check_list:

    for i in range(len(test_df)):

        if type(test_df[item][i]) == str:

            test_df[item][i] = test_df[item][i].replace(',','.')

            test_df[item][i] = float(test_df[item][i])
train_X['Agri_pop_num'] = train_X[['Population', 'Agriculture']].apply(lambda x: x[0]*x[1], axis=1)

train_X['Industry_pop_num'] = train_X[['Population', 'Industry']].apply(lambda x: x[0]*x[1], axis=1)

train_X['Service_pop_num'] = train_X[['Population', 'Service']].apply(lambda x: x[0]*x[1], axis=1)

train_X['age_0-14_num'] = train_X[['Population', 'age_0-14']].apply(lambda x: x[0]*x[1], axis=1)

train_X['age_15-64_num'] = train_X[['Population', 'age_15-64']].apply(lambda x: x[0]*x[1], axis=1)

train_X['age_65plus_num'] = train_X[['Population', 'age_65plus']].apply(lambda x: x[0]*x[1], axis=1)



test_df['Agri_pop_num'] = test_df[['Population', 'Agriculture']].apply(lambda x: x[0]*x[1], axis=1)

test_df['Industry_pop_num'] = test_df[['Population', 'Industry']].apply(lambda x: x[0]*x[1], axis=1)

test_df['Service_pop_num'] = test_df[['Population', 'Service']].apply(lambda x: x[0]*x[1], axis=1)

test_df['age_0-14_num'] = test_df[['Population', 'age_0-14']].apply(lambda x: x[0]*x[1], axis=1)

test_df['age_15-64_num'] = test_df[['Population', 'age_15-64']].apply(lambda x: x[0]*x[1], axis=1)

test_df['age_65plus_num'] = test_df[['Population', 'age_65plus']].apply(lambda x: x[0]*x[1], axis=1)
features = features + ['Agri_pop_num', 'Industry_pop_num', 'Service_pop_num', 

                       'age_0-14_num', 'age_15-64_num', 'age_65plus_num']
features.remove('Agriculture')

features.remove('Industry')

features.remove('Service')

features.remove('Urban_Pop')

features.remove('age_0-14')

features.remove('age_15-64')

features.remove('age_65plus')

#features.remove('Weekday')

#features.remove('Day_of_month')
matrix = train_X.corr(method='spearman')

mask = np.triu(np.ones_like(matrix, dtype=np.bool))

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(15, 12))

    ax = sns.heatmap(matrix, mask=mask, annot=True, cmap="YlGnBu",vmax=.3, square=True, linewidths=.4)

plt.show();
scaler = StandardScaler()

train_X = scaler.fit_transform(train_X[features])

train_X = pd.DataFrame(data=train_X, columns=features)

test = test_df[features].copy()

test = scaler.transform(test)

test = pd.DataFrame(data=test, columns=features)
train = train_X[features]

skf = StratifiedKFold(n_splits=2020, random_state=42)

score_c = []

preds_c = []

for i, (tdx, vdx) in enumerate(skf.split(train, train_yc)):

    #print(f'Fold : {i}')

    X_train, X_val, y_train, y_val = train.iloc[tdx], train.iloc[vdx], train_yc[tdx], train_yc[vdx]

    #model_c = RandomForestClassifier(bootstrap=False,

    #                               criterion='entropy',

    #                               max_features=0.4, 

    #                               min_samples_leaf=14, 

    #                               min_samples_split=5,

    #                               n_estimators=500, 

    #                               n_jobs=-1, 

    #                               random_state=42)

    

    #model_c = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', n_jobs=-1)

    model_c = KNeighborsRegressor(n_neighbors=1, weights='distance', algorithm='auto', n_jobs=-1)

    

    model_c.fit(X_train, y_train)

    val_preds_c = model_c.predict(X_val.to_numpy())

    val_score_c = np.sqrt(mean_squared_log_error(y_val, abs(val_preds_c)))

    #print('val_score: ', val_score_c)

    score_c.append(val_score_c)

    pred_c = model_c.predict(test[features])

    preds_c.append(abs(pred_c))

print(np.mean(score_c))
skf = StratifiedKFold(n_splits=2020, random_state=42)

score_f = []

preds_f = []

for i, (tdx, vdx) in enumerate(skf.split(train, train_yf)):

    #print(f'Fold : {i}')

    X_train, X_val, y_train, y_val = train.iloc[tdx], train.iloc[vdx], train_yf[tdx], train_yf[vdx]

    #model_f = RandomForestClassifier(bootstrap=False,

    #                               criterion='entropy',

    #                               max_features=0.4, 

    #                               min_samples_leaf=14, 

    #                               min_samples_split=5, 

    #                               n_estimators=600, 

    #                               n_jobs=-1, 

    #                               random_state=42)

    

    #model_f = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', n_jobs=-1)

    model_f = KNeighborsRegressor(n_neighbors=1, weights='distance', algorithm='auto', n_jobs=-1)

    

    model_f.fit(X_train, y_train)

    val_preds_f = model_f.predict(X_val)

    val_score_f = np.sqrt(mean_squared_log_error(y_val, abs(val_preds_f)))

    #print('val_score: ', val_score_f)

    score_f.append(val_score_f)

    pred_f = model_f.predict(test[features])

    preds_f.append(abs(pred_f))

print(np.mean(score_f))
c_mean = np.mean(score_c)

f_mean = np.mean(score_f)

(c_mean + f_mean) / 2
#model_c_fi = pd.DataFrame(data=model_c.feature_importances_.reshape(-1, 1),

#                          index=np.asarray(features).reshape(-1, 1),

#                          columns=['model_c feature_importances'])

#model_f_fi = pd.DataFrame(data=model_f.feature_importances_.reshape(-1, 1),

#                          index=np.asarray(features).reshape(-1, 1),

#                          columns=['model_f feature_importances'])
#def select_feats(df, coeff):

#    res = []

#    for i in range(len(df.values)):

#        if df.values[i] >= coeff:

#            res.append(df.index[i][0])

#    return res
#model_c_fi.sort_values(by='model_c feature_importances', ascending=False)
#model_f_fi.sort_values(by='model_f feature_importances', ascending=False)
#c_feats = select_feats(model_c_fi, 0.015)

#f_feats = select_feats(model_f_fi, 0.016)
cc_preds = np.vstack(preds_c)

f_preds = np.vstack(preds_f)

y_pred_cc = np.around(np.mean([cc_preds],axis=1))

y_pred_f = np.around(np.mean([f_preds],axis=1))
submission['ConfirmedCases'] = y_pred_cc[0].astype(int)

submission['Fatalities'] = y_pred_f[0].astype(int)

submission.to_csv('submission.csv', index=False)
submission.head()
#concl_df = pd.read_csv(sorted(paths)[9])

#concl_feats = ['ForecastId', 'Country_Region', 'Date']

#conclusion = pd.concat([concl_df[concl_feats], submission[['ConfirmedCases', 'Fatalities']]], axis=1)
#conclusion[conclusion['Country_Region'] == 'Russia']