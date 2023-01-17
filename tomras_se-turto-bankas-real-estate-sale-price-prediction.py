import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier

from sklearn import metrics

from math import sqrt

from tqdm import tqdm

from tabulate import tabulate



%matplotlib inline

tqdm.pandas()
df = pd.read_csv('../input/SE Turto Bankas real estate auctions.csv',

                 sep=';', 

                 header=0,

                 decimal=',',

                 names=['auction_date', 

                        'real_estate_name', 

                        'real_estate_address', 

                        'general_area', 

                        'initial_sale_price', 

                        'final_sale_price'])



# parsing string to date

df['auction_date'] = pd.to_datetime(df['auction_date']) 



# checking dataset columns type

print(df.dtypes)



df.head(n=10)
null_columns = df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
df['real_estate_name'] = df['real_estate_name'].fillna('Nepateikta')



imputer = Imputer(missing_values='NaN', 

                  strategy='mean')



imputer = imputer.fit(df[['general_area']])

df['general_area'] = imputer.transform(df[['general_area']]).ravel().round(2)



df.head(n=10)
df.describe(include='all')
fig, axs = plt.subplots(1,2, figsize=(14,5))



df.plot.scatter(x='initial_sale_price', y='final_sale_price', ax=axs[0])

sns.boxplot(x='initial_sale_price', data=df, ax=axs[1])



plt.tight_layout()
df = df[df['initial_sale_price'] <= 300000]

df = df[df['final_sale_price'] <= 500000]



fig, axs = plt.subplots(1,2, figsize=(14,5))



df.plot.scatter(x='initial_sale_price', y='final_sale_price', ax=axs[0])

sns.boxplot(x='initial_sale_price', data=df, ax=axs[1])



plt.tight_layout()
equal_price_mask = df['initial_sale_price'] == df['final_sale_price']



equal_count = len(df[equal_price_mask].index)

not_equal_count = len(df.index) - equal_count



tmp_data = [['initial price', equal_count],

            ['higher price', not_equal_count]]



tmp_df = pd.DataFrame(tmp_data, columns=['Name','Count'])



tmp_df.plot(kind='bar', x='Name')
df.plot.scatter(x='final_sale_price', y='general_area')
def real_estate_name_to_type(real_estate_name):

    is_any = lambda name, type_words: any(i in name for i in type_words)

    real_estate_name = real_estate_name.lower()

    

    apartment = ['butas', 'butą']

    basement = ['rūsiu', 'rūsys']

    plot = ['sklypas', 'sklypu', 'sklypo']

    house = ['namas', 'namu', 'namelis']

    building = ['pastatas', 'patalpa', 'mokykla', 'fortas']

    administrative_quarters = ['administracinis', 'administracinės']

    garage = ['garažas', 'garažą']

    

    if is_any(real_estate_name, apartment) and is_any(real_estate_name, basement):

        return 'apartment_with_basement'

    elif is_any(real_estate_name, administrative_quarters) and is_any(real_estate_name, plot):

        return 'administrative_quarters_with_plot'

    elif is_any(real_estate_name, house) and is_any(real_estate_name, plot):

        return 'house_with_plot'

    elif is_any(real_estate_name, building) and is_any(real_estate_name, plot):

        return 'building_with_plot'

    elif is_any(real_estate_name, garage) and is_any(real_estate_name, plot):

        return 'garage_with_plot'

    elif is_any(real_estate_name, house):

        return 'house'

    elif is_any(real_estate_name, apartment):

        return 'apratment'

    elif is_any(real_estate_name, administrative_quarters):

        return 'administrative_quarters'

    elif is_any(real_estate_name, building):

        return 'building'

    elif is_any(real_estate_name, garage):

        return 'garage'

    elif is_any(real_estate_name, plot):

        return 'plot'

    else: 

        return 'building'
def real_estate_address_to_region(real_estate_address):

    region = real_estate_address.lower().split(',')[-1].split()[0]

    

    if region in ('alytus', 'alytaus', 'druskininkai', 'druskininkų', 'lazdijai', 'lazdijų', 'varėna', 'varėnos'):

        return 'alytus'

    elif region in ('kaunas', 'kauno', 'birštonas', 'birštono', 'jonava', 'jonavos', 'kaišiadorys', 'kaišiadorių', 'kėdainiai', 'kėdainių', 'prienai', 'prienų', 'raseiniai', 'raseinių'):

        return 'kaunas'

    elif region in ('klaipėda', 'klaipėdos', 'kretinga', 'kretingos', 'neringa', 'neringos', 'palanga', 'palangos', 'skuodo', 'skuodas', 'šilutės', 'šilutė'):

        return 'klaipeda'

    elif region in ('marijampolė', 'marijampolės', 'kalvarijos', 'kalvarija', 'kazlų', 'šakiai', 'šakių', 'vilkaviškis', 'vilkaviškio'):

        return 'marijampole'

    elif region in ('panevėžio', 'panevėžys', 'biržai', 'biržų', 'kupiškis', 'kupiškio', 'pasvalys', 'pasvalio', 'rokiškis', 'rokiškio', 'naujamiesčio'):

        return 'panevezys'

    elif region in ('šiauliai', 'šiaulių', 'akmenė', 'akmenės', 'joniškis', 'joniškio', 'kelmė', 'kelmės', 'pakruojis', 'pakruojo', 'radviliškis', 'radviliškio', 'naujoji', 'martyniškių', 'dubijos', 'saveikių'):

        return 'siauliai'

    elif region in ('tauragė', 'tauragės', 'jurbarkas', 'jurbarko', 'pagėgiai', 'pagėgių', 'šilalė', 'šilalės'):

        return 'taurage'

    elif region in ('telšių', 'telšiai', 'mažeikiai', 'mažeikių', 'plungė', 'plungės', 'rietavas', 'rietavo'):

        return 'telsiai'

    elif region in ('utena', 'utenos', 'anykščiai', 'anykščių', 'ignalina', 'ignalinos', 'molėtai', 'molėtų', 'visaginas', 'visagino', 'zarasai', 'zarasų'):

        return 'utena'

    elif region in ('vilnius', 'vilniaus', 'elektrėnai', 'elektrėnų', 'šalčininkai', 'šalčininkų', 'širvintų', 'širvintos', 'švenčionių', 'švenčionys', 'trakų', 'trakai', 'ukmergės', 'ukmergė'):

        return 'vilnius'

    elif region in ('latvija'):

        return 'latvija'

    else:

        return region
df['real_estate_type'] = df['real_estate_name'].apply(real_estate_name_to_type)

df['real_estate_name_words_count'] = df['real_estate_name'].apply(lambda x: len(x.split()))

df['real_estate_in_city'] = df['real_estate_address'].apply(lambda x: 0 if 'r.' in x.split(',')[-1] else 1)

df['real_estate_region'] = df['real_estate_address'].apply(real_estate_address_to_region)

df['log_initial_sale_price'] = df['initial_sale_price'].apply(lambda x : np.log(x))

df['log_final_sale_price'] = df['final_sale_price'].apply(lambda x : np.log(x))

df.head(n=20)
df = pd.get_dummies(df, 

                    columns=['real_estate_type', 'real_estate_region'], 

                    prefix=['real_estate_type', 'real_estate_region'])

df.head(n=20)
scaler = MinMaxScaler()

df[['general_area', 'real_estate_name_words_count']] = scaler.fit_transform(df[['general_area', 

                                                                                'real_estate_name_words_count']])



df.head()
feature_columns = [col for col in df.columns if col not in ['auction_date',

                                                            'real_estate_name',

                                                            'real_estate_address',

                                                            'final_sale_price',

                                                            'log_final_sale_price']]



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)



X_train = train_df[feature_columns]

y_train = train_df['final_sale_price']

X_test = test_df[feature_columns]

y_test = test_df['final_sale_price']



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
models = {

    'linear_regression': LinearRegression(),

    'decision_tree_regression': DecisionTreeRegressor(random_state=42),

    'random_forest_regression': RandomForestRegressor(random_state=42),

    'gradient_boosting_regressor': GradientBoostingRegressor(random_state=42)

}
df_with_pred = test_df.copy()



for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    df_with_pred['predicted_sale_price_%s' % name] = y_pred

    print('%s: R2 -> %s' % (name, metrics.r2_score(y_test, y_pred)))

    print('%s: RMSE -> %s' % (name, sqrt(metrics.mean_squared_error(y_test, y_pred))))



models_keys = ['predicted_sale_price_%s' % n for n in models.keys()]

models_keys.append('final_sale_price')



df_with_pred[models_keys].head(n=20)
df['higher_price'] = [0 if row['initial_sale_price'] == row['final_sale_price'] else 1 for index, row in df.iterrows()]

df.head()
train_df, test_df = train_test_split(df, stratify=df['higher_price'], test_size=0.2)



X_train = train_df[feature_columns]

y_train = train_df['higher_price']

X_test = test_df[feature_columns]

y_test = test_df['higher_price']



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
models = {

    'gradient_boosting_classifier': GradientBoostingClassifier(random_state=42),

    'random_forest_classifier': RandomForestClassifier(random_state=42)

}
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('%s %s %s' % ('#' * 10, name, '#' * 10))

    print(metrics.classification_report(y_test, y_pred))

    print('Confution matrix')

    print(metrics.confusion_matrix(y_test, y_pred))
headers = ['name', 'score']

values = sorted(zip(X_train.columns, models['random_forest_classifier'].feature_importances_), key=lambda x: x[1] * -1)

print(tabulate(values, headers, tablefmt='plain'))