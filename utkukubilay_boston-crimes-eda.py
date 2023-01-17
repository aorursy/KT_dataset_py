import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from pylab import rcParams

from pandas.api.types import CategoricalDtype

import warnings

import os 

import folium

from folium.plugins import HeatMap

import pandas_profiling



rcParams["figure.figsize"] = 20,9

warnings.filterwarnings("ignore")



print(os.listdir("../input/crimes-in-boston"))
df = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

df.head()
df.info()
df.describe().T
def missing_zero_values_table(df):

        zero_val = (df == 0.00).astype(int).sum(axis=0)

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

        mz_table = mz_table.rename(

        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']

        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)

        mz_table['Data Type'] = df.dtypes

        mz_table = mz_table[

            mz_table.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      

            "There are " + str(mz_table.shape[0]) +

              " columns that have missing values.")

        return mz_table



missing_zero_values_table(df)
figure = plt.figure(figsize=(13,6))

sns.heatmap(df.isnull(),yticklabels='')
df.drop("SHOOTING", axis=1, inplace = True)
df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])



#df.MONTH.replace([1,2,3,4,5,6,7,8,9,10,11,12], 

#                 ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 

#                 inplace = True)



df.OFFENSE_CODE_GROUP    = df.OFFENSE_CODE_GROUP.astype(CategoricalDtype())

df.OFFENSE_DESCRIPTION    = df.OFFENSE_DESCRIPTION.astype(CategoricalDtype())

df.DISTRICT    = df.DISTRICT.astype(CategoricalDtype())

df.DAY_OF_WEEK    = df.DAY_OF_WEEK.astype(CategoricalDtype())

df.UCR_PART    = df.UCR_PART.astype(CategoricalDtype())
rename = {'OFFENSE_CODE_GROUP':'Group',

          'SHOOTING':'Shooting',

          'OFFENSE_DESCRIPTION':'Description',

          'DISTRICT':'District',

          'STREET':'Street',        

          'OCCURRED_ON_DATE':'Date',

          'YEAR':'Year',

          'MONTH':'Month',

          'DAY_OF_WEEK':'Day',

          'HOUR':'Hour'}



df.rename(index=str, columns=rename, inplace=True)
def create_features(df):

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['quarter'] = df['Date'].dt.quarter

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['dayofmonth'] = df['Date'].dt.day

    df['weekofyear'] = df['Date'].dt.weekofyear

    

    X = df[['dayofweek','quarter','dayofyear',

            'dayofmonth','weekofyear']]

    return X

create_features(df).head()



df.quarter    = df.quarter.astype(CategoricalDtype())

df.dayofweek    = df.dayofweek.astype(CategoricalDtype())

df.dayofyear    = df.dayofyear.astype(CategoricalDtype())

df.dayofmonth    = df.dayofmonth.astype(CategoricalDtype())
df.columns
df.info()
sns.countplot(data=df, x='Year');
sns.countplot(data=df, x='Month')
sns.countplot(data=df, x='Hour');
rcParams["figure.figsize"] = 20,9

order = df['Group'].value_counts().head(5).index

sns.countplot(data = df, x='Group',hue='District', order = order);
mask = ((df['Year'] == 2016) | (df['Year'] == 2017) | (df['Year'] == 2018))

grouped = df[mask].groupby(['Month','District']).count()

sns.lineplot(data = grouped.reset_index(), x='Month', y='Group',hue='District')
order = df['Group'].value_counts().head(5).index

sns.countplot(data = df, x='Group',hue='quarter', order = order);
sns.catplot(y='Group',

            kind='count',

            height=11, 

            aspect=2,

            order=df.Group.value_counts().index,

            data=df)
labels = df['Group'].astype('category').cat.categories.tolist()

counts = df['Group'].value_counts()

sizes = [counts[var_cat] for var_cat in labels]

fig1, ax1 = plt.subplots(figsize = (22,12))

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140) 

ax1.axis('equal')

plt.show()
sns.catplot(y='Day',

            kind='count',

            height=6, 

            aspect=1,

            order=df.Day.value_counts().index,

            data=df)
grouped = df.groupby(['Month','District']).count()

sns.boxplot(x ="Month", y = "Group", data = grouped.reset_index(), palette="ch:.25");
sns.FacetGrid(data = grouped.reset_index(), 

             hue = "Month",

             height = 5).map(sns.kdeplot, "Group", shade = True).add_legend();
def eda_object(df,feature):

    a = len(df[feature].unique())

    plt.figure(figsize = [20,min(max(8,a),12)])



    plt.subplot(1,2,1)

    x_ = df.groupby([feature])[feature].count()

    x_.plot(kind='pie')

    plt.title(feature)



    plt.subplot(1,2,2)

    cross_tab = pd.crosstab(df['Year'],df[feature],normalize=0).reset_index()

    x_ = cross_tab.melt(id_vars=['Year'])

    x_['value'] = x_['value']



    sns.barplot(x=feature,y='value',hue ='Year',data=x_,palette = ['b','r','g'],alpha =0.7)

    plt.xticks(rotation='vertical')

    plt.title(feature + " - ")





    plt.tight_layout()

    plt.legend()

    plt.show()



rm_list = ['UCR_PART', 'INCIDENT_NUMBER', 'Location', 'Street']

type_list = ['object']

feature_list = []



for feature in df.columns:

    if (feature not in rm_list) & (df[feature].dtypes in type_list):

        feature_list.append(feature)

B2_district=df.loc[df.District=='B2']

for feature in feature_list:

    eda_object(B2_district,feature)
df.info()
def eda_numeric(df,feature):

    x_ = df[feature]

    y_ = df['District']

    data = pd.concat([x_,y_],1)

    plt.figure(figsize=[20,5])



    ax1 = plt.subplot(1,2,1)

    sns.boxplot(x='District',y=feature,data=data)

    plt.title(feature + " - Boxplot")



    ax2 = plt.subplot(1,2,2)

    plt.title(feature+ " - Density")

    

    p1=sns.kdeplot(data[data['District']=="D4"][feature].apply(np.log), color="b",legend=False)

    

    plt.legend(loc='upper right', labels=['0'])



    plt.tight_layout()

    plt.show()

    

rm_list = ['lat', 'long']

type_list = ['int32','int64']

feature_list = []



for feature in df.columns:

    if (feature not in rm_list) & (df[feature].dtypes in type_list) & (len(df[feature].unique()) > 2):

        feature_list.append(feature)
df_drop = df.dropna().copy()



for feature in feature_list:

    eda_numeric(df_drop,feature)
def get_redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations !")

print(get_top_abs_correlations(df.select_dtypes(include=['int32','int64']), 10))
df.Lat.replace(-1, None, inplace=True)

df.Long.replace(-1, None, inplace=True)



rcParams["figure.figsize"] = 21,11



plt.subplots(figsize=(11,6))

sns.scatterplot(x='Lat',

                y='Long',

                hue='District',

                alpha=0.1,

                data=df)

plt.legend(loc=2)
B2_district=df.loc[df.District=='B2'][['Lat','Long']]

B2_district.Lat.fillna(0, inplace = True)

B2_district.Long.fillna(0, inplace = True) 



map_1=folium.Map(location=[42.356145,-71.064083], 

                 tiles = "OpenStreetMap",

                zoom_start=11)



folium.CircleMarker([42.319945,-71.079989],

                        radius=70,

                        fill_color="#b22222",

                        popup='Homicide',

                        color='red',

                       ).add_to(map_1)





HeatMap(data=B2_district, radius=16).add_to(map_1)



map_1
ballistic_crimes=df.loc[df.Group=='Ballistics'][['Lat','Long']]

ballistic_crimes.Lat.fillna(0, inplace = True)

ballistic_crimes.Long.fillna(0, inplace = True) 



map_1=folium.Map(location=[42.356145,-71.064083], 

                 tiles = "Stamen Toner",

                zoom_start=11)



folium.CircleMarker([42.307945,-71.069989],

                        radius=90,

                        fill_color="#b22222",

                        popup='Homicide',

                        color='red',

                       ).add_to(map_1)





HeatMap(data=ballistic_crimes, radius=16).add_to(map_1)



map_1
pandas_profiling.ProfileReport(df)