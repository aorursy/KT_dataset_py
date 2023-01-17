# Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Interactive Plotting
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

# Additional Packages:
# pycountry: ISO country, subdivision, language, currency and script definitions and their translations
# ppscore: implementation of the Predictive Power Score (PPS)
!pip install pycountry-convert
!pip install ppscore
import pycountry
import pycountry_convert as pc
import ppscore as pps

import warnings
warnings.filterwarnings('ignore')
hotel = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
hotel.head()
hotel.info()
hotel.isna().sum().sort_values(ascending = False)
hotel.drop(columns = ['agent', 'company'], inplace = True)
hotel['country'].fillna("UNKNOWN", inplace = True)
hotel['children'].fillna(0, inplace = True)
hotel.isna().values.any()
category_cols = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status']
boolean_cols = ['is_canceled', 'is_repeated_guest']

boolean_map = {1:'Yes', 0:'No'}

hotel['is_canceled'] = hotel['is_canceled'].map(boolean_map)
hotel['is_repeated_guest'] = hotel['is_repeated_guest'].map(boolean_map)

hotel[category_cols + boolean_cols] = hotel[category_cols + boolean_cols].astype('category')
hotel['is_canceled'].cat.reorder_categories(list(boolean_map.values()), inplace = True)
hotel['is_repeated_guest'].cat.reorder_categories(list(boolean_map.values()), inplace = True)

hotel['children'].apply(float.is_integer).all()
hotel['children'] = hotel['children'].astype('int')
hotel['reservation_status_date'] = hotel['reservation_status_date'].astype('datetime64')
hotel.dtypes
hotel['reserved_room_type'].cat.set_categories(hotel['assigned_room_type'].cat.categories, inplace = True)
hotel['is_assigned_as_reserved'] = (hotel['assigned_room_type'] == hotel['reserved_room_type']).astype('category')
hotel['is_assigned_as_reserved']
arrival_date_cols = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
hotel[arrival_date_cols] = hotel[arrival_date_cols].astype(str)
hotel['arrival_date'] = pd.to_datetime(hotel[arrival_date_cols].apply('-'.join, axis = 1), format = "%Y-%B-%d")
hotel.drop(columns = arrival_date_cols + ['arrival_date_week_number'], inplace = True)
hotel['booking_date'] = hotel['arrival_date'] - pd.to_timedelta(hotel['lead_time'], unit = 'days')
hotel[['booking_date', 'arrival_date', 'lead_time']].head()
additional_code2name = {'TMP': 'East Timor'}

def convertCountryCode2Name(code):
    country_name = None
    try:
        if len(code) == 2:
            country_name = pycountry.countries.get(alpha_2 = code).name
        elif len(code) == 3:
            country_name = pycountry.countries.get(alpha_3 = code).name
    except:
        if code in additional_code2name.keys():
            country_name = additional_code2name[code]
    return country_name if country_name is not None else code
    
hotel['country_name'] = hotel['country'].apply(convertCountryCode2Name).astype('category')
hotel['country_name'].head()
additional_name2continent = {'East Timor': 'Asia', 'United States Minor Outlying Islands': 'North America', 'French Southern Territories': 'Antarctica', 'Antarctica': 'Antarctica'}

def convertCountryName2Continent(country_name):
    continent_name = None
    try:
        alpha2 = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    except:
        if country_name in additional_name2continent.keys():
            continent_name = additional_name2continent[country_name]
        else:
            continent_name = "UNKNOWN"
    return continent_name if continent_name is not None else country_name

hotel['continent_name'] = hotel['country_name'].apply(convertCountryName2Continent).astype('category')
hotel['continent_name'].head()
hotel['total_guest'] = hotel[['adults', 'children', 'babies']].sum(axis = 1)
hotel['total_nights'] = hotel[['stays_in_weekend_nights', 'stays_in_week_nights']].sum(axis = 1)

data2plot = [hotel['total_guest'].value_counts().sort_index(ascending = False),
             hotel['total_nights'].value_counts().sort_index(ascending = False)[-21:]]

ylabs = ["Total Guest per Booking (Person)", "Total Nights per Booking"]
titles = ["FREQUENCY OF TOTAL GUEST PER BOOKING\n", "FREQUENCY OF TOTAL NIGHTS PER BOOKING\n(UP TO 20 NIGHTS ONLY)"]

fig, axes = plt.subplots(1, 2, figsize = (15, 5))
for ax, data, ylab, title in zip(axes, data2plot, ylabs, titles):
    bp = data.plot(kind = 'barh', rot = 0, ax = ax)
    for rect in bp.patches:
        height = rect.get_height()
        width = rect.get_width()
        bp.text(rect.get_x() + width, 
                rect.get_y() + height, 
                int(width), 
                ha = 'left',
                va = 'top',
                fontsize = 8)
    bp.set_xlabel("Frequency")
    bp.set_ylabel(ylab)
    ax.set_title(title, fontweight = "bold")
hotel = hotel[(hotel['total_guest'] != 0) & (hotel['total_nights'] != 0)]
hotel.shape
df_cancel_status = pd.crosstab(index = hotel.is_canceled,
                               columns = hotel.reservation_status,
                               margins = True)

ax = df_cancel_status.iloc[:-1,:-1].plot(kind = 'bar', stacked = True, rot = 0)
for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    if height != 0:
        ax.text(rect.get_x() + width, 
                rect.get_y() + height/2, 
                int(height), 
                ha = 'left',
                va = 'center',
                color = "black",
                fontsize = 10)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles = handles, labels = labels)

percent_no = (100*df_cancel_status/df_cancel_status.iloc[-1,-1]).loc["No", "All"]
ax.set_xticklabels(["Yes\n({:.2f} %)".format(100-percent_no), "No\n({:.2f} %)".format(percent_no)])
ax.set_xlabel("Canceled?")
ax.set_ylabel("Number of Bookings")
plt.title("BOOKING CANCELLATION PROPORTION", fontweight = "bold")
plt.show()
df_choropleth = hotel.copy()
df_choropleth['booking_date_year'] = df_choropleth['booking_date'].dt.year
df_country_year_count = df_choropleth.groupby(['country', 'booking_date_year']).count()['hotel'].fillna(0).reset_index() \
                        .rename(columns={'country': 'country_code', 'booking_date_year': 'year', 'hotel':'count'})
df_country_year_count['country_name'] = df_country_year_count['country_code'].apply(convertCountryCode2Name)
df_country_year_count['count'] = df_country_year_count['count'].astype('int')

fig = px.choropleth(df_country_year_count[df_country_year_count["year"] != 2013], 
                    locations = "country_code", color = "count", animation_frame = "year",
                    hover_name = "country_name", 
                    range_color = (0, 5000),
                    color_continuous_scale = px.colors.sequential.Reds,
                    projection = "natural earth")
fig.update_layout(title = 'ANNUAL HOTEL BOOKING COUNTS',
                  template = "seaborn")
fig.show()
ax = pd.crosstab(index = hotel['continent_name'],
                 columns = hotel['is_canceled'],
                 margins = True).sort_values('All').iloc[:-1,:-1].plot(kind = 'barh')
ax.legend(bbox_to_anchor = (1, 1), title = "Canceled?")
ax.set_xlabel("Number of Bookings")
ax.set_ylabel("Continent Name")
ax.set_title("BOOKINGS BY EACH CONTINENT", fontweight = "bold")
plt.show()
ax = (pd.crosstab(index = hotel['continent_name'],
                  columns = hotel['is_canceled'],
                  normalize = 'index').sort_values('Yes') * 100).plot(kind = 'barh', stacked = True)
ax.legend(bbox_to_anchor = (1, 1), title = "Canceled?")
ax.set_xlabel("Percentage of Bookings")
ax.set_ylabel("Continent Name")
ax.set_title("PERCENTAGE OF BOOKINGS CANCELLATION BY EACH CONTINENT", fontweight = "bold")
plt.show()
df_cancellation = hotel.copy()
df_cancellation['date_period'] = df_cancellation['reservation_status_date'].dt.to_period('M')
df_cancellation_percent = df_cancellation.groupby(['date_period', 'is_canceled', 'hotel'])['hotel'].count() \
                            .groupby(['date_period', 'hotel']).apply(lambda x: 100*x/x.sum()) \
                            .unstack(level = 'is_canceled') \
                            .rename(columns = str).reset_index().rename_axis(None, axis = 1).rename(columns = {'hotel': 'Hotel Type'})
df_cancellation_percent['date_period'] = df_cancellation_percent['date_period'].values.astype('datetime64[M]')

fig = px.line(df_cancellation_percent, x = 'date_period', y = 'Yes', color = 'Hotel Type')
fig.update_traces(mode = "markers+lines",
                  hovertemplate = "Rate: %{y:.2f}%")
fig.update_layout(title = 'CANCELLATION RATE OVER TIME BY HOTEL TYPE',
                  xaxis_title = 'Cancellation Period',
                  yaxis_title = 'Cancellation Rate (%)',
                  hovermode = 'x',
                  template = "seaborn",
                  xaxis = dict(tickformat="%b %Y"))
fig.show()
datetime_cols = ['booking_date', 'reservation_status_date', 'arrival_date']
for col in datetime_cols:
    hotel[f"{col}_dayofyear"] = hotel[col].dt.dayofyear

ignore_cols = ['assigned_room_type', 'reserved_room_type', 'country', 'country_name']
hotel_pps_data = hotel.drop(datetime_cols + ignore_cols, axis = 1)

hotel_pps_dummy = pd.get_dummies(hotel_pps_data)
hotel_pps_dummy.head()
pps_score = []
target = 'is_canceled_Yes'
for col in hotel_pps_dummy.columns:
    if col == target:
        continue
    d = {}
    d['feature'] = col
    d['dtypes'] = 'categorical' if hotel_pps_dummy[col].dtypes == 'uint8' else 'numerical'
    d['pps'] = pps.score(hotel_pps_dummy, x = col, y = target, task = 'classification')['ppscore']
    pps_score.append(d)
    
hotel_pps = pd.DataFrame(pps_score).set_index('feature')
hotel_pps.head()
ax = hotel_pps[hotel_pps['dtypes'] == 'numerical'].sort_values('pps')\
        .plot(kind = 'barh', legend = False, figsize = (5, 5))
for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    ax.text(rect.get_x() + width, 
            rect.get_y() + height, 
            round(width, 5), 
            ha = 'left',
            va = 'top',
            fontsize = 8)
ax.set_xlabel("PPS")
ax.set_ylabel("Predictor Variable")
plt.title("NUMERICAL PREDICTORS PREDICTIVE POWER SCORE\n TARGET: is_canceled", fontweight = "bold")
plt.show()
ax = hotel_pps[hotel_pps['dtypes'] == 'categorical'].sort_values('pps')[:-1]\
        .plot(kind = 'barh', legend = False, figsize = (5, 12))
for rect in ax.patches:
    height = rect.get_height()
    width = rect.get_width()
    ax.text(rect.get_x() + width, 
            rect.get_y() + height, 
            round(width, 5), 
            ha = 'left',
            va = 'top',
            fontsize = 8)
ax.set_xlabel("PPS")
ax.set_ylabel("Predictor Variable")
plt.title("CATEGORICAL PREDICTORS PREDICTIVE POWER SCORE\n TARGET: is_canceled", fontweight = "bold")
plt.show()