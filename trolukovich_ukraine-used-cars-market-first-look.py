import pandas as pd

import numpy as np

import matplotlib.cm

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import plotly.graph_objs as go

import cufflinks

from ast import literal_eval

from sklearn.preprocessing import LabelEncoder

from plotly import tools

from plotly.offline import init_notebook_mode, iplot



plt.style.use('ggplot')

%matplotlib inline

pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')

cufflinks.go_offline(connected=True)

init_notebook_mode(connected = True)
data = pd.read_csv('../input/cars_ver2.csv', encoding = 'windows-1251')
data.head(3)
# Making some changes in data

data['region'] = data['region'].str.replace(' область', '')

data['body_type'] = data['body_type'].replace(['Passenger van (up to 1.5 tons)', 'SUV / Crossover'], ['Pass. van', 'SUV'])



# Function to draw values on plots

def annot(ax, val):

    '''Draws values on plot'''

    for p, i in zip(ax.patches, val):

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

        ax.annotate('{}'.format(i), (x.mean(), y), 

                ha='center', va='bottom')



# Changing size of plots

fig = plt.figure(figsize = (18, 8))



# First plot

plt.subplot(1, 2, 1)

ax = sns.countplot(y = 'region', data = data, order=data.region.value_counts().iloc[:15].index)

plt.title('Adverts by region')

val = data.region.value_counts().iloc[:15].values 

annot(ax, val)



# Second plot

plt.subplot(1, 2, 2)

ax2 = sns.countplot(y = 'city', data = data, order=data.city.value_counts().iloc[:15].index)

plt.title('Adverts by city')

val2 = data.city.value_counts().iloc[:15].values

annot(ax2, val2)
# Preparing data for plots

mark = data.mark.value_counts().iloc[:15]

model = data.model.value_counts().iloc[:15]

body = data['body_type'].value_counts().sort_values(ascending = False)

color = data['color'].value_counts().sort_values(ascending = False)

fuel = data['fuel'].value_counts().sort_values(ascending = False)

gear = data['gearbox'].value_counts().sort_values(ascending = False)



# Making traces for each plot

trace1 = go.Bar(x = mark.index, y = mark.values)

trace2 = go.Bar(x = model.index, y = model.values)

trace3 = go.Bar(x = body.index, y = body.values)

trace4 = go.Bar(x = color.index, y = color.values)

trace5 = go.Bar(x = fuel.index, y = fuel.values)

trace6 = go.Bar(x = gear.index, y = gear.values)



# Making subplots

fig = tools.make_subplots(rows = 3, cols = 2, 

                          subplot_titles = ['Most frequent marks', 'Most frequent models',

                                            'Types of cars', 'Colors',

                                            'Types of fuel', 'Types of gearbox', ])



# Appending traces to fig

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace5, 3, 1)

fig.append_trace(trace6, 3, 2)



# Tuning layout

fig['layout'].update(height = 1000, showlegend=False)



# Show plot

iplot(fig)
# Prepairing our data

year = data['year'].value_counts().sort_index()[1:]

mile = data['mileage'].value_counts().sort_index()[1:]



# Making traces

trace1 = go.Scatter(x = year.index, y = year.values)

trace2 = go.Scatter(x = mile.index, y = mile.values)



# Setting subplots and titles

fig = tools.make_subplots(rows = 1, cols = 2, subplot_titles = ['Distribution by year', 'Distribution by mileage'])



# Appending traces

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)



# Remove legend

fig['layout'].update(showlegend = False)



# Show plot

iplot(fig)
data['year'] = data['year'].replace(0, np.nan)

data.loc[(data['year'] <= 1950) | (data['year'] == 2020)].dropna(subset = ['year']).sort_values(by = 'year')
data.drop(data.loc[(data['year'] == 1900) | (data['year'] == 2020) | (data['mileage'] > 500000) 

                  | (data['engine_vol'] > 5000)].index, inplace = True)
sns.countplot(data['cleared'])
# Adding additional feature log_price

data['log_price'] = np.log1p(data['price'])



# Determine columns which we want to plot

cols = ['year', 'mileage', 'engine_vol', 'year', 'mileage', 'engine_vol']



# Plotting them with cycle

fig = plt.figure(figsize = (24, 12))

for i, j in enumerate(cols):

    fig.add_subplot(2, 3, i+1) # We adding subplot on each step

    plt.tight_layout() # This function makes our plots look better, adjusting distance between plots

    if i <= 2:

        sns.scatterplot(x = j, y = 'price', data = data)

    else:

        sns.regplot(x = j, y = 'log_price', data = data)
cols = ['body_type', 'body_type', 'fuel', 'fuel', 'gearbox', 'gearbox', 'cleared', 'cleared']



fig = plt.figure(figsize = (15, 20))

for i, j in enumerate(cols):

    fig.add_subplot(4, 2, i+1)

    plt.tight_layout()

    if i%2 == 0:

        sns.boxplot(x = j, y = 'price', data = data)

        if j == 'body_type':

            plt.xticks(rotation=45)

    else:

        sns.boxplot(x = j, y = 'log_price', data = data)

        if j == 'body_type':

            plt.xticks(rotation=45)
# Plotting mark feature

mark1 = data[data['mark'].isin(data['mark'].unique()[:23])]

mark2 = data[data['mark'].isin(data['mark'].unique()[23:])]



fig = plt.figure(figsize = (15, 20))

for i, j in enumerate([mark1, mark2, mark1, mark2]):

    fig.add_subplot(4, 1, i+1)

    plt.tight_layout()

    if i<2:

        sns.boxplot(x = 'mark', y = 'price', data = j)

        plt.xticks(rotation=45)

    else:

        sns.boxplot(x = 'mark', y = 'log_price', data = j)

        plt.xticks(rotation=45)
cols = ['color', 'region', 'color', 'region']

fig = plt.figure(figsize = (15, 20))

for i, j in enumerate(cols):

    fig.add_subplot(4, 1, i+1)

    plt.tight_layout()

    if i<2:

        sns.boxplot(x = j, y = 'price', data = data)

        plt.xticks(rotation=45)

    else:

        sns.boxplot(x = j, y = 'log_price', data = data)

        plt.xticks(rotation=45)
lists = ['condition', 'add_opt', 'multimedia', 'security', 'other']



# I'll create DataFrame for these columns

df = data[lists]



# Pandas loads lists from csv as string, so first we need to convert it to list

# First fill nan

df = df.fillna('["XXX"]')



# Then use Literal_eval to convert strings to lists

for col in lists:

    df[col] = pd.Series([literal_eval(i) for i in df[col].values])
df.head()
# Now we can look at unique values in columns

df['condition'].apply(pd.Series).stack().value_counts()
# And now we can start to create dummy variables from lists

def dummies(vals, target):

    '''Creates dummy variables for vals'''

    for val in vals:

        df[val] = 0

        df.loc[df[target].str.contains(val, regex = False), val] = 1        

    df.drop(target, axis = 1, inplace = True)

    

df = df.fillna('XXX')



# Create vals for each column

cond = df['condition'].apply(pd.Series).stack().value_counts().index.values

add = df['add_opt'].apply(pd.Series).stack().value_counts().index.values

mult = df['multimedia'].apply(pd.Series).stack().value_counts().index.values

sec = df['security'].apply(pd.Series).stack().value_counts().index.values

oth = df['other'].apply(pd.Series).stack().value_counts().index.values



# Applying our function

dummies(cond, 'condition')

dummies(add, 'add_opt')

dummies(mult, 'multimedia')

dummies(sec, 'security')

dummies(oth, 'other')



# Check results

df.drop('XXX', axis = 1, inplace = True)

df.head()
# I'll also rename columns

cols = ['gar_storage', 'no_accidents', 'unpainted', 'first_reg',

       'service_book', 'first_owner', 'rep_required',

       'not_on_the_run', 'after_accident', 'taken_on_credit',

       'el_windows', 'power_steering', 'air_conditioning',

       'computer', 'electro_package', 'heated_mirr',

       'security_sys', 'climate_cntrl', 'mf_steering_wheel',

       'cruise_control', 'heated_seats', 'parktronic', 'light_sensor',

       'leather_int', 'rain_sensor', 'sunroof', 'headlights_wash',

       'start_btn', 'heated_steer_wheel', 'cassete_player', 'cd', 'aux', 'usb',

       'acoustics', 'bluetooth', 'gps', 'amplifier',

       'subwoofer', 'central_lock', 'abs',

       'airbag', 'alarm', 'esp',

       'immobilizer', 'halogen_headlights', 'servo_steer_wheel', 'abd',

       'lock_on_gearbox', 'air_susp', 'armored_car',

       'toning', 'gas', 'hook',

       'tuning', 'right_wheel']



df.columns = cols

print(df.shape)

df.head()
# Now we can join our dataframes

data = data.join(df)

data.head(2)
# And now we can make some plots to see, haow these features affects on price

fig = plt.figure(figsize = (15, 30))

for i, col in enumerate(df.columns.values):

    fig.add_subplot(11, 5, i+1)

    plt.tight_layout()

    sns.violinplot(x = col, y = 'price', data = data)
corrmap = data.corr().sort_values(by = 'price', ascending = False)

corrmap['price'].head(15)
corrmap['price'].tail(15)