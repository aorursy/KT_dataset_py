'''Importing Data Manipulattion Moduls'''

import numpy as np

import pandas as pd

from scipy import stats



'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('bmh')                    

%matplotlib inline



'''plotly Visualization'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown
'''Read the dataset from csv file'''

fifa19 = pd.read_csv('../input/fifa19/data.csv')
'''Data at a glance'''

display(fifa19.head())

display(print('Dimension of fifa19:', fifa19.shape))
"""Let's see all columns in the data"""

display(fifa19.columns.values)



'''I choose intersting columns from the dataframe for analysis'''

chosen_columns = ['Name', 

                  'Age', 

                  'Nationality', 

                  'Overall', 

                  'Potential', 

                  'Club', 

                  'Value', 

                  'Wage', 

                  'Special',

                  'Preferred Foot', 

                  'International Reputation', 

                  'Weak Foot',

                  'Skill Moves', 

                  'Body Type', 

                  'Position',

                  'Jersey Number',

                  'Height', 

                  'Weight', 

                  'Crossing', 'Finishing', 'HeadingAccuracy',

                  'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                  'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                  'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                  'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                  'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                  'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                  'GKKicking', 'GKPositioning', 'GKReflexes']
'''create new dataframe with chosen columns'''

data = pd.DataFrame(fifa19, columns = chosen_columns)

display(data.head())

display(print('Dimension of data:', data.shape))
'''Describing the data'''

data.describe()
"""Let's check missing values"""

data.isnull().sum()
'''Imputing Club and Position'''

data['Club'].fillna('No Club', inplace = True)

data['Position'].fillna('ST', inplace = True)
'''There are some Discrete and Continuous variable and will be imputed by mean'''

to_impute_by_mean = data.loc[:, ['Crossing', 'Finishing', 'HeadingAccuracy',

                                 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                                 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                                 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                                 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                                 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                                 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                                 'GKKicking', 'GKPositioning', 'GKReflexes']]

for i in to_impute_by_mean.columns:

    data[i].fillna(data[i].mean(), inplace = True)
'''These are categorical variables and will be imputed by mode.'''

to_impute_by_mode = data.loc[:, ['Body Type','International Reputation', 'Height', 'Weight', 'Preferred Foot','Jersey Number']]

for i in to_impute_by_mode.columns:

    data[i].fillna(data[i].mode()[0], inplace = True)
'''The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median.'''

to_impute_by_median = data.loc[:, ['Weak Foot', 'Skill Moves', ]]

for i in to_impute_by_median.columns:

    data[i].fillna(data[i].median(), inplace = True)
'''Columns remaining to be imputed'''

data.columns[data.isna().any()]
def general(data):

    return int(round((data[['HeadingAccuracy', 'Dribbling', 'Curve', 

                            'BallControl']].mean()).mean()))



def mental(data):

    return int(round((data[['Aggression', 'Interceptions', 'Positioning', 

                            'Vision','Composure']].mean()).mean()))



def mobility(data):

    return int(round((data[['Acceleration', 'SprintSpeed', 

                            'Agility','Reactions']].mean()).mean()))



def power(data):

    return int(round((data[['Balance', 'Jumping', 'Stamina', 

                            'Strength']].mean()).mean()))



def shooting(data):

    return int(round((data[['Finishing', 'Volleys', 'FKAccuracy', 

                            'ShotPower','LongShots', 'Penalties']].mean()).mean()))



def passing(data):

    return int(round((data[['Crossing', 'ShortPassing', 

                            'LongPassing']].mean()).mean()))



def defending(data):

    return int(round((data[['Marking', 'StandingTackle', 

                            'SlidingTackle']].mean()).mean()))



def goalkeeping(data):

    return int(round((data[['GKDiving', 'GKHandling', 'GKKicking', 

                            'GKPositioning', 'GKReflexes']].mean()).mean()))



def rating(data):

    return int(round((data[['Potential', 'Overall']].mean()).mean()))

'''Adding these categories to the data'''

data['General'] = data.apply(general, axis = 1)

data['Mental'] = data.apply(mental, axis =1)

data['Mobility'] = data.apply(mobility, axis = 1)

data['Power'] = data.apply(power, axis = 1)

data['Shooting'] = data.apply(shooting, axis = 1)

data['Passing'] = data.apply(passing, axis = 1)

data['Defending'] = data.apply(defending, axis =1)

data['Goalkeeping'] = data.apply(goalkeeping, axis = 1)

data['Rating'] = data.apply(rating, axis =1)
'''Defining a funtion for cleaning Height, Weight, Value and Wage columns'''



'''For weight'''

def extract_value_from(value):

    x = value.replace('lbs', '')

    return float(x)



'''For Height'''

def feet_to_inches(value): 

    tmp = value.split("'")

    return int(tmp[0]) * 12 + int(tmp[1]) #converting feet to inches



'''For Value and wage'''

def convert_currency(value):

    x = value.replace('€', '')

    if 'M' in x:

        x = float(x.replace('M', ''))*100000

    elif 'K' in value:

        x = float(x.replace('K', ''))*1000

    return float(x)
'''Appling the funtion to Height, Weight, Value and Wage columns'''

data['Weight'] = data['Weight'].apply(lambda x: extract_value_from(x))

data['Height'] = data['Height'].apply(lambda x: feet_to_inches(x))

data['Value'] =  data['Value'].apply(lambda x: convert_currency(x))

data['Wage'] =  data['Wage'].apply(lambda x: convert_currency(x))



display(Markdown('**Processed  Height, Weight, Value and Wage columns**'))

display(data[['Weight', 'Height', 'Value', 'Wage']].head(3))
# defining the features of players



player_features = ('Acceleration', 'Aggression', 'Agility', 

                   'Balance', 'BallControl', 'Composure', 

                   'Crossing', 'Dribbling', 'FKAccuracy', 

                   'Finishing', 'GKDiving', 'GKHandling', 

                   'GKKicking', 'GKPositioning', 'GKReflexes', 

                   'HeadingAccuracy', 'Interceptions', 'Jumping', 

                   'LongPassing', 'LongShots', 'Marking', 'Penalties')



# Top four features for every position in football



for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
# source: https://www.kaggle.com/dczerniawko/fifa19-analysis 

from math import pi



idx = 1

plt.figure(figsize=(15,45))

for position_name, features in data.groupby(data['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(9, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)



    # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

     # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1
"""Let's see all the variables we currently have with their category."""

display(data.head(2))



'''Drop the features that would not be useful anymore.'''

data.drop(columns = ['Crossing', 'Finishing', 'HeadingAccuracy',

                     'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',

                     'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed',

                     'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping',

                     'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions',

                     'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking',

                     'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',

                     'GKKicking', 'GKPositioning', 'GKReflexes', 'Overall', 'Potential'], inplace = True, axis =1)



'''Features Remaining after Dropping:'''

display(Markdown('**Features Remaining after Dropping:**'))

display(data.columns.values)
'''Create the funtion for different plots and visualization'''



'''Function to distribution plot'''

def distplot(variable, color):

    global ax

    font_size = 16

    title_size = 20

    plt.rcParams['figure.figsize'] = (18, 7)

    ax = sns.distplot(variable, color = color)

    plt.xlabel('%s' %variable.name, fontsize = font_size)

    plt.ylabel('Count of the Players', fontsize = font_size)

    plt.xticks(fontsize = font_size)

    plt.yticks(fontsize = font_size)

    plt.title('%s' %variable.name + ' Distribution of Players', fontsize = title_size)

    plt.show()

    

'''Function to count plot'''     

def countplot(variable, title,  color):

    global ax

    font_size = 14

    title_size = 20

    plt.rcParams['figure.figsize'] = (18, 8)

    ax = sns.countplot(variable, palette = color)

    plt.xlabel('%s' %variable.name, fontsize = font_size)

    plt.xticks(fontsize = font_size)

    plt.yticks(fontsize = font_size)

    plt.title(title, fontsize = title_size)

    plt.show()

    

'''Function to pie chart''' 

def piechart(variable, title, color):

    labels = ['1', '2', '3', '4', '5']

    variable = variable.value_counts()

    explode = [0.1, 0.1, 0.2, 0.5, 0.9]

    plt.rcParams['figure.figsize'] = (9, 9)

    plt.pie(variable, labels = labels, colors = color, explode = explode, shadow = True)

    plt.title(title, fontsize = 20)

    plt.legend()

    plt.show()
'''Distribution of the Age of the players'''

distplot(data['Age'], 'teal')
'''Distribution of the value of the players'''

distplot(data['Value'], 'g')
'''Distribution of the wage of the players'''

distplot(data['Wage'], 'm')
'''Distribution of Speciality Score of the players'''

distplot(data['Special'], 'b')
'''Distribution of Height of the players(In feet)'''

distplot(data['Height'], 'c')
'''Distribution of weight of the players'''

distplot(data['Weight'], 'r')
'''comparison of preferred foot over the different players'''

countplot(data['Preferred Foot'], 'Most Preferred Foot of the Player', 'Set2')
'''comparison of weak foot over the different players'''

countplot(data['Weak Foot'], 'Distribution Of Week Foot', 'Paired')
'''Pie chart to represent share of international repuatation'''

piechart(data['International Reputation'], 'International Repuatation of the Football Player', plt.cm.afmhot(np.linspace(0, 1, 5)))
'''Pie chart to represent share of Player's Skill Move'''

piechart(data['Skill Moves'], "Share of Player's Skill Move", plt.cm.viridis(np.linspace(0, 1, 5)))
countplot(data['Position'],'Comparison of positions and Player', 'copper')
'''Different nations participating in the FIFA 2019'''

data['Nationality'].value_counts().head(70).plot.bar(color = 'c', figsize = (20, 7))

plt.title('Different Nations Participating in FIFA 2019', fontsize = 30, fontweight = 20)

plt.xlabel('Name of The Country')

plt.ylabel('count')

plt.show()
'''Plot regression plot to see how Rating is correlated with numerical variables.'''

corr_num = data.loc[:, ['Value', 'Wage', 'Special', 'Jersey Number', 'Height',

                        'Weight', 'General', 'Mental', 'Mobility', 'Power', 'Shooting',

                         'Passing', 'Defending', 'Goalkeeping']]

for i in corr_num.columns:

    x = corr_num[i]

    y = data['Rating']

    

    # Creating the dataset and generating the plot

    trace = go.Scatter(

                       x = x,

                       y = y,

                       mode = 'markers',

                       marker = dict(color = 'olive'))

    

    # Layout for regression plot

    title = '{} vs Rating'.format(corr_num[i].name)

    layout = go.Layout(title = title, yaxis = dict(title = 'Rating'))

    

    fig = go.Figure(data = trace, layout = layout)

    iplot(fig)
'''Create boxplots to see the association between categorical and Rating variables.'''

corr_cat = data.loc[:, ['Preferred Foot', 'International Reputation', 'Weak Foot',

                        'Skill Moves', 'Position']]

for i in corr_cat.columns:

    trace = go.Box(

                   x = corr_cat[i],

                   y = data['Rating'],

                   marker = dict(color = 'teal'))

    layout = go.Layout(title = '{} vs Rating'. format(i), yaxis = dict(title = 'Rating'))

    fig = go.Figure(data = trace, layout = layout)

    iplot(fig)
'''Create bar plot to see the association between 'Name', 'Nationality', 'Club and Rating variables.'''

def bar_plot(x,y, xlabel, ylabel, label, color):

    global ax

    font_size = 16

    title_size = 20

    plt.rcParams['figure.figsize'] = (18, 7)

    ax = sns.barplot(x, y, palette = color)

    ax.set_xlabel(xlabel = xlabel, fontsize = font_size)

    ax.set_ylabel(ylabel = ylabel, fontsize = font_size)

    ax.set_title(label = label, fontsize = font_size)

    plt.show()
'''picking up the countries with highest number of players to compare their Rating scores'''

data['Nationality'].value_counts().head(10)
countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlans')

data_countries = data.loc[data['Nationality'].isin(countries) & data['Rating']]



'''Visualization'''

bar_plot(data_countries['Nationality'], data_countries['Rating'], 'Countries', 'Rating',

                    'Distribution of overall scores of players from different countries', 'binary_r')
'''Distribution of Wages of players from different countries'''

countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlans')

data_countries = data.loc[data['Nationality'].isin(countries) & data['Wage']]



'''Visualization'''

bar_plot(data_countries['Nationality'], data_countries['Wage'], 'Countries', 'Wage',

                    'Distribution of Wages of players from different countries', 'YlGnBu')
'''Distribution of International Reputation of players from different countries'''

countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil', 'Italy', 'Columbia', 'Japan', 'Netherlans')

data_countries = data.loc[data['Nationality'].isin(countries) & data['International Reputation']]



'''Visualization'''

bar_plot(data_countries['Nationality'], data_countries['Wage'], 'Countries', 'International Reputation',

                    'Distribution of International Reputation of players from different countries', 'Greens_r')
'''Clubs with highest number of players'''

data['Club'].value_counts().head(20)
'''Picking up the popular clubs around the globe'''

clubs = ('Arsenal','Liverpool', 'RC Celta', 'Empoli', 'Atlético Madrid', 'Manchester City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')

data_clubs = data.loc[data['Club'].isin(clubs) & data['Rating']]



'''Visualization'''

bar_plot(data_clubs['Club'], data_clubs['Rating'], 'Popular Clubs','Ratings',

                'Distribution of Rating Score in Different popular Clubs', 'inferno')
'''Top Rated players in FIFA 19'''

data[data['Rating'] > 90]['Name']
'''Distribution of International Reputation in some Popular clubs'''

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchester City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['International Reputation']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'International Reputation', data = data_club, palette = 'magma')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 14)

ax.set_ylabel(ylabel = 'Distribution of Reputation', fontsize = 14)

ax.set_title(label = 'Distribution of International Reputation in some Popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
'''Distribution of Age in some Popular clubs'''

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchester City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['Age']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.violinplot(x = 'Club', y = 'Age', data = data_club, palette = 'GnBu')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 14)

ax.set_ylabel(ylabel = 'Distribution of Age', fontsize = 14)

ax.set_title(label = 'Distribution of Age in some Popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
'''Distribution of wage in some Popular clubs'''

some_clubs = ('CD Leganés', 'Southampton', 'RC Celta', 'Empoli', 'Fortuna Düsseldorf', 'Manchester City',

             'Tottenham Hotspur', 'FC Barcelona', 'Valencia CF', 'Chelsea', 'Real Madrid')



data_club = data.loc[data['Club'].isin(some_clubs) & data['Wage']]



plt.rcParams['figure.figsize'] = (16, 8)

ax = sns.boxenplot(x = 'Club', y = 'Age', data = data_club, palette = 'cividis')

ax.set_xlabel(xlabel = 'Names of some popular Clubs', fontsize = 14)

ax.set_ylabel(ylabel = 'Distribution of Wage', fontsize = 14)

ax.set_title(label = 'Distribution of Wage in some Popular Clubs', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
''' Plotting a correlation heatmap'''

plt.rcParams['figure.figsize'] = (30, 20)

sns.heatmap(data.corr(), annot = True, cmap = 'viridis')

plt.title('Heatmap of the Dataset', fontsize = 30)

plt.show()
plt.rcParams['figure.figsize'] = (30, 20)

sns.heatmap(data[['General', 'Mental', 'Mobility', 'Power', 'Shooting',

       'Passing', 'Defending', 'Goalkeeping', 'Rating']].corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f')

plt.title('Heatmap of the Dataset', fontsize = 30)

plt.show()
'''Comparison of rating scores and value with popular clubs'''

clubs = ('Manchester City', 'FC Barcelona',  'Chelsea', 'Real Madrid')



sns.lmplot(x='Value', y='Rating', hue='Club', palette =  'tab20',

           data=data.loc[data['Club'].isin(clubs)], 

           fit_reg=False, size = 7)

plt.title('Comparison of Rating Scores and Value with Popular Clubs', fontsize = 14)

plt.show()
'''Type of Offensive Players Tends to get Paid the Most'''

sns.lmplot(x='Value', y='Rating', hue='Position', palette =  'Set1',

           data=data.loc[data['Position'].isin(['ST', 'RW', 'LW'])], 

           fit_reg=False, size = 7)

plt.title('Type of Offensive Players Tends to get Paid the Most', fontsize = 14)

plt.show()
'''Comparison of rating scores and value with Nationality'''

countries = ('England', 'Germany', 'Spain', 'Argentina', 'France', 'Brazil')



sns.lmplot(x='Value', y='Rating', hue ='Nationality', palette =  'rocket',

           data=data.loc[data['Nationality'].isin(countries)], 

           fit_reg=False, size = 7)

plt.title('Comparison of Rating Scores and Value with Nationality', fontsize = 14)

plt.show()
'''Comparison of Rating Scores and age with Preferred foot'''

plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(data['Rating'], data['Age'], hue = data['Preferred Foot'], palette = 'Greys')

plt.title('Comparison of Overall Scores and age with Preferred foot', fontsize = 20)

plt.show()
from pandas.plotting import parallel_coordinates



'''Comparition between ST and GK Positions based on Player feature'''

p = (

    data.iloc[:, 16:-1]

        .loc[data['Position'].isin(['ST', 'GK'])])



p['Position'] = data['Position']

p = p.sample(200)

plt.rcParams['figure.figsize'] = (20, 7)

parallel_coordinates(p, 'Position')
'''Comparition between FC Barcelona and FC Barcelona based on Player feature'''

p = (

    data.iloc[:, 16:-1]

        .loc[data['Club'].isin(['FC Barcelona', 'Real Madrid'])])



p['Club'] = data['Club']

p = p.sample(50)

plt.rcParams['figure.figsize'] = (20, 7)

parallel_coordinates(p, 'Club', colormap = 'RdBu')
'''Comparition between left and right foot based on Player feature'''

p = (

    data.iloc[:, 18:-3]

        .loc[data['Preferred Foot'].isin(['Left', 'Right'])])



p['Preferred Foot'] = data['Preferred Foot']

p = p.sample(200)

plt.rcParams['figure.figsize'] = (20, 7)

parallel_coordinates(p, 'Preferred Foot', colormap = 'gnuplot')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(' '.join(data['Nationality']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Country',fontsize = 30)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

plt.rcParams['figure.figsize'] = (15, 15)

wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(' '.join(data['Club']))

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Most Popular Club',fontsize = 30)

plt.show()
'''Best players per each position with their age, club, and nationality based on their overall scores'''



fifa19.iloc[fifa19.groupby(data['Position'])['Overall'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
'''Best players from each positions with their age, nationality, club based on their potential scores'''



fifa19.iloc[fifa19.groupby(data['Position'])['Potential'].idxmax()][['Position', 'Name', 'Age', 'Club', 'Nationality']]
'''Top 10 player of FIFA 19 based on rating'''



data[data['Rating'] > 90][['Name', 'Age', 'Club', 'Nationality']]
'''Top 10 youngest player from FIFA 19'''



data.sort_values('Age', ascending = True)[['Name', 'Age', 'Club', 'Nationality']].head(11)
'''Top 10 eldest player from FIFA 19'''



data.sort_values('Age', ascending = False)[['Name', 'Age', 'Club', 'Nationality']].head(11)
'''Top 10 left footed footballer'''



data[data['Preferred Foot'] == 'Left'][['Name', 'Age', 'Club', 'Nationality']].head(11)
'''Top 10 Right footed footballers'''



data[data['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(11)
'''Ploting the location based on Rating'''

rating = pd.DataFrame(data.groupby(['Nationality'])['Rating'].sum().reset_index())

count = pd.DataFrame(rating.groupby('Nationality')['Rating'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'plasma',

            locationmode = 'country names',

            locations = count['Nationality'],

            text = count['Nationality'],

            z = count['Rating'],

            reversescale=True)]



layout = go.Layout(title = 'Country vs Ratings')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
'''Ploting the location based on Shooting'''

rating = pd.DataFrame(data.groupby(['Nationality'])['Shooting'].sum().reset_index())

count = pd.DataFrame(rating.groupby('Nationality')['Shooting'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'viridis',

            locationmode = 'country names',

            locations = count['Nationality'],

            text = count['Nationality'],

            z = count['Shooting'],

            reversescale=True)]



layout = go.Layout(title = 'Country vs Shooting')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)
'''Ploting the location based on Defending'''

rating = pd.DataFrame(data.groupby(['Nationality'])['Defending'].sum().reset_index())

count = pd.DataFrame(rating.groupby('Nationality')['Defending'].sum().reset_index())



trace = [go.Choropleth(

            colorscale = 'magma',

            locationmode = 'country names',

            locations = count['Nationality'],

            text = count['Nationality'],

            z = count['Defending'],

            reversescale=True)]



layout = go.Layout(title = 'Country vs Defending')



fig = go.Figure(data = trace, layout = layout)

py.iplot(fig)