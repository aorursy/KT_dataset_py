import pandas as pd
import seaborn as sns 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Layout
init_notebook_mode(connected=True)
%matplotlib inline
crime_df = pd.read_csv('../input/crime.csv')
crime_df.head()
crime_df.drop(['Rape\r(legacy\rdefinition)2', 'Arson3', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'], inplace = True, axis = 1)
# According to the count result, I found that city(9292<9302) seems to have na values. Therefore we need to remove City's NAs
crime_df = crime_df[pd.notnull(crime_df['City'])]
print(crime_df.count())
# And then fill some variables' NAs with zero
crime_df[['Violent\rcrime', 'Rape\r(revised\rdefinition)1']] = crime_df[['Violent\rcrime', 'Rape\r(revised\rdefinition)1']].fillna(0)
crime_df[['Aggravated\rassault', 'Property\rcrime', 'Burglary', 'Larceny-\rtheft']] = crime_df[['Aggravated\rassault', 'Property\rcrime', 'Burglary', 'Larceny-\rtheft']].fillna(0)
# And then fill Population with average population
crime_df['Population'] = crime_df['Population'].fillna(crime_df['Population'].mean())
crime_df.count()
crime_df['Total Crime'] = crime_df['Violent\rcrime'] + crime_df['Murder and\rnonnegligent\rmanslaughter'] + crime_df['Rape\r(revised\rdefinition)1'] + crime_df['Robbery'] + crime_df['Aggravated\rassault'] + crime_df['Property\rcrime'] + crime_df['Burglary'] + crime_df['Larceny-\rtheft'] + crime_df['Motor\rvehicle\rtheft']
crime_group_df = crime_df.groupby(['State'], as_index=False).sum()[['State', 'Population', 'Total Crime']]
crime_group_df['Crime Rate'] = crime_group_df['Total Crime'] / crime_group_df['Population']
crime_group_df.head()
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}
us_state_code = pd.DataFrame.from_dict(us_state_abbrev, orient='index')
us_state_code = us_state_code.reset_index()
us_state_code = us_state_code.rename(columns={'index': 'State', 0: 'Code'})
us_state_code['State'] = us_state_code['State'].str.upper()
us_state_crime_df = crime_group_df.merge(us_state_code, on='State', how='inner')
us_state_crime_df.head()
trace = go.Scatter(
    x=us_state_crime_df['Population'],
    y=us_state_crime_df['Crime Rate'],
    mode='markers',
    text=us_state_crime_df['State'],
    marker=dict(
        size=12,               
        color=us_state_crime_df['Crime Rate'],
        colorscale='Viridis',  
        opacity=0.8
    )
)

data = [trace]
layout = go.Layout(
    showlegend=False,
    title='The USA Crime',
    xaxis= dict(
        title= 'Population',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Crime Rate',
        ticklen= 5,
        gridwidth= 2,
    ),
    width=800,
    height=600,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
data = dict(type = 'choropleth', 
            colorscale = 'Jet', 
            locations = us_state_crime_df['Code'], 
            z = us_state_crime_df['Crime Rate'], 
            locationmode = 'USA-states', 
            text = us_state_crime_df['State'], 
            marker = dict(line = dict(color = 'rgb(255, 255,255)', width = 2)),
            colorbar = {'title':"Crime Rate"}
           )

layout = dict(title = 'The USA Crime Rate',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )

choromap = go.Figure(data = [data], layout=layout)

iplot(choromap)
