### IMPORT USEFUL PACKAGES ###
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

from collections import Counter
from scipy.stats.stats import pearsonr
from string import ascii_letters

# Some helper functions to make our plots cleaner with Plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
init_notebook_mode(connected=True)

pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')
%matplotlib inline

def correction(x):
    '''
    Columns value corrections
    '''
    if type(x)==str:
        x=x.replace('$','')
        x=x.replace(',','')
        x=float(x)    
    return (x)

def correction2(x):
    '''
    Columns value corrections
    '''
    if type(x)==str:
        x=x.replace('%','')
        x=float(x)/100.0
    return (x)

def to_int(x):
    '''
    Columns value corrections
    '''
    if x=='f':
        x=x.replace('f','0')
    elif x=='t':
        x=x.replace('t','1')
    else:
        x= '0'
    return int(x)

def changeTime(x):
    '''
    change host_response_time columns from string into numerical.
    '''
    if x == 'within an hour':
        x='1'
    elif x == 'within a few hours':
        x='4'
    elif x == 'within a day':
        x='24'
    elif x == 'a few days or more':
        x='48'
    else:
        x='96'
        
    return x


def changeStr(x):
    '''
    change back the host_response_time from the numerical into strings
    '''
    if x == 1:
        x='within an hour'
    elif x == 4:
        x='within a few hours'
    elif x == 24:
        x='within a day'
    elif x == 48:
        x= 'a few days or more'
    elif x == 96:
        x= 'Not Response'
        
    return x

def createAmenities(x):
    '''
    Convert the Amenities column into more analytical words
    '''
    val = x.replace('{','').replace('}','').replace('"','').replace(' ','_').replace(',',' ')
    val = val.split()
    return val


def rangeScore(x):
    '''
    Set the bins for the score-range.
    '''
    value = ''
    if (x>= 0 and x < 10):
        value = '0-10'
    elif (x>= 10 and x < 20):
        value = '10-20'
    elif (x>= 20 and x < 30):
        value = '20-30'
    elif (x>= 30.0 and x < 40.0):
        value = '30-40'
    elif (x>= 40 and x < 50):
        value = '40-50'
    elif (x>= 50 and x < 60):
        value = '50-60'
    elif (x>= 60 and x < 70):
        value = '60-70'        
    elif (x>= 70 and x < 80):
        value = '70-80'
    elif (x>= 80 and x < 90):
        value = '80-90'
    elif (x>= 90 and x < 100):
        value = '90-100'
    elif x>= 100:
        value = '100+'
        
    return value


'''
    ### VIZ FUNCTIONS ###
    this functions actually using the functions from
    https://www.kaggle.com/andresionek/what-makes-a-kaggler-valuable/notebook
'''

def gen_xaxis(title):
    """
    Creates the X Axis layout and title
    """
    xaxis = dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return xaxis


def gen_yaxis(title):
    """
    Creates the Y Axis layout and title
    """
    yaxis=dict(
            title=title,
            titlefont=dict(
                color='#AAAAAA'
            ),
            showgrid=False,
            color='#AAAAAA',
            )
    return yaxis


def gen_layout(charttitle, xtitle, ytitle, lmarg, h, annotations=None):  
    """
    Creates whole layout, with both axis, annotations, size and margin
    """
    return go.Layout(title=charttitle, 
                     height=h, 
                     width=800,
                     showlegend=False,
                     xaxis=gen_xaxis(xtitle), 
                     yaxis=gen_yaxis(ytitle),
                     annotations = annotations,
                     margin=dict(l=lmarg),
                    )


def gen_bars(data, color, orient):
    """
    Generates the bars for plotting, with their color and orient
    """
    bars = []
    for label, label_df in data.groupby(color):
        if orient == 'h':
            label_df = label_df.sort_values(by='x', ascending=True)
        if label == 'a':
            label = 'lightgray'
        bars.append(go.Bar(x=label_df.x,
                           y=label_df.y,
                           name=label,
                           marker={'color': label},
                           orientation = orient
                          )
                   )
    return bars


def gen_annotations(annot):
    """
    Generates annotations to insert in the chart
    """
    if annot is None:
        return []
    
    annotations = []
    # Adding labels
    for d in annot:
        annotations.append(dict(xref='paper', x=d['x'], y=d['y'],
                           xanchor='left', yanchor='bottom',
                           text= d['text'],
                           font=dict(size=13,
                           color=d['color']),
                           showarrow=False))
    return annotations


def generate_barplot(text, annot_dict, orient='v', lmarg=120, h=400):
    """
    Generate the barplot with all data, using previous helper functions
    """
    layout = gen_layout(text[0], text[1], text[2], lmarg, h, gen_annotations(annot_dict))
    fig = go.Figure(data=gen_bars(barplot, 'color', orient=orient), layout=layout)
    return iplot(fig)
### import files ###

csvs = glob.glob('../input/*.csv')
base= pd.read_csv(csvs[0])
listings_df= base.copy()
listings_df.head(2)

    ### create a new metrics ###

listings_df['new_score_reviews2'] = listings_df['reviews_per_month'] * listings_df['review_scores_rating'] / 10
listings_df['new_score_reviews2'].fillna(0, inplace = True)
'''
the definition and print the value.
'''
top90flag = listings_df['new_score_reviews2'].quantile(0.9)
upto25flag = listings_df['new_score_reviews2'].quantile(0.25)

listings_df['top90'] = listings_df.new_score_reviews2 >= top90flag
listings_df['upto25'] = listings_df.new_score_reviews2 <= upto25flag

print('The boundaries of top performer listings:',top90flag)
print('The boundaries of low performer listings:',upto25flag)
### Create a table for the visualization essentials ###
### Generate score bins, creating new tables for the class colors, and count distributions of each bins. ###
# a columns of bins.
listings_df['score_ranges'] = listings_df['new_score_reviews2'].apply(rangeScore)

# table coloring purpose.
top90 = listings_df.groupby('score_ranges', as_index = False)['top90'].max(key = 'count').rename(columns={'score_ranges':'Score'})
upto25 = listings_df.groupby('score_ranges', as_index = False)['upto25'].max(key = 'count').rename(columns={'score_ranges':'Score'})

# count distributions of score bins.
barplot = listings_df[['id','new_score_reviews2']]
barplot['Qty'] = barplot['new_score_reviews2'].apply(rangeScore)
barplot = barplot.Qty.value_counts(sort=True).to_frame().reset_index()
barplot = barplot.rename(columns={'index': 'Score'})

# merging color flag.
barplot = barplot.merge(top90, on = 'Score')
barplot = barplot.merge(upto25)
# creating color for the vis.
barplot['color'] = barplot.top90.apply(lambda x: 'mediumaquamarine' if x else 'lightgray')
# manually change the color of the first index become crimson, to indicate the class of low performer listings.
barplot.iloc[0,4] = 'crimson'

# change Score column and Qty column into x and y for the vis purpose.
barplot = barplot.rename(columns={'Score':'x','Qty':'y'})

# Some of the annotations for the vis.
title_text = ['<b>Comparison Listings Performance between Top Performer and Low Performer</b>', 'Reviews per Month x Review Score Ratings / 10', 'Quantity of Listings']
annotations = [{'x': 0.03, 'y': 1900, 'text': 'Low Performer Had Score Up to 25 Percentile','color': 'gray'},
              {'x': 0.39, 'y': 300, 'text': 'Top Performer Had Score above 90 Percentile','color': 'mediumaquamarine'}]

generate_barplot(title_text, annotations)
#some useless columns: url, and unique value all of the rows. 
unique_value_columns=[]
url_columns=[]

for i in listings_df.columns:
    
    if len((listings_df[i]).unique())==1:
        print ('a un-used column because same value:', i, (listings_df[i]).unique())
        unique_value_columns=unique_value_columns+[i]
    if 'url' in i:
        url_columns=url_columns+[i]
        
# url columns.
# print ('\n''url columns:\n\n', url_columns)
# unique value columns.
# print ('\n''unique value columns:\n\n', unique_value_columns)

# Drop it.
listings_df = listings_df.drop(url_columns+unique_value_columns, axis = 1)
# Change the string of boolean (t / f) into int of boolean (1/0)
for i in listings_df.columns:
    
    if set(listings_df[i])=={'t','f'}:
        listings_df[i]=listings_df[i].apply(to_int)
        
    elif set(listings_df[i]) == {'t','f',np.nan}:
        listings_df[i]=listings_df[i].apply(to_int)
        
# Dollar corrections.
listings_df['price']=listings_df['price'].map(lambda x: correction(x))
listings_df['weekly_price'] = listings_df['weekly_price'].map(lambda x: correction(x))
listings_df['monthly_price'] = listings_df['monthly_price'].map(lambda x: correction(x))
listings_df['security_deposit'] = listings_df['security_deposit'].map(lambda x: correction(x))
listings_df['cleaning_fee'] = listings_df['cleaning_fee'].map(lambda x: correction(x))
listings_df['extra_people'] = listings_df['extra_people'].map(lambda x: correction(x))

# Change the rate percentage.
listings_df['host_response_rate'] = listings_df['host_response_rate'].fillna('0%').apply(correction2)
listings_df['host_acceptance_rate'] = listings_df['host_acceptance_rate'].fillna('0%').apply(correction2)

# Change time indicators
listings_df['host_response_time'] = listings_df['host_response_time'].apply(changeTime).astype(int)

# Amenities change into reproduceable column.
listings_df['amenities']= base['amenities']
listings_df['array_amenities'] = listings_df['amenities'].apply(lambda x: createAmenities(x))
listings_df['len_amenities'] = listings_df['amenities'].apply(lambda x: len(createAmenities(x)))

# filling some null value.
listings_df['security_deposit'].fillna(0, inplace = True)
listings_df['cleaning_fee'].fillna(0, inplace = True)

# Pick onlly the relevant columns
# relevant_columns = list(listings_df.columns)

# Some irrelevant columns personally
irrelevant_colmuns = ['id','host_id','host_listings_count','host_total_listings_count','latitude','longitude','is_location_exact','square_feet','price','weekly_price','monthly_price','minimum_nights','maximum_nights','availability_30','availability_60','availability_90','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','calculated_host_listings_count','reviews_per_month','require_guest_profile_picture','require_guest_phone_verification']
relevant_df = listings_df.drop(irrelevant_colmuns, axis = 1)

# pick only numerical columns
# listings_df= listings_df.select_dtypes(np.number)
sns.set(style="white")

# Compute the correlation matrix 'top90', 'upto25', .drop(['scrape_id', 'license'], axis=1)
corr = relevant_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
A=relevant_df.corr().unstack().sort_values(ascending=False)
print('The correlation of the new_score_reviews against all:', A['new_score_reviews2'][1:-1])
### defide them into 2 dataframe class ###

top_listings = listings_df[listings_df['new_score_reviews2'] >= np.percentile(listings_df['new_score_reviews2'],90)]
low_listings = listings_df[listings_df['new_score_reviews2'] <= np.percentile(listings_df['new_score_reviews2'],25)]

### Host Acceptance Rate DataFrame ###

hostAR_top_performer = pd.DataFrame(top_listings['host_acceptance_rate'].reset_index(drop = True))
hostAR_top_performer['status'] = 'Top Performer'

hostAR_low_performer = pd.DataFrame(low_listings['host_acceptance_rate'].reset_index(drop = True))
hostAR_low_performer['status'] = 'Low Performer'

hostAR = hostAR_low_performer.append(hostAR_top_performer).sample(frac=1)

### Identity verified.

identify_verified_top = pd.DataFrame(top_listings['host_identity_verified'].reset_index(drop = True))
identify_verified_top['status'] = 'Top Performer'

identify_verified_low = pd.DataFrame(low_listings['host_identity_verified'].reset_index(drop = True))
identify_verified_low['status'] = 'Low Performer'

identify_verified = identify_verified_low.append(identify_verified_top).sample(frac=1)

### Host is Superhost DataFrame

superhost_top_performer = pd.DataFrame(top_listings['host_is_superhost'].reset_index(drop = True))
superhost_top_performer['status'] = 'Top Performer'

superhost_low_performer = pd.DataFrame(low_listings['host_is_superhost'].reset_index(drop = True))
superhost_low_performer['status'] = 'Low Performer'

superhost = superhost_top_performer.append(superhost_low_performer).sample(frac=1)

### instant bookable

instantBookable_top_performer = pd.DataFrame(top_listings['instant_bookable'].reset_index(drop = True))
instantBookable_top_performer['status'] = 'Top Performer'

instantBookable_low_performer = pd.DataFrame(low_listings['instant_bookable'].reset_index(drop = True))
instantBookable_low_performer['status'] = 'Low Performer'

host_bookable = instantBookable_low_performer.append(instantBookable_top_performer).sample(frac=1)

### Host response Rate.

top_listings['host_response_time_str'] = top_listings['host_response_time'].apply(changeStr)
low_listings['host_response_time_str'] = low_listings['host_response_time'].apply(changeStr)

host_response_top = pd.DataFrame(top_listings['host_response_time_str'].reset_index(drop = True))
host_response_top['status'] = 'Top Performer'

host_response_low = pd.DataFrame(low_listings['host_response_time_str'].reset_index(drop = True))
host_response_low['status'] = 'Low Performer'

host_response = host_response_low.append(host_response_top).sample(frac=1)

### **Host Response time.**

host_responserate_top_performer = pd.DataFrame(top_listings['host_response_rate'].reset_index(drop = True))
host_responserate_top_performer['status'] = 'Top Performer'

host_responserate_low_performer = pd.DataFrame(low_listings['host_response_rate'].reset_index(drop = True))
host_responserate_low_performer['status'] = 'Low Performer'

host_responserate = host_responserate_low_performer.append(host_responserate_top_performer).sample(frac=1)
# ---
percentage_low = list(host_responserate[host_responserate['status'] == 'Low Performer']['host_response_rate'].unique())
percentage_low.sort(reverse = True)
percentage_top = list(host_responserate[host_responserate['status'] == 'Top Performer']['host_response_rate'].unique())
percentage_top.sort(reverse = True)
host_responserate_fig = host_responserate[host_responserate['host_response_rate'].isin([1.0, 0.99, 0.96, 0.95, 0.94, 0.0])]

### collecting data into one array ###
data= []
data.append(hostAR)
data.append(identify_verified)
data.append(superhost)
data.append(host_bookable)
data.append(host_response)
data.append(host_responserate_fig)
### Vis ###

fig, ax = plt.subplots(figsize=(15,20), nrows=3, ncols=2)
x_data= ['host_acceptance_rate','host_identity_verified','host_is_superhost','instant_bookable','host_response_time_str','host_response_rate']
title= ['Acceptance Rate','# of identified listings','Superhost Listings', 'Instant Bookable Feature', 'How long hosts will respond?' , 'Responses Probability']

x_axis= ['Percentage','Activate/Not','True/Not','Activate/Not', 'Respond Time', 'Respond Rate']
y_axis= ['Count']
cnt=0

for x in range(3):
    for y in range(2):
        
        ax[x][y].set_title(title[cnt], fontsize=12)
        sns.countplot(x=x_data[cnt], hue='status', data=data[cnt], palette='GnBu_d', orient='h', ax=ax[x][y])
        plt.setp(ax[x][y].get_legend().get_texts(), fontsize='8') # for legend text
        plt.setp(ax[x][y].get_legend().get_title(), fontsize='10') # for legend title
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        plt.xlabel(x_axis[cnt], fontsize=13)
        p=plt.ylabel(y_axis[0], fontsize=13)
        
        cnt+=1
### table preparations

viz1 = top_listings[['len_amenities','id']].groupby(['len_amenities']).count().sort_values(by='id', ascending=False).head(7)
viz1['percentage'] = viz1['id'] / viz1['id'].sum()
viz2 = low_listings[['len_amenities','id']].groupby(['len_amenities']).count().sort_values(by='id', ascending=False).head(7)
viz2['percentage'] = viz2['id'] / viz2['id'].sum()

ids = [room for room in top_listings['id']]
viz_base1 = listings_df[listings_df['id'].isin(ids)].reset_index(drop = True)
arr = []
for row in range(viz_base1.shape[0]):
    arr = arr+viz_base1['array_amenities'][row]
    
test3 = Counter(arr)
viz3 = test3.most_common(10)
viz3 = pd.DataFrame(viz3).rename(columns={0:'name',1:'frequency'})
viz3['percentage'] = viz3['frequency'] / viz_base1.shape[0]
viz3.index = viz3['name']
viz3 = viz3.drop('name', axis = 1)

ids = [room for room in low_listings['id']]
viz_base2 = listings_df[listings_df['id'].isin(ids)].reset_index(drop = True)
arr = []
for row in range(viz_base2.shape[0]):
    arr = arr+viz_base2['array_amenities'][row]
    
test4 = Counter(arr)

viz4 = test4.most_common(10)
viz4 = pd.DataFrame(viz4).rename(columns={0:'name',1:'frequency'})
viz4['percentage'] = viz4['frequency'] / viz_base2.shape[0]
viz4.index = viz4['name']
viz4 = viz4.drop('name', axis = 1)

### visualization

fig, ax= plt.subplots(figsize= (30,17), nrows=2, ncols= 2)
sns.barplot(y=viz2.index.astype(str) + '_types',x=viz2['id'], ax=ax[0][0]).set_title('How much amenities does top 10% provide in every service', fontsize=11)
sns.barplot(y=viz1.index.astype(str) + '_types',x=viz1['id'], ax=ax[0][1]).set_title('How much amenities does Low Performer provide in every service', fontsize=11)
sns.barplot(y=viz4.index,x=viz4["percentage"], ax=ax[1][0]).set_title('The top miscellaneous provided by hosts (Low-Performancer Listings)', fontsize=11)
sns.barplot(y=viz3.index,x=viz3["percentage"], ax=ax[1][1]).set_title('The top miscellaneous provided by hosts (Top-Performer Listings)', fontsize=11)