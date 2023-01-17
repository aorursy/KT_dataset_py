import numpy as np # linear algebra
import pandas as pd
import os
import matplotlib.pyplot as plt
#print(os.listdir("../input"))
import bq_helper
# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
# query and export data 
query = """SELECT * FROM `bigquery-public-data.usa_names.usa_1910_current` """
names = usa_names.query_to_pandas_safe(query)
#print(len(names))
names.sample(5)
from wordcloud import WordCloud

wc = WordCloud(background_color="white", max_words=1000)
# generate word cloud
total_freq = names.groupby('name').number.sum().to_dict()
wc.generate_from_frequencies(total_freq)

# show
plt.figure(figsize=(24,12))

plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


total_freq = names.groupby('name').number.sum()
total_freq.sort_values(ascending=False).reset_index().head(20)
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/James_Dean_in_Rebel_Without_a_Cause.jpg/440px-James_Dean_in_Rebel_Without_a_Cause.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/John_Wayne_-_still_portrait.jpg/440px-John_Wayne_-_still_portrait.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Robert_De_Niro_3_by_David_Shankbone.jpg/440px-Robert_De_Niro_3_by_David_Shankbone.jpg'></td></tr></table>"))
import seaborn as sns
gender_freq = names.groupby('gender').number.sum()
sns.barplot(x="gender", y="number",data = gender_freq.reset_index())
from matplotlib import pyplot as plt
plt.figure(figsize=(15,8))

names['decade'] = names.year // 10 *10
gender_decade_freq = names.groupby(['gender','decade']).number.sum()
sns.barplot(x="decade", y="number",hue='gender',data = gender_decade_freq.reset_index())
total_per_year_overall = names.groupby('year').number.sum()
total_per_year_men = names[names.gender == 'M'].groupby('year').number.sum()
total_per_year_women = names[names.gender == 'F'].groupby('year').number.sum()

top_20_per_year_overall = names.groupby(['year','name']).number.sum().reset_index().groupby('year').number.apply(lambda x: x.nlargest(20).sum())
top_20_per_year_men = names[names.gender == 'M'].groupby(['year','name']).number.sum().reset_index().groupby('year').number.apply(lambda x: x.nlargest(20).sum())
top_20_per_year_women = names[names.gender == 'F'].groupby(['year','name']).number.sum().reset_index().groupby('year').number.apply(lambda x: x.nlargest(20).sum())
name_homegenuity_per_year = pd.concat([top_20_per_year_overall/total_per_year_overall,top_20_per_year_men/total_per_year_men,top_20_per_year_women/total_per_year_women],axis=1)
name_homegenuity_per_year.columns = ['overall','men','women']
name_homegenuity_per_year.plot.line(figsize=(15,8))
all_usa_names = names.groupby(['year','name','gender']).number.sum().reset_index()
unique_per_year = all_usa_names.groupby(['year','gender']).name.nunique()
unique_per_year.unstack().plot.line(figsize=(15,8))
names['name_length'] = names.name.str.len()
by_decade_and_length = names.groupby(['decade','name_length']).number.sum().unstack()
by_decade_and_length
plt.figure(figsize=(15,8))
sns.boxplot(x="decade", y="name_length", data=by_decade_and_length.stack().reset_index())
names_by_year_counts = names.groupby(['name','year']).number.sum().reset_index()
diffs_by_year = names_by_year_counts.set_index(['year','name']).unstack().pct_change().stack().reset_index()
spikes = diffs_by_year[(diffs_by_year.number>5) & (diffs_by_year.year > 1930)]
spikes.sample(10)
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://www.etonline.com/sites/default/files/styles/max_1280x720/public/images/2018-10/1280_gwyneth_paltrow.jpg?itok=rcm9rlcK'></td><td><img src='https://cdn-images-1.medium.com/max/2000/1*cRmipwCGD5drJv6tQNNwBQ.jpeg'></td></tr></table>"))
spikes[spikes.year == 1998]
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='http://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/1966.png&w=350&h=254'></td><td><img src='https://m.media-amazon.com/images/M/MV5BMTM0Nzc5ODkyM15BMl5BanBnXkFtZTcwOTczMTgxNw@@._V1_UY317_CR0,0,214,317_AL_.jpg'></td></tr></table>"))
spikes[spikes.year == 2004]
names[(names.state=='MI') & (names.name.str.startswith('Taysh'))]
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Madonna_Rebel_Heart_Tour_2015_-_Stockholm_%2823051472299%29_%28cropped_2%29.jpg/440px-Madonna_Rebel_Heart_Tour_2015_-_Stockholm_%2823051472299%29_%28cropped_2%29.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/a/a7/Whitney_Houston_Welcome_Home_Heroes_1_cropped.jpg'></td></tr></table>"))
spikes[spikes.year == 1985]
recent_names = names[names.year >=2015]
state_name_table = recent_names[['state','name','number']].groupby(['state','name']).sum().unstack().fillna(0)
state_name_table
len(state_name_table.columns)
from sklearn.metrics.pairwise import cosine_similarity
state_sim_matrix = pd.DataFrame(cosine_similarity(state_name_table.values))
state_sim_matrix.index = state_name_table.index
state_sim_matrix.columns = state_name_table.index
state_sim_matrix = state_sim_matrix.stack()
state_sim_matrix.index.names = ['state1','state2']
state_sim_matrix = state_sim_matrix.reset_index()
state_sim_matrix = state_sim_matrix[state_sim_matrix.state1 < state_sim_matrix.state2] # to not have same pairs from differnet sides
state_sim_matrix.sort_values(0,ascending=False).head(20)
state_sim_matrix.sort_values(0,ascending=False).tail(5)
from sklearn.cluster import AgglomerativeClustering
clustering =AgglomerativeClustering(n_clusters=8, affinity='cosine',linkage = 'average')
clustering.fit(state_name_table)
state_clusters=pd.Series( clustering.labels_)
state_clusters.index = state_name_table.index
state_clusters.reset_index().sort_values(0)
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon

# Abbreviations from here https://gist.github.com/rogerallen/1583593
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
inv_us_state_abbrev = {v: k for k, v in us_state_abbrev.items()}

plt.figure(figsize=(18,10))
# Code from here https://stackoverflow.com/questions/7586384/color-states-with-pythons-matplotlib-basemap
# create the map
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

# load the shapefile, use the name 'states'
map.readshapefile('../input/st99_d00', name='states', drawbounds=True)

# collect the state names from the shapefile attributes so we can
# look up the shape obect for a state by it's name
state_names = []
for shape_dict in map.states_info:
    state_names.append(shape_dict['NAME'])

ax = plt.gca() # get current axes instance

cluster_color_mapping = {0:'red',1:'blue',2:'black',3:'yellow',4:'brown',5:'pink',6:'green',7:'purple'}

for state, cluster in state_clusters.to_dict().items():
    if state not in inv_us_state_abbrev:
        continue
    seg = map.states[state_names.index(inv_us_state_abbrev[state])]
    poly = Polygon(seg, facecolor=cluster_color_mapping[cluster],edgecolor=cluster_color_mapping[cluster])
    ax.add_patch(poly)

plt.show()
names_a = names.copy()
names_a['ends_with_a'] = names.name.str.strip().str[-1] =='a'
names_a[names_a.ends_with_a].sample(3)
names_a[names_a.ends_with_a].groupby('gender').number.sum().plot.bar()
names_a[names_a.gender == "F"].groupby('decade')[['number','ends_with_a']].apply(lambda x: sum(x['number']*x['ends_with_a'])/sum(x['number'])).plot.line()
plt.figure(figsize=(15,6))
sns.countplot(names[names.gender =='F'].name.str.strip().str[-1])
plt.figure(figsize=(15,6))
sns.countplot(names[names.gender =='M'].name.str.strip().str[-1])
names_n = names.copy()
names_n['ends_with_n'] = names.name.str.strip().str[-1] =='n'
names_n[names_a.gender == "M"].groupby('decade')[['number','ends_with_n']].apply(lambda x: sum(x['number']*x['ends_with_n'])/sum(x['number'])).plot.line()

names['prefix_2'] = names.name.str.strip().str[0:2]
prefixes_by_gender = names.groupby(['prefix_2','gender']).number.sum().unstack()
prefixes_by_gender['total'] = prefixes_by_gender.sum(axis=1)
prefixes_by_gender.sort_values('total',ascending=False)[['M','F']].head(20).plot.bar(figsize=(15,6))
prefixes = names.groupby('prefix_2').number.sum().nlargest(20).index.to_series().values
names[names.prefix_2.isin(prefixes)].groupby(['prefix_2','name']).number.sum().reset_index().sort_values('number',ascending=False).groupby('prefix_2').apply(lambda x :pd.Series(x.head(5).name.values))#.set_index(['prefix_2','name']).unstack()

male_percentage = names.groupby(['name','gender']).number.sum().unstack().fillna(0).apply(lambda x : x[1]/(x[0]+x[1]),axis=1)

frequent_names_df = total_freq.reset_index()
frequent_names_df = frequent_names_df[frequent_names_df.number > 3000]
frequent_names_df['prefix'] = frequent_names_df.name.str[0:2]
pairs = frequent_names_df.merge(frequent_names_df, on='prefix')
pairs['male_x'] = pairs.name_x.map(male_percentage)
pairs['male_y'] = pairs.name_y.map(male_percentage)

#pairs = pairs[pairs.name_x < pairs.name_y]
print(len(pairs))
pairs.sample(5)
def is_variation(str_1,str_2):
    if str_2 == str_1 + 'a':
        return True
    elif str_2 == str_1 + 'e':
        return True
    elif str_2 == str_1 + str_1[-1] +'a':
        return True
    elif str_2 == str_1 + str_1[-1] +'e':
        return True
    else:
        return False
         
variations = pairs[pairs[['name_x','name_y']].apply(lambda x: is_variation(x[0],x[1]),axis=1)]
variations = variations[(variations.male_x - variations.male_y) > 0.7]
variations['total'] = variations[['number_x','number_y']].sum(axis=1)
variations.sort_values('total',ascending=False).head(20)[['name_x','name_y']]
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg/1280px-Robert_Downey_Jr_2014_Comic_Con_%28cropped%29.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/0/0d/Roberta_Flack.jpg'></td></tr></table>"))
gender_pivoted = names.groupby(['year','name','gender']).number.sum().unstack().reset_index().fillna(0)
gender_pivoted['masc_score'] = gender_pivoted[['F','M']].apply(lambda x : x[1]/(x[0]+x[1]),axis=1)
gender_pivoted['unisex_score'] = gender_pivoted['masc_score'].apply(lambda x : min(x,1-x))
gender_pivoted['total'] = gender_pivoted[['F','M']].sum(axis=1)
gender_pivoted.sample(10)
gender_pivoted[gender_pivoted.year==2017].sort_values('unisex_score',ascending=False).head(20)
gender_pivoted[(gender_pivoted.year==2017) & (gender_pivoted.total > 500)].sort_values('unisex_score',ascending=False).head(20)
gender_pivoted[gender_pivoted.name.isin(['Charlie','Skyler','Justice'])][['year','name','unisex_score']].set_index(['year','name']).unisex_score.unstack().plot.line(figsize=(15,8))
def weighted_unisex_score(x, unisex_col, total_col):
    return sum(x[unisex_col]*x[total_col])/sum(x[total_col])
unisex_score_by_year = gender_pivoted.groupby('year').apply(weighted_unisex_score,'unisex_score','total')
unisex_score_by_year.plot.line(figsize=(15,6))
female_to_unisex = set(gender_pivoted[(gender_pivoted.year < 1950) & (gender_pivoted.masc_score < 0.01)].name) & \
set(gender_pivoted[(gender_pivoted.year > 2015) & (gender_pivoted.unisex_score > 0.3)].name)
female_to_unisex = total_freq.loc[female_to_unisex].sort_values()
female_to_unisex.reset_index()
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://upload.wikimedia.org/wikipedia/commons/3/33/Joan_Crawford_1946_by_Paul_Hesse.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/e/ea/Mrs_Kennedy_in_the_Diplomatic_Reception_Room_cropped.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/c/cd/Robin_Wright_Cannes_2017.jpg'></td></tr></table>"))
gender_pivoted[gender_pivoted.name.isin(female_to_unisex.tail(5).index)][['year','name','masc_score']].set_index(['year','name']).masc_score.unstack().plot.line(figsize=(20,10),subplots=True,layout=(3,2))

male_to_unisex = set(gender_pivoted[(gender_pivoted.year < 1950) & (gender_pivoted.masc_score > 0.99)].name) & \
set(gender_pivoted[(gender_pivoted.year > 2015) & (gender_pivoted.unisex_score > 0.3)].name)
male_to_unisex = total_freq.loc[male_to_unisex].sort_values()
male_to_unisex.reset_index()
from IPython.display import HTML, display
display(HTML("<table><tr><td><img src='https://upload.wikimedia.org/wikipedia/commons/a/a2/Blake_Shelton_July_2017_%28cropped%29.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/8/8a/Hayden-cfda2010-0004%281%29_%28cropped%29.jpg'></td><td><img src='https://upload.wikimedia.org/wikipedia/commons/0/0b/Caseyaffleck2018_%28cropped%29.png'></td></tr></table>"))
gender_pivoted[gender_pivoted.name.isin(male_to_unisex.tail(6).index)][['year','name','masc_score']].set_index(['year','name']).masc_score.unstack().plot.line(figsize=(20,10),subplots=True,layout=(3,2))

unisex_var = gender_pivoted.groupby('name').agg({'unisex_score':['mean','std']})
unisex_var.columns = unisex_var.columns.droplevel()
unisex_var = unisex_var[unisex_var['mean']> 0.05]
unisex_var['total'] = total_freq
unisex_var.sample(5)
unisex_var[unisex_var['std']> 0.12].sort_values('total',ascending=False).head(20)
gender_pivoted[gender_pivoted.name.isin(['Willie','Taylor','Angel','Dana','Gail','Sydney'])][['year','name','masc_score']].set_index(['year','name']).masc_score.unstack().plot.line(figsize=(20,10),subplots=True,layout=(4,2))



