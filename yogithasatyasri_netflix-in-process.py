import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import pandas as pd 



df = pd.read_csv("../input/netflix-shows/netflix_titles_nov_2019.csv")



## add new features in the dataset

df["date_added"] = pd.to_datetime(df['date_added'])

df['year_added'] = df['date_added'].dt.year

df['month_added'] = df['date_added'].dt.month



df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

df.head()
col = "type"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))

layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
d1 = df[df["type"] == "TV Show"]

d2 = df[df["type"] == "Movie"]



col = "year_added"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "release_year"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = 'month_added'

vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="In which month, the conent is added the most?", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
small = df.sort_values("release_year", ascending = True)

small = small[small['duration'] != ""]

small[['title', "release_year"]][:15]
country_codes = {'afghanistan': 'AFG',

 'albania': 'ALB',

 'algeria': 'DZA',

 'american samoa': 'ASM',

 'andorra': 'AND',

 'angola': 'AGO',

 'anguilla': 'AIA',

 'antigua and barbuda': 'ATG',

 'argentina': 'ARG',

 'armenia': 'ARM',

 'aruba': 'ABW',

 'australia': 'AUS',

 'austria': 'AUT',

 'azerbaijan': 'AZE',

 'bahamas': 'BHM',

 'bahrain': 'BHR',

 'bangladesh': 'BGD',

 'barbados': 'BRB',

 'belarus': 'BLR',

 'belgium': 'BEL',

 'belize': 'BLZ',

 'benin': 'BEN',

 'bermuda': 'BMU',

 'bhutan': 'BTN',

 'bolivia': 'BOL',

 'bosnia and herzegovina': 'BIH',

 'botswana': 'BWA',

 'brazil': 'BRA',

 'british virgin islands': 'VGB',

 'brunei': 'BRN',

 'bulgaria': 'BGR',

 'burkina faso': 'BFA',

 'burma': 'MMR',

 'burundi': 'BDI',

 'cabo verde': 'CPV',

 'cambodia': 'KHM',

 'cameroon': 'CMR',

 'canada': 'CAN',

 'cayman islands': 'CYM',

 'central african republic': 'CAF',

 'chad': 'TCD',

 'chile': 'CHL',

 'china': 'CHN',

 'colombia': 'COL',

 'comoros': 'COM',

 'congo democratic': 'COD',

 'Congo republic': 'COG',

 'cook islands': 'COK',

 'costa rica': 'CRI',

 "cote d'ivoire": 'CIV',

 'croatia': 'HRV',

 'cuba': 'CUB',

 'curacao': 'CUW',

 'cyprus': 'CYP',

 'czech republic': 'CZE',

 'denmark': 'DNK',

 'djibouti': 'DJI',

 'dominica': 'DMA',

 'dominican republic': 'DOM',

 'ecuador': 'ECU',

 'egypt': 'EGY',

 'el salvador': 'SLV',

 'equatorial guinea': 'GNQ',

 'eritrea': 'ERI',

 'estonia': 'EST',

 'ethiopia': 'ETH',

 'falkland islands': 'FLK',

 'faroe islands': 'FRO',

 'fiji': 'FJI',

 'finland': 'FIN',

 'france': 'FRA',

 'french polynesia': 'PYF',

 'gabon': 'GAB',

 'gambia, the': 'GMB',

 'georgia': 'GEO',

 'germany': 'DEU',

 'ghana': 'GHA',

 'gibraltar': 'GIB',

 'greece': 'GRC',

 'greenland': 'GRL',

 'grenada': 'GRD',

 'guam': 'GUM',

 'guatemala': 'GTM',

 'guernsey': 'GGY',

 'guinea-bissau': 'GNB',

 'guinea': 'GIN',

 'guyana': 'GUY',

 'haiti': 'HTI',

 'honduras': 'HND',

 'hong kong': 'HKG',

 'hungary': 'HUN',

 'iceland': 'ISL',

 'india': 'IND',

 'indonesia': 'IDN',

 'iran': 'IRN',

 'iraq': 'IRQ',

 'ireland': 'IRL',

 'isle of man': 'IMN',

 'israel': 'ISR',

 'italy': 'ITA',

 'jamaica': 'JAM',

 'japan': 'JPN',

 'jersey': 'JEY',

 'jordan': 'JOR',

 'kazakhstan': 'KAZ',

 'kenya': 'KEN',

 'kiribati': 'KIR',

 'north korea': 'PRK',

 'south korea': 'KOR',

 'kosovo': 'KSV',

 'kuwait': 'KWT',

 'kyrgyzstan': 'KGZ',

 'laos': 'LAO',

 'latvia': 'LVA',

 'lebanon': 'LBN',

 'lesotho': 'LSO',

 'liberia': 'LBR',

 'libya': 'LBY',

 'liechtenstein': 'LIE',

 'lithuania': 'LTU',

 'luxembourg': 'LUX',

 'macau': 'MAC',

 'macedonia': 'MKD',

 'madagascar': 'MDG',

 'malawi': 'MWI',

 'malaysia': 'MYS',

 'maldives': 'MDV',

 'mali': 'MLI',

 'malta': 'MLT',

 'marshall islands': 'MHL',

 'mauritania': 'MRT',

 'mauritius': 'MUS',

 'mexico': 'MEX',

 'micronesia': 'FSM',

 'moldova': 'MDA',

 'monaco': 'MCO',

 'mongolia': 'MNG',

 'montenegro': 'MNE',

 'morocco': 'MAR',

 'mozambique': 'MOZ',

 'namibia': 'NAM',

 'nepal': 'NPL',

 'netherlands': 'NLD',

 'new caledonia': 'NCL',

 'new zealand': 'NZL',

 'nicaragua': 'NIC',

 'nigeria': 'NGA',

 'niger': 'NER',

 'niue': 'NIU',

 'northern mariana islands': 'MNP',

 'norway': 'NOR',

 'oman': 'OMN',

 'pakistan': 'PAK',

 'palau': 'PLW',

 'panama': 'PAN',

 'papua new guinea': 'PNG',

 'paraguay': 'PRY',

 'peru': 'PER',

 'philippines': 'PHL',

 'poland': 'POL',

 'portugal': 'PRT',

 'puerto rico': 'PRI',

 'qatar': 'QAT',

 'romania': 'ROU',

 'russia': 'RUS',

 'rwanda': 'RWA',

 'saint kitts and nevis': 'KNA',

 'saint lucia': 'LCA',

 'saint martin': 'MAF',

 'saint pierre and miquelon': 'SPM',

 'saint vincent and the grenadines': 'VCT',

 'samoa': 'WSM',

 'san marino': 'SMR',

 'sao tome and principe': 'STP',

 'saudi arabia': 'SAU',

 'senegal': 'SEN',

 'serbia': 'SRB',

 'seychelles': 'SYC',

 'sierra leone': 'SLE',

 'singapore': 'SGP',

 'sint maarten': 'SXM',

 'slovakia': 'SVK',

 'slovenia': 'SVN',

 'solomon islands': 'SLB',

 'somalia': 'SOM',

 'south africa': 'ZAF',

 'south sudan': 'SSD',

 'spain': 'ESP',

 'sri lanka': 'LKA',

 'sudan': 'SDN',

 'suriname': 'SUR',

 'swaziland': 'SWZ',

 'sweden': 'SWE',

 'switzerland': 'CHE',

 'syria': 'SYR',

 'taiwan': 'TWN',

 'tajikistan': 'TJK',

 'tanzania': 'TZA',

 'thailand': 'THA',

 'timor-leste': 'TLS',

 'togo': 'TGO',

 'tonga': 'TON',

 'trinidad and tobago': 'TTO',

 'tunisia': 'TUN',

 'turkey': 'TUR',

 'turkmenistan': 'TKM',

 'tuvalu': 'TUV',

 'uganda': 'UGA',

 'ukraine': 'UKR',

 'united arab emirates': 'ARE',

 'united kingdom': 'GBR',

 'united states': 'USA',

 'uruguay': 'URY',

 'uzbekistan': 'UZB',

 'vanuatu': 'VUT',

 'venezuela': 'VEN',

 'vietnam': 'VNM',

 'virgin islands': 'VGB',

 'west bank': 'WBG',

 'yemen': 'YEM',

 'zambia': 'ZMB',

 'zimbabwe': 'ZWE'}



## countries 

from collections import Counter

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",

    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",

    "#08519c", "#0b4083", "#08306b"

]

    

def geoplot(ddf):

    country_with_code, country = {}, {}

    shows_countries = ", ".join(ddf['country'].dropna()).split(", ")

    for c,v in dict(Counter(shows_countries)).items():

        code = ""

        if c.lower() in country_codes:

            code = country_codes[c.lower()]

        country_with_code[code] = v

        country[c] = v



    data = [dict(

            type = 'choropleth',

            locations = list(country_with_code.keys()),

            z = list(country_with_code.values()),

            colorscale = [[0,"rgb(5, 10, 172)"],[0.65,"rgb(40, 60, 190)"],[0.75,"rgb(70, 100, 245)"],\

                        [0.80,"rgb(90, 120, 245)"],[0.9,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],

            autocolorscale = False,

            reversescale = True,

            marker = dict(

                line = dict (

                    color = 'gray',

                    width = 0.5

                ) ),

            colorbar = dict(

                autotick = False,

                title = ''),

          ) ]



    layout = dict(

        title = '',

        geo = dict(

            showframe = False,

            showcoastlines = False,

            projection = dict(

                type = 'Mercator'

            )

        )

    )



    fig = dict( data=data, layout=layout )

    iplot( fig, validate=False, filename='d3-world-map' )

    return country



country_vals = geoplot(df)

tabs = Counter(country_vals).most_common(25)



labels = [_[0] for _ in tabs][::-1]

values = [_[1] for _ in tabs][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
import plotly.figure_factory as ff

x1 = d2['duration'].fillna(0.0).astype(float)

fig = ff.create_distplot([x1], ['a'], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])

fig.update_layout(title_text='Distplot with Normal Distribution')

fig.show()
col = 'season_count'

vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

data = [trace1]

layout = go.Layout(title="Seasons", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "rating"



vc1 = d1[col].value_counts().reset_index()

vc1 = vc1.rename(columns = {col : "count", "index" : col})

vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))

vc1 = vc1.sort_values(col)



vc2 = d2[col].value_counts().reset_index()

vc2 = vc2.rename(columns = {col : "count", "index" : col})

vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))

vc2 = vc2.sort_values(col)



trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))

trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
col = "listed_in"

categories = ", ".join(d2['listed_in']).split(", ")

counter_list = Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
traces = []

titles = ["United States","", "United Kingdom"]

for title in titles:

    if title != "":

        traces.append(country_trace(title, flag="tv_shows"))



fig = make_subplots(rows=1, cols=3, subplot_titles=titles)

fig.add_trace(traces[0], 1,1)

fig.add_trace(traces[1], 1,3)



fig.update_layout(height=600, showlegend=False)

fig.show()
small = df[df["type"] == "Movie"]

small = small[small["country"] == "India"]



col = "director"

categories = ", ".join(small[col].fillna("")).split(", ")

counter_list = Counter(categories).most_common(12)

counter_list = [_ for _ in counter_list if _[0] != ""]

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="orange"))



data = [trace1]

layout = go.Layout(title="Movie Directors from India with most content", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
small = df[df["type"] == "Movie"]

small = small[small["country"] == "United States"]



col = "director"

categories = ", ".join(small[col].fillna("")).split(", ")

counter_list = Counter(categories).most_common(12)

counter_list = [_ for _ in counter_list if _[0] != ""]

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="orange"))



data = [trace1]

layout = go.Layout(title="Movie Directors from US with most content", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()
tag = "jay karas"

df["relevant"] = df['director'].fillna("").apply(lambda x : 1 if tag in x.lower() else 0)

small = df[df["relevant"] == 1]

small[['title', 'release_year', 'listed_in']]
tag = "Stand-Up Comedy"

df["relevant"] = df['listed_in'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0)

small = df[df["relevant"] == 1]

small[small["country"] == "United States"][["title", "country","release_year"]].head(10)
tag = "Stand-Up Comedy"

df["relevant"] = df['listed_in'].fillna("").apply(lambda x : 1 if tag.lower() in x.lower() else 0)

small = df[df["relevant"] == 1]

small[small["country"] == "India"][["title", "country","release_year"]].head(10)