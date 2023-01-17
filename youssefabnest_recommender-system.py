import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
%matplotlib inline

import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

df1=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')
df2.info()
df2.head(5)
df2.isnull()
sns.heatmap(df2.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df2.select_dtypes('object').nunique()
plt.figure(figsize=(25,6))


plt.subplot(2, 3, 1)
sns.distplot(df2['revenue'])

plt.subplot(2, 3, 2)
sns.distplot(df2['vote_count'])

plt.subplot(2, 3, 3)
sns.distplot(df2['budget'])

plt.subplot(2, 3, 4)
sns.distplot(df2['vote_average'].fillna(0).astype(int))

plt.subplot(2, 3, 5)
sns.distplot(df2['runtime'].fillna(0).astype(int))

plt.subplot(2, 3, 6)
sns.distplot(df2['popularity'].fillna(0).astype(int))

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()
pop= df2.sort_values('revenue', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['revenue'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("revenue")
plt.title("revenue Movies")

pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")

movies = df2
movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


s = movies.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genres_count'
con_df = movies.drop('genres', axis=1).join(s)
con_df = pd.DataFrame(con_df['genres_count'].value_counts())
con_df['genre'] = con_df.index
con_df.columns = ['num_genre', 'genre']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(30)
# spoken_languages
fig = plt.figure(figsize=(20,20))
sns.barplot(data = con_df, x='genre', y = 'num_genre')

plt.tight_layout()
movies = df2
movies['spoken_languages'] = movies['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


s = movies.apply(lambda x: pd.Series(x['spoken_languages']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'spoken_languages_count'
con_df = movies.drop('spoken_languages', axis=1).join(s)
con_df = pd.DataFrame(con_df['spoken_languages_count'].value_counts())
con_df['spoken_language'] = con_df.index
con_df.columns = ['num_spoken_language', 'spoken_language']


con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(100)
con_df = con_df[:5]

fig = plt.figure(figsize=(20,20))
sns.barplot(data = con_df, x='spoken_language', y = 'num_spoken_language')

plt.tight_layout()
movies = df2
movies['production_countries'] = movies['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = movies.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'countries'
con_df = movies.drop('production_countries', axis=1).join(s)
con_df = pd.DataFrame(con_df['countries'].value_counts())
con_df['country'] = con_df.index
con_df.columns = ['num_movies', 'country']
con_df = con_df.reset_index().drop('index', axis=1)
con_df.head(20)



con_df.loc[con_df.country == 'United States of America', 'num_movies'] = 700
con_df.head(20)
con_df.to_csv('mycsvfile.csv')
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_movies'],
        text = con_df['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0,255)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(0,0,0)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]

layout = dict(
    title = 'Production Countries for the MovieLens Movies (USA is being 700+ to be apple to watch other countries)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
html_p1 = """<!DOCTYPE html>
<meta charset='utf-8'>

<!-- Load d3.js -->
<script src='https://d3js.org/d3.v4.js'></script>

<!-- Color palette -->
<script src='https://d3js.org/d3.v5.min.js'></script>

<!-- Create a div where the graph will take place -->
<div id='my_dataviz'></div>

<style>
.node:hover{
  stroke-width: 7px !important;
  opacity: 1 !important;
}
</style>
"""

from IPython.core.display import display, HTML, Javascript
from string import Template
import IPython.display

flare=pd.read_csv('../input/flare1/flare1.csv')
flare.to_csv('mycsvfile.csv')    
fout = open("flare1.csv", "w")
fout.write(flare.to_string())


js_p1 = """
require.config({paths: {d3: "https://d3js.org/d3.v4.min"}});
require(["d3"], function(d3) {
var data = [{"key": "United States","value": 20309},{"key": "India","value": 13721},{"key": "Germany","value": 6459},{"key": "United Kingdom","value": 6221},{"key": "Canada","value": 3393},{"key": "Russian Federation","value": 2869},{"key": "France","value": 2572},{"key": "Brazil","value": 2505},{"key": "Poland","value": 2122},{"key": "Australia","value": 2018},{"key": "Netherlands","value": 1841},{"key": "Spain","value": 1769},{"key": "Italy","value": 1535},{"key": "Ukraine","value": 1279},{"key": "Sweden","value": 1164},{"key": "Pakistan","value": 1050},{"key": "China","value": 1037},{"key": "Switzerland","value": 1010},{"key": "Turkey","value": 1004},{"key": "Israel","value": 1003},{"key": "Iran","value": 921},{"key": "Romania","value": 793},{"key": "Austria","value": 788},{"key": "Czech Republic","value": 784},{"key": "Belgium","value": 743},{"key": "Mexico","value": 736},{"key": "Bangladesh","value": 697},{"key": "Denmark","value": 653},{"key": "South Africa","value": 637},{"key": "Indonesia","value": 630},{"key": "Argentina","value": 611},{"key": "Norway","value": 565},{"key": "New Zealand","value": 557},{"key": "Ireland","value": 554},{"key": "Portugal","value": 528},{"key": "Finland","value": 521},{"key": "Philippines","value": 520},{"key": "Greece","value": 516},{"key": "Hungary","value": 470},{"key": "Sri Lanka","value": 454},{"key": "Bulgaria","value": 425},{"key": "Egypt","value": 419},{"key": "Nigeria","value": 399},{"key": "Singapore","value": 376},{"key": "Malaysia","value": 363},{"key": "Japan","value": 361},{"key": "Serbia","value": 358},{"key": "Colombia","value": 339},{"key": "Belarus","value": 339},{"key": "Viet Nam","value": 331},{"key": "Nepal","value": 295},{"key": "Lithuania","value": 257},{"key": "Croatia","value": 241},{"key": "Slovakia","value": 238},{"key": "Chile","value": 238},{"key": "Slovenia","value": 238},{"key": "Hong Kong (S.A.R.)","value": 219},{"key": "Thailand","value": 213},{"key": "Morocco","value": 213},{"key": "Taiwan","value": 207},{"key": "Kenya","value": 194},{"key": "United Arab Emirates","value": 193},{"key": "Estonia","value": 189},{"key": "South Korea","value": 169},{"key": "Tunisia","value": 163},{"key": "Latvia","value": 145},{"key": "Algeria","value": 130},{"key": "Saudi Arabia","value": 130},{"key": "Peru","value": 128},{"key": "Bosnia and Herzegovina","value": 125},{"key": "Venezuela","value": 123},{"key": "Armenia","value": 117},{"key": "Dominican Republic","value": 115},{"key": "Albania","value": 109},{"key": "Kazakhstan","value": 107},{"key": "Lebanon","value": 107},{"key": "Uruguay","value": 102},{"key": "Costa Rica","value": 98},{"key": "Other Country (Not Listed Above)","value": 84},{"key": "Jordan","value": 83},{"key": "Azerbaijan","value": 78},{"key": "Ghana","value": 76},{"key": "Republic of Moldova","value": 76},{"key": "Georgia","value": 75},{"key": "Uganda","value": 67},{"key": "Malta","value": 66},{"key": "Cuba","value": 65},{"key": "Ecuador","value": 65},{"key": "Afghanistan","value": 64},{"key": "Republic of Korea","value": 62},{"key": "Ethiopia","value": 60},{"key": "Cambodia","value": 60},{"key": "Luxembourg","value": 59},{"key": "Uzbekistan","value": 56},{"key": "Syrian Arab Republic","value": 56},{"key": "Myanmar","value": 55},{"key": "The former Yugoslav Republic of Macedonia","value": 54},{"key": "Guatemala","value": 50},{"key": "Paraguay","value": 49}];
// set the dimensions and margins of the graph
var width = 460
var height = 460

// append the svg object to the body of the page
var svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width)
    .attr("height", height)

// Read data


  // Filter a bit the data -> more than 1 million inhabitants
 

  // Color palette for continents?
  // var color = d3.scaleOrdinal()
  //   .domain(["Asia", "Europe", "Africa", "Oceania", "Americas"])
  //   .range(d3.schemeSet1);

  // Size scale for countries
  var size = d3.scaleLinear()
    .domain([0, 20309])
    .range([7,55])  // circle will be between 7 and 55 px wide

  // create a tooltip
  var Tooltip = d3.select("#my_dataviz")
    .append("div")
    .style("opacity", 0)
    .attr("class", "tooltip")
    .style("background-color", "white")
    .style("border", "solid")
    .style("border-width", "2px")
    .style("border-radius", "5px")
    .style("padding", "5px")

  // Three function that change the tooltip when user hover / move / leave a cell
  var mouseover = function(d) {
    Tooltip
      .style("opacity", 1)
  }
  var mousemove = function(d) {
    Tooltip
      .html('<u>' + d.key + '</u>' + "<br>" + d.value + " inhabitants")
      .style("left", (d3.mouse(this)[0]+20) + "px")
      .style("top", (d3.mouse(this)[1]) + "px")
  }
  var mouseleave = function(d) {
    Tooltip
      .style("opacity", 0)
  }

  // Initialize the circle: all located at the center of the svg area
  var node = svg.append("g")
    .selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
      .attr("class", "node")
      .attr("r", function(d){ return size(d.value)})
      .attr("cx", width / 2)
      .attr("cy", height / 2)
      .style("fill-opacity", 0.8)
      .attr("stroke", "red")
      .style("stroke-width", 1)
      .on("mouseover", mouseover) // What to do when hovered
      .on("mousemove", mousemove)
      .on("mouseleave", mouseleave)
      .call(d3.drag() // call specific function when circle is dragged
           .on("start", dragstarted)
           .on("drag", dragged)
           .on("end", dragended));

  // Features of the forces applied to the nodes:
  var simulation = d3.forceSimulation()
      .force("center", d3.forceCenter().x(width / 2).y(height / 2)) // Attraction to the center of the svg area
      .force("charge", d3.forceManyBody().strength(.1)) // Nodes are attracted one each other of value is > 0
      .force("collide", d3.forceCollide().strength(.2).radius(function(d){ return (size(d.value)+3) }).iterations(1)) // Force that avoids circle overlapping

  // Apply these forces to the nodes and update their positions.
  // Once the force algorithm is happy with positions ('alpha' value is low enough), simulations will stop.
  simulation
      .nodes(data)
      .on("tick", function(d){
        node
            .attr("cx", function(d){ return d.x; })
            .attr("cy", function(d){ return d.y; })
      });

  // What happens when a circle is dragged?
  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(.03).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }
  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(.03);
    d.fx = null;
    d.fy = null;
  }
});"""
# h = display(HTML(html_p1))
# j = IPython.display.Javascript(js_p1)
# IPython.display.display_javascript(j)