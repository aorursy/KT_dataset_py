#Library imports

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

from plotly import tools

init_notebook_mode(connected=True)



import numpy as np

import pandas as pd

import os

import json  ,IPython                        

from IPython.core.display import display, HTML, Javascript

#print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#Read data

restaurants = pd.read_csv('../input/zomato.csv')



restaurants.transpose()
print('There are '+ str(len(restaurants['name'].unique()))+ ' restaurants, listed in ' +str(len(restaurants['location'].unique())) + ' different locations.')
#Add ratings column

restaurants['rate'].fillna('-1/5', inplace= True)

restaurants['ratings'] = restaurants['rate'].apply(lambda x : -2 if (x=='NEW' or x=='-') else float(x.split('/')[0]))

#restaurants['ratings'].unique()



temp = restaurants.fillna('Missing')

temp = temp.applymap(lambda x: x if x == 'Missing' else 'Available')

figsize_width = 12

figsize_height = len(temp.columns)*0.5

plt_data = pd.DataFrame()

for col in temp.columns:

    temp_col = temp.groupby(col).size()/len(temp.index)

    temp_col = pd.DataFrame({col:temp_col})

    plt_data = pd.concat([plt_data, temp_col], axis=1)

    

ax = plt_data.T.plot(kind='barh', stacked=True, figsize=(figsize_width, figsize_height))



# Annotations

labels = []

for i in plt_data.index:

    for j in plt_data.columns:

        label = '{:.2%}'.format(plt_data.loc[i][j])

        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):

    width = rect.get_width()

    if width > 0:

        x = rect.get_x()

        y = rect.get_y()

        height = rect.get_height()

        ax.text(x + width/2., y + height/2., label, ha='center', va='center')



plt.xlabel('Frequency')

plt.title('Missing values')

plt.xticks(np.arange(0, 1.05, 0.1))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
def getTopNRestsWithVotesCount(restaurants, location, n=10):

    temp_rest = restaurants[restaurants['location']==location][['name','votes','ratings',

                                                                'approx_cost(for two people)']].groupby(['name',

                                                                                                         'approx_cost(for two people)']).agg({'votes': np.sum,'ratings': np.mean}).reset_index()

    #temp_rest = temp_rest.groupby(['name','votes']).size().reset_index(name='Freq').sort_values('votes',ascending = False)

    temp_rest = temp_rest.sort_values('ratings',ascending = False).head(n)

    return temp_rest.round({'ratings': 2})



def getLocationWiseActiveRestaurants(restaurants, locationCount=10, restaurantsCount=10):

    temp_votes = restaurants.groupby(['location'])['votes'].sum().reset_index().sort_values('votes', ascending=False).head(locationCount)

    result = pd.DataFrame()

    for index,row in temp_votes.iterrows():

        df = getTopNRestsWithVotesCount(restaurants, row['location'], restaurantsCount)

        df['location'] = row['location']

        df['location_votes'] = row['votes']

        result = result.append(df)

    return result



def getChildrenNodes(df, size='', title='1', level=0):

    resultList = []

    result = {}

    result['name'] = str(title)

    result['size'] = size

    result['level'] = level

    children = df[df['name']==title][['children','size']].values

    if len(children)>0:

        for child,sz in children:

            resultList.append(getChildrenNodes(df, sz, child,level+1))

        result['children'] = resultList

    else:

        return result

    return result



def createDataForBubblePlot(df):

    temp_rests = df.groupby(['location','location_votes']).size().reset_index(name='Freq')

    result = df

    temp_df = pd.DataFrame(columns = ['name','votes','ratings','location','location_votes'])

    for index,row in temp_rests.iterrows():

        temp_df.loc[0]=[row['location'],row['location_votes'],'0','1','0']

        result = result.append(temp_df)

    result['children'],result['name'],result['size'] = result['name'],result['location'],result['votes']

    #getChildrenNodes(result)

    with open('output.json', 'w') as outfile:  

        json.dump(getChildrenNodes(result), outfile)

    #return result
htmllocationbubble = """<!DOCTYPE html><meta charset="utf-8"><style>.node {cursor: pointer;}.node:hover {stroke: #000;stroke-width: 1.5px;}.node--leaf {fill: white;}

.label {font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;text-anchor: middle;text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;}

.label,.node--root,.node--leaf {pointer-events: none;}</style><svg id="two" width="760" height="760"></svg>

"""

js_locationbubble="""

require.config({

        paths: {

            d3: "https://d3js.org/d3.v4.min"

         }

     });

require(["d3"], function(d3) {

var svg = d3.select("#two"),

    margin = 20,

    diameter = +svg.attr("width"),

    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")"),

    color = d3.scaleSequential(d3.interpolateViridis).domain([-2, 2]),

    pack = d3.pack().size([diameter - margin, diameter - margin]).padding(2);

d3.json("output.json", function(t, n) {

if (t) throw t;

var r, e = n = d3.hierarchy(n).sum(function(t) {

        return t.size

    }).sort(function(t, n) {

        return n.value - t.value

    }),

    a = pack(n).descendants(),

    i = g.selectAll("circle").data(a).enter().append("circle").attr("class", function(t) {

        return t.parent ? t.children ? "node" : "node node--leaf" : "node node--root"

    }).style("fill", function(t) {

        return t.children ? color(t.depth) : null

    }).on("click", function(t) {

        e !== t && (l(t), d3.event.stopPropagation())

    }),

    o = (g.selectAll("text").data(a).enter().append("text").attr("class", "label").style("fill-opacity", function(t) {

        return t.parent === n ? 1 : 0

    }).style("display", function(t) {

        return t.parent === n ? "inline" : "none"

    }).text(function(t) {

        return t.data.name + ": " + t.data.size

    }), g.selectAll("circle,text"));



function l(t) {

    e = t, d3.transition().duration(d3.event.altKey ? 7500 : 750).tween("zoom", function(t) {

        var n = d3.interpolateZoom(r, [e.x, e.y, 2 * e.r + margin]);

        return function(t) {

            c(n(t))

        }

    }).selectAll("text").filter(function(t) {

        return t.parent === e || "inline" === this.style.display

    }).style("fill-opacity", function(t) {

        return t.parent === e ? 1 : 0

    }).on("start", function(t) {

        t.parent === e && (this.style.display = "inline")

    }).on("end", function(t) {

        t.parent !== e && (this.style.display = "none")

    })

}



function c(n) {

    var e = diameter / n[2];

    r = n, o.attr("transform", function(t) {

        return "translate(" + (t.x - n[0]) * e + "," + (t.y - n[1]) * e + ")"

    }), i.attr("r", function(t) {

        return t.r * e

    })

}

svg.style("background", color(-1)).on("click", function() {

    l(n)

}), c([n.x, n.y, 2 * n.r + margin])

});

});"""
locationWiseActiveRestaurants = getLocationWiseActiveRestaurants(restaurants)

createDataForBubblePlot(locationWiseActiveRestaurants)



h = display(HTML(htmllocationbubble))

display(HTML('''<h3>Graph is interactive. Click on the circles for more info.</h3>'''))

j = IPython.display.Javascript(js_locationbubble)

IPython.display.display_javascript(j)
def getTopRestaurantsForDish(data, dishname='Pasta', n=10):

    dishData = data[data['dish_liked']==dishname]

    dishData = dishData.groupby(['name','location','dish_liked','ratings']).agg({'votes': np.mean})

    dishData = dishData.sort_values(['votes'], ascending=False).sort_values(['ratings'], ascending=False).reset_index()

    dishData['index'] = dishData.index

    dishData['Rank'] = dishData.groupby('location')['index'].rank(ascending=True)

    return dishData.loc[dishData['Rank']<=n]



def getTopRestaurantsFromTopLocations(data, n=10):

    restData = data.groupby(['location'])['votes'].sum().reset_index()

    restData = restData.sort_values('votes', ascending=False)

    restData = restData.head(n)

    return data[data['location'].isin(restData['location'].values)]



def createDishGraph(dishName, data):

    nodes = []

    links = []

    nodes.append({"id": 'Dish: '+dishName, "group": 0, "size": 15})

    tempLocations = data.groupby('location')['votes'].sum().reset_index()

    tempLocations['perc'] = tempLocations['votes']*100/tempLocations['votes'].sum()

    for index,row in tempLocations.iterrows():

        nodes.append({"id": 'Location: '+row['location'], "group": index+1, "size": int(round(row['perc'],0))})

        links.append({"source": 'Dish: '+dishName, "target": 'Location: '+row['location'], "value": 1})

        tempRests = data[data['location']==row['location']]

        tempRests['perc'] = tempRests['votes']*100/tempRests['votes'].sum()

        for ind,rr in tempRests.iterrows():

            nodes.append({"id": row['location']+'- Restaurant: '+rr['name']+'. Rated: '+str(rr['ratings']), "group": index+1, "size": int(round(rr['perc'],0))})

            links.append({"source": 'Location: '+row['location'], "target": row['location']+'- Restaurant: '+rr['name']+'. Rated: '+str(rr['ratings']), "value": 1})

    doc = {'nodes' : nodes, 'links' : links}

    with open('dishGraph.json', 'w') as outfile:  

        json.dump(doc, outfile)
htmlDishGraph = """<!DOCTYPE html>

<meta charset="utf-8">

<style>



.links line {

  stroke: #999;

  stroke-opacity: 0.8;

}

.node text {

  pointer-events: none;

  font: 10px sans-serif;

}



.tooldiv {

    display: inline-block;

    width: 120px;

    background-color: white;

    color: #000;

    text-align: center;

    padding: 5px 0;

    border-radius: 6px;

    z-index: 1;

}

.nodes circle {

  stroke: #fff;

  stroke-width: 1.5px;

}



div.tooltip {

    position: absolute;

    text-align: center;

    width: 100px;

    height: 65px;

    padding: 2px;

    font: 12px sans-serif;

    background: lightsteelblue;

    border: 0px;

    border-radius: 8px;

    pointer-events: none;

}



</style>

<svg id="dg" width="760" height="760"></svg>"""



jsDishGraph = """require.config({

    paths: {

        d3: "https://d3js.org/d3.v4.min"

     }

 });

 

 require(["d3"], function(d3) {

var svg = d3.select("#dg"),

    width = +svg.attr("width"),

    height = +svg.attr("height");



var color = d3.scaleOrdinal(d3.schemeCategory20);



var simulation = d3.forceSimulation()

    // fix the link distance, charge and the center layout  

    .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(120).strength(1))

    .force("charge", d3.forceManyBody().strength(-155))

    .force("center", d3.forceCenter(width / 2, height / 2));



d3.json("dishGraph.json", function(error, graph) {

  if (error) throw error;



  var link = svg.append("g")

      .attr("class", "links")

    .selectAll("line")

    .data(graph.links)

    .enter().append("line")

      .attr("stroke-width", function(d) { return Math.sqrt(d.value); });



// Define the div for the tooltip

var div = d3.select("body").append("div")

    .attr("class", "tooltip")

    .style("opacity", 0);



  var node = svg.append("g")

      .attr("class", "nodes")

    .selectAll("circle")

    .data(graph.nodes)

    .enter().append("circle")

      .attr("r", function(d) {return d.size})

      .attr("fill", function(d) { return color(d.group); })

      .call(d3.drag()

          .on("start", dragstarted)

          .on("drag", dragged)

          .on("end", dragended)).on("mouseover", function(d) {

            div.transition()

                .duration(200)

                .style("opacity", .9);

            div.html(d.id )

                .style("left", (d3.event.pageX) + "px")

                .style("top", (d3.event.pageY - 28) + "px");

            })

        .on("mouseout", function(d) {

            div.transition()

                .duration(500)

                .style("opacity", 0);

        });

          

    

  //node.append("title")

   // .text(function(d) { return d.id; });



  simulation

      .nodes(graph.nodes)

      .on("tick", ticked);

      



  simulation.force("link")

      .links(graph.links);



  function ticked() {

    link

        .attr("x1", function(d) { return d.source.x; })

        .attr("y1", function(d) { return d.source.y; })

        .attr("x2", function(d) { return d.target.x; })

        .attr("y2", function(d) { return d.target.y; });



    node

        .attr("cx", function(d) { return d.x; })

        .attr("cy", function(d) { return d.y; });

  }

});



function dragstarted(d) {

  if (!d3.event.active) simulation.alphaTarget(0.3).restart();

  d.fx = d.x;

  d.fy = d.y;

}



function dragged(d) {

  d.fx = d3.event.x;

  d.fy = d3.event.y;

}



function dragended(d) {

  if (!d3.event.active) simulation.alphaTarget(0);

  d.fx = null;

  d.fy = null;

}

 });

"""



temp = restaurants.groupby(['name','location','dish_liked','ratings']).agg({'votes': np.mean}).reset_index()

s = temp["dish_liked"].str.split(',', expand=True).stack()

i = s.index.get_level_values(0)

temp = temp.loc[i].copy()

temp["dish_liked"] = s.values

temp['dish_liked'] = temp['dish_liked'].apply(lambda x: x.strip())



dfTemp = getTopRestaurantsForDish(temp)

dfTemp = getTopRestaurantsFromTopLocations(dfTemp,5)
createDishGraph('Pasta', dfTemp)

h = display(HTML(htmlDishGraph))

j = IPython.display.Javascript(jsDishGraph)

IPython.display.display_javascript(j)
temp = getLocationWiseActiveRestaurants(restaurants, 5, 20).reset_index()

#temp.head()



colors = ['blue', 'orange', 'green', 'red', 'purple']



opt = []

opts = []

for i in range(0, len(colors)):

    opt = dict(

        target = temp['location'][[i]].unique(), value = dict(marker = dict(color = colors[i]))

    )

    opts.append(opt)



data = [dict(

  type = 'scatter',

  mode = 'markers',

  x = temp['approx_cost(for two people)'],

  y = temp['ratings'],

  text = temp['name'],

  hoverinfo = 'text',

  opacity = 0.8,

  marker = dict(

      size = temp['votes'],

      sizemode = 'area',

      sizeref = 100

  ),

  transforms = [

      dict(

        type = 'groupby',

        groups = temp['location'],

        styles = opts

    )]

)]



layout = dict(

    title = '<b>Location wise top rated restaurants</b>',

    yaxis = dict(

        title='Ratings'

        #type = 'log'

    ),

    xaxis = dict(

        title='Approx cost for two people'

    )

)





iplot({'data': data, 'layout': layout}, validate=False)
def group_lower_ranking_values(pie_raw, column):

    """Converts pie_raw dataframe with multiple categories to a dataframe with fewer categories

    

    Calculate the 85th quantile and group the lesser values together.

    Lesser values will be labelled as 'Other'

    

    Parameters

    ----------

    pie_raw : DataFrame

        dataframe with the data to be aggregated

    column : str

        name of the column based on which dataframe values will be aggregated

    """

    pie_counts = pie_raw.groupby(column).agg('count')

    pct_value = pie_counts[lambda df: df.columns[0]].quantile(.85)

    values_below_pct_value = pie_counts[lambda df: df.columns[0]].loc[lambda s: s < pct_value].index.values

    def fix_values(row):

        if row[column] in values_below_pct_value:

            row[column] = 'Other'

        return row 

    pie_grouped = pie_raw.apply(fix_values, axis=1).groupby(column).agg('count')

    return pie_grouped



temp = restaurants.groupby(['name','cuisines']).size().reset_index(name='Freq')

s = temp["cuisines"].str.split(',', expand=True).stack()

i = s.index.get_level_values(0)

temp = temp.loc[i].copy()

temp["cuisines"] = s.values

temp['cuisines'] = temp['cuisines'].apply(lambda x: x.strip())

temp = group_lower_ranking_values(temp, 'cuisines').sort_values('name', ascending=False)

temp.drop('Other', inplace=True)



trace = go.Bar(

            y=temp['name'],

            x=temp.index

    )

data = [trace]

layout = go.Layout(xaxis=dict(tickangle=-45),

                   yaxis = dict(title='Number of restaurants'),

    title='Most popular cuisines in Bangalore',

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='cuisine-bangalore')
def getCuisineTopRestaurants(data, cuisine, n=10):

    temp = data[data['cuisines']==cuisine]

    temp = temp.groupby(['name','cuisines']).agg({'ratings': np.mean, 'votes': np.mean}).reset_index()

    temp = temp.sort_values('ratings', ascending=False)

    temp = temp.sort_values('votes', ascending=False)

    return temp.round({'votes':0}).head(n)



temp = restaurants.groupby(['name','cuisines','ratings']).agg({'votes': np.mean}).reset_index()

s = temp["cuisines"].str.split(',', expand=True).stack()

i = s.index.get_level_values(0)

temp = temp.loc[i].copy()

temp["cuisines"] = s.values

temp['cuisines'] = temp['cuisines'].apply(lambda x: x.strip())



interestCuisines = ['North Indian', 'Chinese', 'South Indian', 'Fast Food']

north_df = getCuisineTopRestaurants(temp, 'North Indian')

chinese_df = getCuisineTopRestaurants(temp, 'Chinese')

south_df = getCuisineTopRestaurants(temp, 'South Indian')

fast_df = getCuisineTopRestaurants(temp, 'Fast Food')
trace_north = go.Scatter(x=list(north_df['name']),

                        y=list(north_df['votes']),

                         name='votes',

                        line=dict(color='#33CFA5'))



trace_north_rate = go.Scatter(x=list(north_df['name']),

                        y=list(north_df['ratings']),

                        yaxis='y2',

                        name='rated',

                        line=dict(color='#ff7f0e'))



trace_chinese = go.Scatter(x=list(chinese_df['name']),

                            y=list(chinese_df['votes']),

                           visible=False,

                            name='votes',

                            line=dict(color='#33CFA5'))



trace_chinese_rate = go.Scatter(x=list(chinese_df['name']),

                            y=list(chinese_df['ratings']),

                            yaxis='y2',

                           visible=False,

                            name='rated',

                            line=dict(color='#ff7f0e'))



trace_south = go.Scatter(x=list(south_df['name']),

                       y=list(south_df['votes']),

                       name='votes',

                         visible=False,

                       line=dict(color='#33CFA5'))



trace_south_rate = go.Scatter(x=list(south_df['name']),

                       y=list(south_df['ratings']),

                        yaxis='y2',

                       name='rated',

                         visible=False,

                       line=dict(color='#ff7f0e'))



trace_fast = go.Scatter(x=list(fast_df['name']),

                           y=list(fast_df['votes']),

                           name='votes',

                        visible=False,

                           line=dict(color='#33CFA5'))



trace_fast_rate = go.Scatter(x=list(fast_df['name']),

                           y=list(fast_df['ratings']),

                            yaxis='y2',

                           name='rated',

                        visible=False,

                           line=dict(color='#ff7f0e'))



data = [trace_north, trace_chinese, trace_south, trace_fast, 

        trace_north_rate, trace_chinese_rate, trace_south_rate, trace_fast_rate]



updatemenus = list([

    dict(active=0,

         buttons=list([   

            dict(label = 'North Indian',

                 method = 'update',

                 args = [{'visible': [True, False, False, False, True, False, False, False]},

                         {'title': 'North Indian'}]),

            dict(label = 'Chinese',

                 method = 'update',

                 args = [{'visible': [False, True, False, False, False, True, False, False]},

                         {'title': 'Chinese'}]),

            dict(label = 'South Indian',

                 method = 'update',

                 args = [{'visible': [False, False, True, False, False, False, True, False]},

                         {'title': 'South Indian'}]),

            dict(label = 'Fast Food',

                 method = 'update',

                 args = [{'visible': [False, False, False, True, False, False, False, True]},

                         {'title': 'Fast Food'}])

        ]),

    )

])



layout = dict(title='<b>Top cuisine serving restaurants.</b>', showlegend=False,

              yaxis=dict(title='Votes'),

              xaxis=dict(title='Restaurants'),

              yaxis2=dict(title='Ratings',             

                overlaying='y',

                side='right'),

              updatemenus=updatemenus)



fig = dict(data=data, layout=layout)

iplot(fig, filename='update_dropdown')