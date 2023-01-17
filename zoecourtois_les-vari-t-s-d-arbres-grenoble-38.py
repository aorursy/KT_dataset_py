import math

import random



import geopandas as gpd

import descartes



import folium

import folium.plugins

print("Installation réussie")
arbres = gpd.read_file("../input/arbres-grenoble/arbres.json")
color = {

    espece: "#%06x" % random.randint(0, 0xFFFFFF) for espece in arbres["ESPECE"]

}
carte = folium.Map(

    location=[45.180712, 5.721979], 

    zoom_start=12.5, 

    min_zoom=11, 

    max_zoom=19,

    legend_name="Les variétés d'arbres à Grenoble, Isère - Z. Courtois",

    tiles='Stamen Toner'

)



for _, row in arbres.iterrows():

    x = row["geometry"].x

    y = row["geometry"].y

    if not math.isnan(y) and not math.isnan(x):

        folium.Circle(

            location=[y, x],

            radius=0.5,

            color=color[row["ESPECE"]],

            fill=False

        ).add_to(carte)
carte