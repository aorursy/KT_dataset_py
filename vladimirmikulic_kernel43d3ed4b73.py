import plotly.express as px



# Greenhouse Gas Emissions data visualized in treemap plotly epxress chart

# http://www.coolingman.org/wp-content/uploads/2016/08/BM-GHG-Inventory-1.png

fig = px.treemap(

    names=[

        "Event GHG",

        "Travel Emissions",

        "Air Travel",

        "International",

        "Domestic",

        "Road Travel",

        "On-playa emissions",

        "Power generation",

        "Art Cars / Vehicles",

        "Fire Art",

        "Burning of the Man",

    ],

    parents=[

        "",

        "Event GHG",

        "Travel Emissions",

        "Air Travel",

        "Air Travel",

        "Travel Emissions",

        "Event GHG",

        "On-playa emissions",

        "On-playa emissions",

        "On-playa emissions",

        "On-playa emissions",

    ],

    # Total tons (sizes of the boxes depend on these values)

    values=[27492, 25019, 7697, 4329, 3368, 17322, 2473, 1432, 685, 243, 112],

    # Total tons in percentages (%)

    hover_data={

        "% of Total": [

            "100%",

            "91%",

            "28%",

            "16%",

            "12%",

            "63%",

            "9%",

            "5%",

            "2%",

            "1%",

            "0%",

        ],

    },

    # Boxes fill-in the available space (default value is remainder)

    branchvalues="total",

)



fig.show()
