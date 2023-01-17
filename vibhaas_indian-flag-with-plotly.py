# Create Dataframe with datapoint for flag

# Flag proportions are in 2 Height and 3 Width

import pandas as pd



rows = []

columns = []



for r in range(90):

    for c in range(60):

     rows.append([r,c])

        





df = pd.DataFrame(rows, columns=["A", "B"])

print(df)
# Add Structure for Flag

import plotly.express as px



fig = px.scatter(df, x="A", y="B")



fig.show()
# Adding Color Band

import pandas as pd



df['C'] = df['B'].apply(lambda x: 'g' if x <=20 else ('w' if x <=40 else 'o'))



print (df)
import plotly.express as px

#Create flag in 2:3 proportion



fig = px.scatter(df, x="A", y="B", color = "C", color_discrete_sequence=["green", "white", "orange"], width=900, height=600)

fig.update_traces(marker=dict(size=10),selector=dict(mode='markers'))



# Add Ashok Chakra

fig.update_layout(

    shapes=[

       

        # filled circle

        dict(

            type="circle",xref="x",yref="y",

            fillcolor="white",

            x0=34,y0=20,x1=54,y1=40,

            line_color="#000080",

        ),

    ]

)



# Add line to Ashok Chakra

fig.add_shape( type="line", xref="x", yref="y", x0=44, y0=20, x1=44, y1=40, line=dict( color="#000080", width=4,),)



fig.add_shape( type="line",xref="x",yref="y",x0=34,y0=30,x1=54,y1=30,line=dict( color="#000080", width=4,),)

fig.add_shape( type="line",xref="x",yref="y",x0=37,y0=23,x1=51, y1=37,line=dict( color="#000080",width=4,),)



fig.add_shape( type="line",xref="x",yref="y",x0=37,y0=37,x1=51,y1=23,line=dict( color="#000080",width=4, ),)

fig.add_shape( type="line",xref="x",yref="y",x0=35,y0=25,x1=53,y1=35,line=dict( color="#000080",width=4, ),)

fig.add_shape( type="line",xref="x",yref="y",x0=34,y0=27,x1=54,y1=33,line=dict( color="#000080",width=4, ),)



fig.add_shape( type="line",xref="x",yref="y",x0=34,y0=32,x1=54,y1=28,line=dict( color="#000080",width=4, ),)

fig.add_shape( type="line",xref="x",yref="y",x0=35,y0=35,x1=53,y1=25,line=dict( color="#000080",width=4, ),)



fig.add_shape( type="line",xref="x",yref="y",x0=39,y0=39,x1=49,y1=21,line=dict( color="#000080",width=4, ),)

fig.add_shape( type="line",xref="x",yref="y",x0=41,y0=40,x1=47,y1=20,line=dict( color="#000080",width=4, ),)



fig.add_shape( type="line",xref="x",yref="y",x0=46,y0=40,x1=42,y1=20,line=dict( color="#000080",width=4, ),)

fig.add_shape( type="line",xref="x",yref="y",x0=49,y0=39,x1=39,y1=21,line=dict( color="#000080",width=4, ),)

fig.show()