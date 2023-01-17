import plotly.express as px

gapminder = px.data.gapminder()

px.scatter(gapminder, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

           size="pop", color="continent", hover_name="country",

           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
import plotly.graph_objects as go



fig = go.Figure(

    data=[go.Scatter(x=[0, 1], y=[0, 1])],

    layout=go.Layout(

        xaxis=dict(range=[0, 5], autorange=False),

        yaxis=dict(range=[0, 5], autorange=False),

        title="Start Title",

        updatemenus=[dict(

            type="buttons",

            buttons=[dict(label="Play",

                          method="animate",

                          args=[None])])]

    ),

    frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),

            go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),

            go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])],

                     layout=go.Layout(title_text="End Title"))]

)



fig.show()
import plotly.graph_objects as go



import numpy as np



# Generate curve data

t = np.linspace(-1, 1, 100)

x = t + t ** 2

y = t - t ** 2

xm = np.min(x) - 1.5

xM = np.max(x) + 1.5

ym = np.min(y) - 1.5

yM = np.max(y) + 1.5

N = 50

s = np.linspace(-1, 1, N)

xx = s + s ** 2

yy = s - s ** 2





# Create figure

fig = go.Figure(

    data=[go.Scatter(x=x, y=y,

                     mode="lines",

                     line=dict(width=2, color="blue")),

          go.Scatter(x=x, y=y,

                     mode="lines",

                     line=dict(width=2, color="blue"))],

    layout=go.Layout(

        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),

        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),

        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",

        updatemenus=[dict(type="buttons",

                          buttons=[dict(label="Play",

                                        method="animate",

                                        args=[None])])]),

    frames=[go.Frame(

        data=[go.Scatter(

            x=[xx[k]],

            y=[yy[k]],

            mode="markers",

            marker=dict(color="red", size=10))])



        for k in range(N)]

)



fig.show()
import plotly.graph_objects as go

import numpy as np



# Create figure

fig = go.Figure()



# Add traces, one for each slider step

for step in np.arange(0, 5, 0.1):

    fig.add_trace(

        go.Scatter(

            visible=False,

            line=dict(color="#00CED1", width=6),

            name="𝜈 = " + str(step),

            x=np.arange(0, 10, 0.01),

            y=np.sin(step * np.arange(0, 10, 0.01))))



# Make 10th trace visible

fig.data[10].visible = True



# Create and add slider

steps = []

for i in range(len(fig.data)):

    step = dict(

        method="restyle",

        args=["visible", [False] * len(fig.data)],

    )

    step["args"][1][i] = True  # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active=10,

    currentvalue={"prefix": "Frequency: "},

    pad={"t": 50},

    steps=steps

)]



fig.update_layout(

    sliders=sliders

)



fig.show()
import plotly.graph_objects as go



fig =go.Figure(go.Sunburst(

 ids=[

    "North America", "Europe", "Australia", "North America - Football", "Soccer",

    "North America - Rugby", "Europe - Football", "Rugby",

    "Europe - American Football","Australia - Football", "Association",

    "Australian Rules", "Autstralia - American Football", "Australia - Rugby",

    "Rugby League", "Rugby Union"

  ],

  labels= [

    "North<br>America", "Europe", "Australia", "Football", "Soccer", "Rugby",

    "Football", "Rugby", "American<br>Football", "Football", "Association",

    "Australian<br>Rules", "American<br>Football", "Rugby", "Rugby<br>League",

    "Rugby<br>Union"

  ],

  parents=[

    "", "", "", "North America", "North America", "North America", "Europe",

    "Europe", "Europe","Australia", "Australia - Football", "Australia - Football",

    "Australia - Football", "Australia - Football", "Australia - Rugby",

    "Australia - Rugby"

  ],

))

fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))



fig.show()