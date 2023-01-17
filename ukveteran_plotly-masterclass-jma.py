import plotly.express as px

iris = px.data.iris()

fig = px.scatter(iris, x="sepal_width", y="sepal_length")

fig.show()
import plotly.express as px

iris = px.data.iris()

fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species",

                 size='petal_length', hover_data=['petal_width'])

fig.show()
import plotly.express as px

gapminder = px.data.gapminder().query("continent == 'Oceania'")

fig = px.line(gapminder, x='year', y='lifeExp', color='country')

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Scatter(

    x=[1, 2, 3, 4],

    y=[10, 11, 12, 13],

    mode='markers',

    marker=dict(size=[40, 60, 80, 100],

                color=[0, 1, 2, 3])

))



fig.show()
import plotly.express as px

data_canada = px.data.gapminder().query("country == 'Canada'")

fig = px.bar(data_canada, x='year', y='pop')

fig.show()
import plotly.express as px

data = px.data.gapminder()



data_canada = data[data.country == 'Canada']

fig = px.bar(data_canada, x='year', y='pop',

             hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',

             labels={'pop':'population of Canada'}, height=400)

fig.show()
import plotly.graph_objects as go

import numpy as np



x = np.arange(10)



fig = go.Figure(data=go.Scatter(x=x, y=x**2))

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=[go.Sankey(

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(color = "black", width = 0.5),

      label = ["A1", "A2", "B1", "B2", "C1", "C2"],

      color = "blue"

    ),

    link = dict(

      source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A2, B1, ...

      target = [2, 3, 3, 4, 4, 5],

      value = [8, 4, 2, 8, 4, 2]

  ))])



fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data =

    go.Contour(

        z=[[10, 10.625, 12.5, 15.625, 20],

           [5.625, 6.25, 8.125, 11.25, 15.625],

           [2.5, 3.125, 5., 8.125, 12.5],

           [0.625, 1.25, 3.125, 6.25, 10.625],

           [0, 0.625, 2.5, 5.625, 10]]

    ))

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data = 

    go.Contour(

        z=[[10, 10.625, 12.5, 15.625, 20],

           [5.625, 6.25, 8.125, 11.25, 15.625],

           [2.5, 3.125, 5., 8.125, 12.5],

           [0.625, 1.25, 3.125, 6.25, 10.625],

           [0, 0.625, 2.5, 5.625, 10]],

        x=[-9, -6, -5 , -3, -1], # horizontal axis

        y=[0, 1, 4, 5, 7] # vertical axis

    ))

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Heatmap(

                    z=[[1, 20, 30],

                      [20, 1, 60],

                      [30, 60, 1]]))

fig.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Heatmap(

                   z=[[1, 20, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],

                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],

                   y=['Morning', 'Afternoon', 'Evening']))

fig.show()
import plotly.graph_objects as go

import numpy as np



# Build the rectangles as a heatmap

# specify the edges of the heatmap squares

phi = (1 + np.sqrt(5) )/2. # golden ratio

xe = [0, 1, 1+(1/(phi**4)), 1+(1/(phi**3)), phi]

ye = [0, 1/(phi**3), 1/phi**3+1/phi**4, 1/(phi**2), 1]



z = [ [13,3,3,5],

      [13,2,1,5],

      [13,10,11,12],

      [13,8,8,8]

    ]



fig = go.Figure(data=go.Heatmap(

          x = np.sort(xe),

          y = np.sort(ye),

          z = z,

          type = 'heatmap',

          colorscale = 'Viridis'))



# Add spiral line plot



def spiral(th):

    a = 1.120529

    b = 0.306349

    r = a*np.exp(-b*th)

    return (r*np.cos(th), r*np.sin(th))



theta = np.linspace(-np.pi/13,4*np.pi,1000); # angle

(x,y) = spiral(theta)



fig.add_trace(go.Scatter(x= -x+x[0], y= y-y[0],

     line =dict(color='white',width=3)))



axis_template = dict(range = [0,1.6], autorange = False,

             showgrid = False, zeroline = False,

             linecolor = 'black', showticklabels = False,

             ticks = '' )



fig.update_layout(margin = dict(t=200,r=200,b=200,l=200),

    xaxis = axis_template,

    yaxis = axis_template,

    showlegend = False,

    width = 700, height = 700,

    autosize = False )



fig.show()
import plotly.graph_objects as go

import numpy as np

np.random.seed(1)



N = 70



fig = go.Figure(data=[go.Mesh3d(x=(70*np.random.randn(N)),

                   y=(55*np.random.randn(N)),

                   z=(40*np.random.randn(N)),

                   opacity=0.5,

                   color='rgba(244,22,100,0.6)'

                  )])



fig.update_layout(scene = dict(

        xaxis = dict(nticks=4, range=[-100,100],),

                     yaxis = dict(nticks=4, range=[-50,100],),

                     zaxis = dict(nticks=4, range=[-100,100],),),

                     width=700,

                     margin=dict(r=20, l=10, b=10, t=10))



fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots

import numpy as np



N = 50



fig = make_subplots(rows=2, cols=2,

                    specs=[[{'is_3d': True}, {'is_3d': True}],

                           [{'is_3d': True}, {'is_3d': True}]],

                    print_grid=False)

for i in [1,2]:

    for j in [1,2]:

        fig.append_trace(

            go.Mesh3d(

                x=(60*np.random.randn(N)),

                y=(25*np.random.randn(N)),

                z=(40*np.random.randn(N)),

                opacity=0.5,

              ),

            row=i, col=j)



fig.update_layout(width=700, margin=dict(r=10, l=10, b=10, t=10))

# fix the ratio in the top left subplot to be a cube

fig.update_layout(scene_aspectmode='cube')

# manually force the z-axis to appear twice as big as the other two

fig.update_layout(scene2_aspectmode='manual',

                  scene2_aspectratio=dict(x=1, y=1, z=2))

# draw axes in proportion to the proportion of their ranges

fig.update_layout(scene3_aspectmode='data')

# automatically produce something that is well proportioned using 'data' as the default

fig.update_layout(scene4_aspectmode='auto')

fig.show()
import plotly.express as px

iris = px.data.iris()

fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',

              color='species')

fig.show()
import plotly.express as px

iris = px.data.iris()

fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',

                    color='petal_length', symbol='species')

fig.show()