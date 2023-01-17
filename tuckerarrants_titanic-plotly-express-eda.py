import pandas as pd
#sneak peek of data

train = pd.read_csv('../input/titanic/train.csv')

train.head()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=train.columns),

                 cells=dict(values=[train[i] for i in train.columns]))

                     ])

fig.show()
import plotly.express as px
fig = px.scatter(train, x='Fare', y='Age', color='Survived', size='Fare')

fig.show()
fig = px.pie(train, names='Survived', title='Passenger Survival')

fig.show()
fig = px.pie(train, names='Survived', title='Passenger Survival', hole=0.4)

fig.show()
fig = go.Figure(data=[go.Pie(labels=train['Embarked'], pull=[.1, .15, .15, 0])])

fig.show()
from plotly.subplots import make_subplots
fig = make_subplots(rows=1, cols=3, specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]])







fig.add_trace(

            go.Pie(labels=train.loc[train['Embarked'] == 'C']['Survived'], pull = [.1, .1],

                   title = 'Embarked C vs. Survived'), row=1, col=1)



fig.add_trace(

            go.Pie(labels=train.loc[train['Embarked'] == 'S']['Survived'], pull = [.07, .07],

                   title = 'Embarked S vs. Survived'),row=1, col=2)



fig.add_trace(

            go.Pie(labels=train.loc[train['Embarked'] == 'Q']['Survived'], pull = [.1, .1],

                   title = 'Embarked Q vs. Survived'), row=1, col=3)





fig.update_layout(height=500, width=800, title_text="Gene Expression Features")

fig.show()
fig = px.histogram(train, x='Age', nbins=30, marginal='box')

fig.show()
fig = px.histogram(train, x='Age', nbins=50, histnorm='probability density')

fig.show()
fig = px.box(train, x='Pclass', y="Age")

fig.show()
fig = px.box(train, x='Pclass', y="Age", points="all")

fig.show()
fig = px.box(train, x='Pclass', y="Age", notched=True, color="Survived")

fig.show()
fig = px.violin(train, y="Age", points="all")

fig.show()
fig = px.violin(train, x='Sex', y="Age", color='Survived', points="all")

fig.show()
fig = px.violin(train, x='Pclass', y="Age", color='Survived', box=True)

fig.show()
fig = px.violin(train, x='Pclass', y="Age", color='Survived', violinmode='overlay')

fig.show()
fig = px.strip(train, x='Pclass', y="Age", color='Survived')

fig.show()
fig = px.strip(train, x='Sex', y="Age", color='Survived', stripmode="overlay")

fig.show()
fig = px.density_heatmap(train, x="Embarked", y="Pclass",

                        height=500, width=500)

fig.show()
fig = px.imshow(train.corr(method='pearson'), 

                title='Correlations Among Training Features',

                height=700, width=700)

fig.show()
fig = px.density_contour(train, x="SibSp", y="Parch",

                         height=400, width=800)

fig.show()
fig = px.density_contour(train, x="SibSp", y="Parch", color='Survived',

                         height=400, width=800)

fig.show()
fig = px.density_contour(train, x="SibSp", y="Parch", color='Survived',

                        height=400, width=800)

fig.update_traces(contours_coloring="fill", contours_showlabels = True)

fig.show()
fig = px.scatter_3d(train, x='Pclass', y='Fare', z='Age',

              color='Survived')

fig.show()
fig = px.scatter_3d(train, x='Pclass', y='Fare', z='Age',

                    color='Survived', symbol='Sex')

fig.show()