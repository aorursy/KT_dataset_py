import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.svm import SVC
from yellowbrick.features import Rank2D
#3D plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot;
import cufflinks as cf; init_notebook_mode(connected = True); cf.go_offline()
import plotly.graph_objs as go

#sklearn
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

#Loading dataset
wine = pd.read_csv('../input/winequality-red.csv');wine.head()

wine.info()
wine.describe()
def grading(row):
    if row['quality'] > 6:
        return 1
    else:
        return 0

 #1 = worth buying, 0 = not worth buying   
    
wine['grade'] = wine.apply(grading, axis=1)
wine.head()

g = sns.FacetGrid(wine, col="grade",  hue="quality")
g = (g.map(plt.scatter, "alcohol", "quality", edgecolor="w").add_legend())

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Wine Grade Heatmap")
corr = wine.corr() #heat map only take 2d array
#insert mask
# mask = np.zeros_like(corr)
# mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): sns.heatmap(corr, mask=None, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

#volatile, acidity, alcholhol 
sns.pairplot(wine)

trace1 = go.Scatter3d(
    x=wine["volatile acidity"],
    y=wine["alcohol"],
    z=wine["quality"],
    mode='markers',
    text=wine["total sulfur dioxide"],
    marker=dict(
        size=12,
        color=wine["pH"],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    title="Wine Quality",
    scene = dict(
        xaxis = dict(title='X: volatile acidity'),
        yaxis = dict(title="Y: alcohol"),
        zaxis = dict(title="Z: quality"),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

modified_df = wine[["grade", "alcohol", "volatile acidity"]]; modified_df.head()
y = modified_df.grade
X = modified_df[["grade", "alcohol", "volatile acidity"]]
classes = ["good","bad"]
# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from yellowbrick.classifier import ROCAUC
from sklearn.linear_model import LogisticRegression

# Instantiate the visualizer with the classification model
visualizer = ROCAUC(LogisticRegression(), classes=classes)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data



g = visualizer.poof()






