import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py

import plotly.graph_objs as go

from sklearn.preprocessing import MinMaxScaler
Churn_train = pd.read_csv(r"../input/traindata/train.csv")

Churn_train.head()
Churn_test = pd.read_csv(r"../input/testdata/test.csv")

Churn_test.head()
print ("Dataset  : TRAINING set ")

print ("Rows     : " ,Churn_train.shape[0])

print ("Columns  : " ,Churn_train.shape[1])

print ("\nFeatures : \n" ,Churn_train.columns.tolist())

print ("\nTotal Missing values :  ", Churn_train.isnull().sum().values.sum())

print ("\nUnique values :  \n",Churn_train.nunique())

Churn_train.head(3)
print ("Dataset  : TESTING set ")

print ("Rows     : " ,Churn_test.shape[0])

print ("Columns  : " ,Churn_test.shape[1])

print ("\nFeatures : \n" ,Churn_test.columns.tolist())

print ("\nTotal Missing values :  ", Churn_test.isnull().sum().values.sum())

print ("\nUnique values :  \n",Churn_test.nunique())

Churn_test.head(3)
print("Quantity of missing values per variable in Train and Test set")

print("\n")

print(Churn_train.isna().sum())

print("\n")

print(Churn_test.isna().sum())

print("\n")

print("Percentage of missing values per variable in Train and Test set")

print("\n")

print(Churn_train.isna().sum()/len(Churn_train)*100)

print("\n")

Churn_train.isna().sum()/len(Churn_train)*100
print("Duplicate records in Training: ", Churn_train[Churn_train.duplicated(keep=False)])

print("\n")

print("Duplicate records in Training: ", Churn_test[Churn_test.duplicated(keep=False)])
Churn_train_ID=Churn_train['Employee_ID']

print("Duplicate IDs: ", Churn_train_ID[Churn_train_ID.duplicated(keep=False)])
Churn_train.set_index('Employee_ID', inplace=True)

Churn_train.head(3)
Churn_test.set_index('Employee_ID', inplace=True)

Churn_test.head(3)
Churn_train.info()
Churn_train.describe(include=['O']).T
print(Churn_train['Gender'].value_counts())

print('\n')

print(Churn_train['Marital_status'].value_counts())

print('\n')

print(Churn_train['Department'].value_counts())

print('\n')

Churn_train['Churn_risk'].value_counts()
#labels

lab = Churn_train["Churn_risk"].value_counts().keys().tolist()

#values

val = Churn_train["Churn_risk"].value_counts().values.tolist()



trace = go.Pie(labels = lab ,

               values = val ,

               marker = dict(colors =  [ 'cornflowerlblue' ,'lightblue', 'red'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Employee Churn_risk in data",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
df = Churn_train.copy()

Id_col     = ['Employee_ID']



summary = (df[[i for i in df.columns if i not in Id_col]].

           describe().transpose().reset_index())



summary = summary.rename(columns = {"index" : "feature"})

summary = np.around(summary,3)



val_lst = [summary['feature'], summary['count'],

           summary['mean'],summary['std'],

           summary['min'], summary['25%'],

           summary['50%'], summary['75%'], summary['max']]



trace  = go.Table(header = dict(values = summary.columns.tolist(),

                                line = dict(color = ['white']),

                                fill = dict(color = ['steelblue']),

                               ),

                  cells  = dict(values = val_lst,

                                line = dict(color = ['white']),

                                fill = dict(color = ["yellow",'lavender'])

                               ),

                  columnwidth = [200,60,100,100,60,60,80,80,80])

layout = go.Layout(dict(title = "Numerical Variable Summary"))

figure = go.Figure(data=[trace],layout=layout)

py.iplot(figure)
#Binning variables

def overtime_lab(df) :

    if df["Overtime"] <= 9.5 :

        return "Overtime_0-9.5"

    elif (df["Overtime"] > 9.5) & (df["Overtime"] <= 10.3 ):

        return "Overtime_9.5-10.3"

    elif (df["Overtime"] > 10.3) & (df["Overtime"] <= 11.3) :

        return "Overtime_10.3-11.3"

    elif df["Overtime"] > 14.2:

        return "Overtime_gt_14.2"

df["overtime_group"] = df.apply(lambda df:overtime_lab(df), axis = 1)



def tenure_lab(df) :

    if df["Tenure"] <= 8 :

        return "Tenure_0-8"

    elif (df["Tenure"] > 8) & (df["Tenure"] <= 13 ):

        return "Tenure_8-13"

    elif (df["Tenure"] > 13) & (df["Tenure"] <= 17) :

        return "Tenure_13-17"

    elif (df["Tenure"] > 17) & (df["Tenure"] <= 41) :

        return "Tenure_17-41"

    elif df["Tenure"] > 41 :

        return "Tenure_gt_41"

df["tenure_group"] = df.apply(lambda df:tenure_lab(df),axis = 1)



def age_lab(df) :

    if df["Age"] <= 33 :

        return "young_adults"

    elif (df["Age"] > 33) & (df["Age"] <= 74 ):

        return "older_adults"

    elif df["Age"] > 74 :

        return "Age_gt_4"

df["age_group"] = df.apply(lambda df:age_lab(df), axis = 1)

  



#Separating churn and non churn employee

churn_low     = df[df["Churn_risk"] == "low"]

churn_medium  = df[df["Churn_risk"] == "medium"]

churn_high    = df[df["Churn_risk"] == "high"]



#Separating catagorical and numerical columns

Id_col     = ['Employee_ID']

target_col = ["Churn_risk"]

cat_cols   = df.nunique()[df.nunique() < 8].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

num_cols   = [x for x in df.columns if x not in cat_cols + target_col + Id_col]
#employee churn in age

tg_low  =  churn_low["age_group"].value_counts().reset_index()

tg_low.columns  = ["age_group","count"]

tg_medium =  churn_medium["age_group"].value_counts().reset_index()

tg_medium.columns = ["age_group","count"]

tg_high =  churn_high["age_group"].value_counts().reset_index()

tg_high.columns = ["age_group","count"]



#bar - low churn

trace1 = go.Bar(x = tg_low["age_group"]  , y = tg_low["count"],

                name = "Low Churn Employee",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='lightblue',

                opacity = .9)



#bar - medium churn

trace2 = go.Bar(x = tg_medium["age_group"] , y = tg_medium["count"],

                name = "Medium Churn Employee",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='cornflowerblue',

                opacity = .9)



#bar - high churn

trace3 = go.Bar(x = tg_high["age_group"] , y = tg_high["count"],

                name = "High Churn Employee",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='red',

                opacity = .9)



layout = go.Layout(dict(title = "Employee churn in age_group",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "age_group",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "count",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1, trace2, trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
#labels

lab = df["age_group"].value_counts().keys().tolist()

#values

val = df["age_group"].value_counts().values.tolist()



trace = go.Pie(labels = lab ,

               values = val ,

               marker = dict(colors =  [ 'lightcoral', 'maroon'],

                             line = dict(color = "white",

                                         width =  1.3)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Employee age_group in data",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )



data = [trace]

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
#employee churn in Kids

tg_low  =  churn_low["Kids"].value_counts().reset_index()

tg_low.columns  = ["Kids","count"]

tg_medium =  churn_medium["Kids"].value_counts().reset_index()

tg_medium.columns = ["Kids","count"]

tg_high =  churn_high["Kids"].value_counts().reset_index()

tg_high.columns = ["Kids","count"]



#bar - low churn

trace1 = go.Bar(x = tg_low["Kids"]  , y = tg_low["count"],

                name = "Low Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='lightblue',

                opacity = .9)



#bar - medium churn

trace2 = go.Bar(x = tg_medium["Kids"] , y = tg_medium["count"],

                name = "Medium Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='cornflowerblue',

                opacity = .9)



#bar - high churn

trace3 = go.Bar(x = tg_high["Kids"] , y = tg_high["count"],

                name = "High Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='red',

                opacity = .9)



layout = go.Layout(dict(title = "Employee churn in Kids",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "Kids",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "count",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1, trace2, trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
#Employee churn in tenure groups

tg_low  =  churn_low["tenure_group"].value_counts().reset_index()

tg_low.columns  = ["tenure_group","count"]

tg_medium =  churn_medium["tenure_group"].value_counts().reset_index()

tg_medium.columns = ["tenure_group","count"]

tg_high =  churn_high["tenure_group"].value_counts().reset_index()

tg_high.columns = ["tenure_group","count"]



#bar - low churn

trace1 = go.Bar(x = tg_low["tenure_group"]  , y = tg_low["count"],

                name = "Low Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='lightblue',

                opacity = .9)



#bar - medium churn

trace2 = go.Bar(x = tg_medium["tenure_group"] , y = tg_medium["count"],

                name = "Medium Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='cornflowerblue',

                opacity = .9)



#bar - high churn

trace3 = go.Bar(x = tg_high["tenure_group"] , y = tg_high["count"],

                name = "High Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='red',

                opacity = .9)



layout = go.Layout(dict(title = "Employee churn in tenure groups",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "tenure group",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "count",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1, trace2, trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
# Overtime by Churn Risk

tg_low  =  churn_low["overtime_group"].value_counts().reset_index()

tg_low.columns  = ["overtime_group","count"]

tg_medium =  churn_medium["overtime_group"].value_counts().reset_index()

tg_medium.columns = ["overtime_group","count"]

tg_high =  churn_high["overtime_group"].value_counts().reset_index()

tg_high.columns = ["overtime_group","count"]



#bar - low churn

trace1 = go.Bar(x = tg_low["overtime_group"]  , y = tg_low["count"],

                name = "Low Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='lightblue',

                opacity = .9)



#bar - medium churn

trace2 = go.Bar(x = tg_medium["overtime_group"] , y = tg_medium["count"],

                name = "Medium Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                 marker_color='cornflowerblue',

                opacity = .9)



#bar - high churn

trace3 = go.Bar(x = tg_high["overtime_group"] , y = tg_high["count"],

                name = "High Churn Employees",

                marker = dict(line = dict(width = .5,color = "black")),

                marker_color='red',

                opacity = .9)



layout = go.Layout(dict(title = "Employee overtime_group",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "tenure group",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "count",

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                       )

                  )

data = [trace1, trace2, trace3]

fig  = go.Figure(data=data,layout=layout)

py.iplot(fig)
# all numerical varibles by Churn_risk

def histogram(column) :

    trace1 = go.Histogram(x  = churn_low[column],

                          histnorm= "percent",

                          name = "Low Churn Employees",

                          marker = dict(line = dict(width = .5,color = "black")),

                          marker_color='lightblue',

                          opacity = .9) 

    

    trace2 = go.Histogram(x  = churn_medium[column],

                          histnorm = "percent",

                          name = "Medium Churn Employees",

                          marker = dict(line = dict(width = .5,color = "black")),

                          marker_color='cornflowerblue',

                          opacity = .9)

                         

    trace3 = go.Histogram(x  = churn_high[column],

                          histnorm = "percent" ,

                          name = "High Churn Employees",

                          marker = dict(line = dict(width = .5,color = "black")),

                          marker_color='red',

                          opacity = .9)

    

    data = [trace1,trace2, trace3]

    layout = go.Layout(dict(title =column + " distribution in employee churn",

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = column,

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=3

                                            ),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                             title = "percent",

                                             zerolinewidth=1,

                                             ticklen=5,

                                             gridwidth=3

                                            ),

                           )

                      )

    fig  = go.Figure(data=data,layout=layout)

    

    py.iplot(fig)

    

#for all categorical columns plot histogram    

for i in num_cols :

    histogram(i)
import plotly.figure_factory as ff

from plotly.offline import iplot





# prepare data



datascatter = Churn_train[["Age","Days_off", "Emails", 'Tenure']]

datascatter["index"] = np.arange(1,len(datascatter)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(datascatter, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
# Scatterplot by emails & bonus by Churn Risk

Churn_train[['Bonus', 'Overtime']]





def plot_Churn_risk_scatter(Churn_risk,color) :

    tracer = go.Scatter(x = Churn_train[Churn_train["Churn_risk"] == Churn_risk]["Bonus"],

                        y = Churn_train[Churn_train["Churn_risk"] == Churn_risk]["Overtime"],

                        mode = "markers",marker = dict(line = dict(color = "black",

                                                                   width = .2),

                                                       size = 4 , color = color,

                                                       symbol = "diamond-dot",

                                                      ),

                        name = "Employee Churn - " + Churn_risk,

                        opacity = .9

                       )

    return tracer





trace1 = plot_Churn_risk_scatter("low","lightblue")

trace2 = plot_Churn_risk_scatter("medium","cornflowerblue")

trace3 = plot_Churn_risk_scatter("high","red")



data   = [trace1,trace2,trace3] 



#layout

def layout_title(title) :

    layout = go.Layout(dict(title = title,

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "Bonus",

                                         zerolinewidth=1,ticklen=5,gridwidth=2),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "Overtime",

                                         zerolinewidth=1,ticklen=5,gridwidth=2),

                            height = 600

                           )

                      )

    return layout



layout  = layout_title("Bonus & Overtime by Churn_risk")

fig = go.Figure(data = data,layout = layout)

py.iplot(fig)
#Emails, Days_Off and Tenure by Churn Risk



churn_df = Churn_train.copy()

#Drop tenure column



trace1 = go.Scatter3d(x = churn_low["Emails"],

                      y = churn_low["Days_off"],

                      z = churn_low["Tenure"],

                      mode = "markers",

                      name = "Low Churn Employees",

                      marker = dict(size = 1,color = "lightblue")

                     )

trace2 = go.Scatter3d(x = churn_medium["Emails"],

                      y = churn_medium["Days_off"],

                      z = churn_medium["Tenure"],

                      name = "Medium Churn Employees",

                      mode = "markers",

                      marker = dict(size = 1,color= "cornflowerblue")

                     )

trace3 = go.Scatter3d(x = churn_high["Emails"],

                      y = churn_high["Days_off"],

                      z = churn_high["Tenure"],

                      name = "High Churn Employees",

                      mode = "markers",

                      marker = dict(size = 1,color= "red")

                     )







layout = go.Layout(dict(title = "Emails, Days_off & Tenure by Churn_risk",

                        scene = dict(camera = dict(up=dict(x= 0 , y=0, z=0),

                                                   center=dict(x=0, y=0, z=0),

                                                   eye=dict(x=1.25, y=1.25, z=1.25)),

                                     xaxis  = dict(title = "Emails",

                                                   gridcolor='rgb(255, 255, 255)',

                                                   zerolinecolor='rgb(255, 255, 255)',

                                                   showbackground=True,

                                                   backgroundcolor='rgb(230, 230,230)'),

                                     yaxis  = dict(title = "Days_off",

                                                   gridcolor='rgb(255, 255, 255)',

                                                   zerolinecolor='rgb(255, 255, 255)',

                                                   showbackground=True,

                                                   backgroundcolor='rgb(230, 230,230)'

                                                  ),

                                     zaxis  = dict(title = "Tenure",

                                                   gridcolor='rgb(255, 255, 255)',

                                                   zerolinecolor='rgb(255, 255, 255)',

                                                   showbackground=True,

                                                   backgroundcolor='rgb(230, 230,230)'

                                                  )

                                    ),

                        height = 700,

                       )

                  )

                  



data = [trace1,trace2,trace3]

fig  = go.Figure(data = data,layout = layout)

py.iplot(fig)
import plotly.express as px



def check_outliers(column):

    '''

    This function will show the outliers for a column

    '''

    print(column.describe())

    p1 = column.quantile(q=0.01)

    p25 = column.quantile(q=0.25)

    p75 = column.quantile(q=0.75)

    p99 = column.quantile(q=0.99)

    iqr = p75 - p25

    mu = np.mean(column)

    sigma = np.std(column)

    print('Variable:', column.name)

    print('\n')

    print('Percentile outliers: ')

    print('Valid interval: [', p1, ', ', p99, ']')

    print('Number of lower outliers: ', np.sum(column < p1))

    print('Number of upper outliers: ', np.sum(column > p99))

    print('Total Percentile Outliers: ', np.sum(column > p99)+np.sum(column < p1),

          " (", round(((np.sum(column > p99)+np.sum(column < p1))/5200*100),2), "%)")

    print('\n')

    print('Tukey’s fence outliers: ')

    print('Valid interval: [', p25 - 1.5*iqr, ', ', p75 + 1.5*iqr, ']')

    print('Number of lower outliers: ', np.sum(column < p25 - 1.5*iqr))

    print('Number of upper outliers: ', np.sum(column > p75 + 1.5*iqr))

    print('Total Tueky Fences Outliers: ', np.sum(column < p25 - 1.5*iqr)+np.sum(column > p75 + 1.5*iqr),

         " (", round(((np.sum(column < p25 - 1.5*iqr)+np.sum(column > p75 + 1.5*iqr))/5200*100),2), "%)")

    print('\n')

    print('Standard deviation outliers: ')

    print('Valid interval: [', mu - 3*sigma, ', ', mu + 3*sigma, ']')

    print('Number of lower outliers: ', np.sum(column < mu - 3*sigma))

    print('Number of upper outliers: ', np.sum(column > mu + 3*sigma))

    print('Total STD Outliers: ', np.sum(column < mu - 3*sigma)+np.sum(column > mu + 3*sigma),

          " (", round(((np.sum(column < mu - 3*sigma)+np.sum(column > mu + 3*sigma))/5200*100),2), "%)")

    print('\n')
check_outliers(Churn_train['Age'])

fig = px.box(Churn_train, y="Age", points="all", color_discrete_sequence =['lawngreen'])

fig.show()
check_outliers(Churn_train['Days_off'])

fig = px.box(Churn_train, y="Days_off", points="all", color_discrete_sequence =['lightgreen'])

fig.show()
check_outliers(Churn_train['Rotations'])

fig = px.box(Churn_train, y="Rotations", points="all", color_discrete_sequence =['lime'])

fig.show()
check_outliers(Churn_train['Satis_leader'])

fig = px.box(Churn_train, y="Satis_leader", points="all", color_discrete_sequence =['springgreen'])

fig.show()
check_outliers(Churn_train['Satis_team'])

fig = px.box(Churn_train, y="Satis_team", points="all", color_discrete_sequence =['darkseagreen'])

fig.show()
check_outliers(Churn_train['Emails'])

fig = px.box(Churn_train, y="Emails", points="all", color_discrete_sequence =['turquoise'])

fig.show()
check_outliers(Churn_train['Tenure'])

fig = px.box(Churn_train, y="Tenure", points="all", color_discrete_sequence =['teal'])

fig.show()
check_outliers(Churn_train['Bonus'])

fig = px.box(Churn_train, y="Bonus", points="all", color_discrete_sequence =['green'])

fig.show()
check_outliers(Churn_train['Distance'])

fig = px.box(Churn_train, y="Distance", points="all", color_discrete_sequence =['forestgreen'])

fig.show()
check_outliers(Churn_train['Kids'])

fig = px.box(Churn_train, y="Kids", points="all", color_discrete_sequence =['darkgreen'])

fig.show()
check_outliers(Churn_train['Overtime'])

fig = px.box(Churn_train, y="Overtime", points="all", color_discrete_sequence =['olive'])

fig.show()
#correlation

correlation = Churn_train.corr()

#tick labels

matrix_cols = correlation.columns.tolist()

#convert to array

corr_array  = np.array(correlation)



#Plotting

trace = go.Heatmap(z = corr_array,

                   x = matrix_cols,

                   y = matrix_cols,

                   colorscale = "greens"

                   )





layout = go.Layout(dict(title = "Pearson Correlation Matrix for numerical Variables",

                        autosize = False,

                        height  = 720,

                        width   = 800,

                        margin  = dict(r = 0 ,l = 210,

                                       t = 25,b = 210,

                                      ),

                        yaxis   = dict(tickfont = dict(size = 9)),

                        xaxis   = dict(tickfont = dict(size = 9))

                       )

                  )



data = [trace]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)                
Churn_train_median = Churn_train.copy()

Churn_train_median.fillna(Churn_train_median.median(), inplace = True)
Churn_train_median_test = Churn_test.copy()

Churn_train_median_test.fillna(Churn_train_median.median(), inplace = True)
print(Churn_train_median.isna().sum())

Churn_train_median_test.isna().sum()
#Subgroups grouped by Gender, Marital Status & Department - Median - Train Set

med_age = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Age'].transform('median')

med_days = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Days_off'].transform('median')

med_rot = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Rotations'].transform('median')

med_ld = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Satis_leader'].transform('median')

med_tm = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Satis_team'].transform('median')

med_dist = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Distance'].transform('median')

med_kids = Churn_train.groupby(['Gender', 'Marital_status', 'Department'])['Kids'].transform('median')

# Missing values filled - Train set

Churn_train['Age'].fillna(med_age, inplace=True)

Churn_train['Days_off'].fillna(med_days, inplace=True)

Churn_train['Rotations'].fillna(med_rot, inplace=True)

Churn_train['Satis_leader'].fillna(med_ld, inplace=True)

Churn_train['Satis_team'].fillna(med_tm, inplace=True)

Churn_train['Distance'].fillna(med_dist, inplace=True)

Churn_train['Kids'].fillna(med_kids, inplace=True)

# Missing values filled - Test Set

Churn_test['Age'].fillna(med_age, inplace=True)

Churn_test['Days_off'].fillna(med_days, inplace=True)

Churn_test['Rotations'].fillna(med_rot, inplace=True)

Churn_test['Satis_team'].fillna(med_tm, inplace=True)

# Subgroups for persisting missing values

med_age2 = Churn_train.groupby(['Gender', 'Marital_status'])['Age'].transform('median')

med_dist2 = Churn_train.groupby(['Gender', 'Marital_status'])['Distance'].transform('median')

med_tm2 = Churn_train.groupby(['Gender', 'Marital_status'])['Satis_team'].transform('median')

# Missing values filled - Train set

Churn_train['Age'].fillna(med_age2, inplace=True)

Churn_train['Distance'].fillna(med_dist2, inplace=True)

# Missing values filled - Test set

Churn_test.fillna(Churn_train.median(), inplace = True)
print(Churn_train.isna().sum())

Churn_test.isna().sum()
Churn_train = pd.concat([Churn_train, pd.get_dummies(Churn_train['Marital_status'])], axis=1)

Churn_train = pd.concat([Churn_train, pd.get_dummies(Churn_train['Gender'])], axis=1)

Churn_train = pd.concat([Churn_train, pd.get_dummies(Churn_train['Department'])], axis=1)
print(Churn_train.columns)

print(Churn_train.shape)

Churn_train.head(3)
Churn_test = pd.concat([Churn_test, pd.get_dummies(Churn_test['Marital_status'])], axis=1)

Churn_test = pd.concat([Churn_test, pd.get_dummies(Churn_test['Gender'])], axis=1)

Churn_test = pd.concat([Churn_test, pd.get_dummies(Churn_test['Department'])], axis=1)
print(Churn_test.columns)

print(Churn_test.shape)

Churn_test.head(3)
# Variable creation for the train set

Churn_train['Tenure_Age'] = (Churn_train['Tenure']/12)/Churn_train['Age']

Churn_train['Rotations_Tenure']=Churn_train['Rotations']/Churn_train['Tenure']

Churn_train['Distance_Overtime']=Churn_train['Distance']*Churn_train['Overtime']

Churn_train['Overtime_Bonus']=Churn_train['Overtime']/np.where(Churn_train['Bonus']>0,Churn_train['Bonus'],1)



# Variable creation for the test set

Churn_test['Tenure_Age'] = (Churn_test['Tenure']/12)/Churn_test['Age']

Churn_test['Rotations_Tenure']=Churn_test['Rotations']/Churn_test['Tenure']

Churn_test['Distance_Overtime']=Churn_test['Distance']*Churn_test['Overtime']

Churn_test['Overtime_Bonus']=Churn_test['Overtime']/np.where(Churn_test['Bonus']>0,Churn_test['Bonus'],1)
print("Quantity of missing values per variable in Train and Test set - after new variables")

print("\n")

print(Churn_train.isna().sum())

print("\n")

print(Churn_test.isna().sum())

print("\n")
check_outliers(Churn_train['Tenure_Age'])

fig = px.box(Churn_train, y="Tenure_Age", points="all", color_discrete_sequence =['lightsalmon'])

fig.show()
check_outliers(Churn_train['Rotations_Tenure'])

fig = px.box(Churn_train, y="Rotations_Tenure", points="all", color_discrete_sequence =['lightcoral'])

fig.show()
check_outliers(Churn_train['Distance_Overtime'])

fig = px.box(Churn_train, y="Distance_Overtime", points="all", color_discrete_sequence =['indianred'])

fig.show()
check_outliers(Churn_train['Overtime_Bonus'])

fig = px.box(Churn_train, y="Overtime_Bonus", points="all", color_discrete_sequence =['darkred'])

fig.show()
data = Churn_train.drop(['Churn_risk', 'Gender','Marital_status', 'Department'], axis=1)

print("Final dataframe shape: ", data.shape)

print("\n")

print("Independent Variables that could be used for modelling:")

print("\n")

data.info()
Churn_train_risk = Churn_train[['Churn_risk']].copy()

Churn_train_risk.head()
dataTest = Churn_test.drop(['Gender','Marital_status', 'Department'], axis=1)

print("Final dataframe shape: ", dataTest.shape)

print("\n")

print("Independent Variables that could be used for predictions:")

print("\n")

dataTest.info()
Data_clip_perc = data.copy()

LowP = Data_clip_perc.quantile(0.01)

HighP = Data_clip_perc.quantile(0.99)

Data_clip_perc = Data_clip_perc.clip(LowP, HighP, axis=1)

Data_clip_perc.describe().T
Data_clip_percTest = dataTest.copy()

Data_clip_percTest = Data_clip_percTest.clip(LowP, HighP, axis=1)

Data_clip_percTest.describe().T
# Min Max applied to data without removing outliers

Data_with_outliers = data.copy()

Data_with_outliersMinMaxInstance = MinMaxScaler().fit(Data_with_outliers)

Data_with_outliersMinMax = Data_with_outliersMinMaxInstance.transform(Data_with_outliers)

Data_with_outliersMinMax
# Min Max applied to data with outliers removed by percentile method

Data_clip_percMinMaxInstance = MinMaxScaler().fit(Data_clip_perc)

Data_clip_percMinMax = Data_clip_percMinMaxInstance.transform(Data_clip_perc)

Data_clip_percMinMax
# Min Max applied to target without removing outliers

Data_with_outliersTest = dataTest.copy()

Data_with_outliersMinMaxTest = Data_with_outliersMinMaxInstance.transform(Data_with_outliersTest)

Data_with_outliersMinMaxTest
# Min Max applied to target with outliers removed by percentile method

Data_clip_percMinMaxTest = Data_clip_percMinMaxInstance.transform(Data_clip_percTest)

Data_clip_percMinMaxTest
Churn_train = pd.read_csv(r"../input/traindata/train.csv")

Churn_train.set_index('Employee_ID', inplace=True)
Churn_train_risk = Churn_train[['Churn_risk']].copy()

Churn_train.drop('Churn_risk', axis=1, inplace =True)
Churn_train_median = Churn_train.copy()

Churn_train_median.fillna(Churn_train_median.median(), inplace = True)

Churn_train_median_class = Churn_train.copy()



med_age = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Age'].transform('median')

med_days = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Days_off'].transform('median')

med_rot = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Rotations'].transform('median')

med_ld = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Satis_leader'].transform('median')

med_tm = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Satis_team'].transform('median')

med_dist = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Distance'].transform('median')

med_kids = Churn_train_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Kids'].transform('median')



Churn_train_median_class['Age'].fillna(med_age, inplace=True)

Churn_train_median_class['Days_off'].fillna(med_days, inplace=True)

Churn_train_median_class['Rotations'].fillna(med_rot, inplace=True)

Churn_train_median_class['Satis_leader'].fillna(med_ld, inplace=True)

Churn_train_median_class['Satis_team'].fillna(med_tm, inplace=True)

Churn_train_median_class['Distance'].fillna(med_dist, inplace=True)

Churn_train_median_class['Kids'].fillna(med_kids, inplace=True)



med_age2 = Churn_train_median_class.groupby(['Gender', 'Marital_status'])['Age'].transform('median')

med_dist2 = Churn_train_median_class.groupby(['Gender', 'Marital_status'])['Distance'].transform('median')



Churn_train_median_class['Age'].fillna(med_age2, inplace=True)

Churn_train_median_class['Distance'].fillna(med_dist2, inplace=True)
Churn_train_median_nv = Churn_train_median.copy()

Churn_train_median_nv['Tenure_Age']=(Churn_train_median_nv['Tenure']/12/Churn_train_median_nv['Age']).copy()

Churn_train_median_nv['Rotations_Tenure']=(Churn_train_median_nv['Tenure']/(Churn_train_median_nv['Rotations']+1)).copy()

Churn_train_median_nv['Distance_Overtime']=(Churn_train_median_nv['Distance']*Churn_train_median_nv['Overtime']).copy()

Churn_train_median_nv['Overtime_Bonus']=(Churn_train_median_nv['Overtime']/np.where(Churn_train_median_nv['Bonus']>0,Churn_train_median_nv['Bonus'],1)).copy()



Churn_train_median_class_nv = Churn_train_median_class.copy()

Churn_train_median_class_nv['Tenure_Age']=(Churn_train_median_class_nv['Tenure']/12/Churn_train_median_class_nv['Age']).copy()

Churn_train_median_class_nv['Rotations_Tenure']=(Churn_train_median_class_nv['Tenure']/(Churn_train_median_class_nv['Rotations']+1)).copy()

Churn_train_median_class_nv['Distance_Overtime']=(Churn_train_median_class_nv['Distance']*Churn_train_median_class_nv['Overtime']).copy()

Churn_train_median_class_nv['Overtime_Bonus']=(Churn_train_median_class_nv['Overtime']/np.where(Churn_train_median_class_nv['Bonus']>0,Churn_train_median_class_nv['Bonus'],1)).copy()
Churn_train_median_dummies = pd.get_dummies(Churn_train_median)

Churn_train_median_class_dummies = pd.get_dummies(Churn_train_median_class)

Churn_train_median_nv_dummies = pd.get_dummies(Churn_train_median_nv)

Churn_train_median_class_nv_dummies = pd.get_dummies(Churn_train_median_class_nv)
Low_median = Churn_train_median_dummies.quantile(0.01)

High_median = Churn_train_median_dummies.quantile(0.99)

Churn_train_median_dummies_pct = Churn_train_median_dummies.clip(Low_median, High_median, axis=1)



Low_median_class = Churn_train_median_class_dummies.quantile(0.01)

High_median_class = Churn_train_median_class_dummies.quantile(0.99)

Churn_train_median_class_dummies_pct = Churn_train_median_class_dummies.clip(Low_median_class, High_median_class, axis=1)



Low_nv_median = Churn_train_median_nv_dummies.quantile(0.01)

High_nv_median = Churn_train_median_nv_dummies.quantile(0.99)

Churn_train_median_nv_dummies_pct = Churn_train_median_nv_dummies.clip(Low_nv_median, High_nv_median, axis=1)



Low_nv_median_class = Churn_train_median_class_nv_dummies.quantile(0.01)

High_nv_median_class = Churn_train_median_class_nv_dummies.quantile(0.99)

Churn_train_median_class_nv_dummies_pct = Churn_train_median_class_nv_dummies.clip(Low_nv_median_class, High_nv_median_class, axis=1)
Churn_train_median_dummies_01 = MinMaxScaler().fit(Churn_train_median_dummies).transform(Churn_train_median_dummies)

Churn_train_median_dummies_pct_01 = MinMaxScaler().fit(Churn_train_median_dummies_pct).transform(Churn_train_median_dummies_pct)

Churn_train_median_class_dummies_01 = MinMaxScaler().fit(Churn_train_median_class_dummies).transform(Churn_train_median_class_dummies)

Churn_train_median_class_dummies_pct_01 = MinMaxScaler().fit(Churn_train_median_class_dummies_pct).transform(Churn_train_median_class_dummies_pct)

Churn_train_median_nv_dummies_01 = MinMaxScaler().fit(Churn_train_median_nv_dummies).transform(Churn_train_median_nv_dummies)

Churn_train_median_nv_dummies_pct_01 = MinMaxScaler().fit(Churn_train_median_nv_dummies_pct).transform(Churn_train_median_nv_dummies_pct)

Churn_train_median_class_nv_dummies_01 = MinMaxScaler().fit(Churn_train_median_class_nv_dummies).transform(Churn_train_median_class_nv_dummies)

Churn_train_median_class_nv_dummies_pct_01 = MinMaxScaler().fit(Churn_train_median_class_nv_dummies_pct).transform(Churn_train_median_class_nv_dummies_pct)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

import time

import warnings

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
scoring =  make_scorer(f1_score,average='micro')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LogisticRegression()

max_iter = [10, 100, 300]

multi_class = ['ovr', 'multinomial']

solver = ['newton-cg', 'lbfgs', 'sag', 'saga']

# define grid search

grid = dict(max_iter=max_iter, multi_class=multi_class, solver=solver)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_dummies_01,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

# FINAL1 features = train.drop(['id', 'target'],axis=1).columns.values

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = GaussianNB()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

cross_val_score(model,Churn_train_median_dummies_01,Churn_train_risk, cv=cv, scoring=scoring).mean()
model = DecisionTreeClassifier()

criterion=['gini','entropy']

max_features = ['sqrt', 'log2']

max_depth=[2,60,150]

max_leaf_nodes=[6,160,600]

# define grid search

grid = dict(criterion=criterion, max_features=max_features, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_dummies_01,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = RandomForestClassifier()

n_estimators=[10,50,100,200]

max_features = ['sqrt','log2']

bootstrap = [True,False]

# define grid search

grid = dict(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_dummies_01,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
cross_val_score(ExtraTreesClassifier(max_features= 'log2', n_estimators= 200, bootstrap= False),Churn_train_median_dummies_01,Churn_train_risk, cv=cv, scoring=scoring).mean()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Churn_train_median_dummies, Churn_train_risk, test_size = 0.2, random_state=5, stratify = Churn_train_risk)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)

X_sm, y_sm = sm.fit_resample(X_train, y_train)
model = RandomForestClassifier(max_features= 'log2', n_estimators= 200, bootstrap= False).fit(X_sm,y_sm) 

y_pred = model.predict(X_test)

# how did our model perform?

count_misclassified = (y_test['Churn_risk'] != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

from sklearn import metrics

print(f1_score(y_test['Churn_risk'], y_pred, average='micro'))
from imblearn.over_sampling import ADASYN

ada = ADASYN(random_state=0)

X_ada, y_ada = ada.fit_resample(X_train, y_train)
model = RandomForestClassifier(max_features= 'log2', n_estimators= 200, bootstrap= False).fit(X_ada,y_ada) 

y_pred = model.predict(X_test)

# how did our model perform?

count_misclassified = (y_test['Churn_risk'] != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

from sklearn import metrics

print(f1_score(y_test['Churn_risk'], y_pred, average='micro'))
model = SVC()

kernel=['linear', 'poly', 'rbf'] 

decision_function_shape=['ovo', 'ovr']

# define grid search

grid = dict( kernel = kernel,decision_function_shape=decision_function_shape)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_dummies_01,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = KNeighborsClassifier()

n_neighbors=[5,10,15] 



# define grid search

grid = dict( n_neighbors=n_neighbors)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_dummies_01,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = MLPClassifier()

activation= ['identity', 'logistic', 'tanh', 'relu']

solver=['lbfgs', 'sgd', 'adam']



# define grid search

grid = dict( activation=activation, solver=solver)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_class_dummies,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = MLPClassifier()

activation= ['logistic']

solver=['lbfgs']

learning_rate=['constant', 'invscaling', 'adaptive']



# define grid search

grid = dict( activation=activation, solver=solver, learning_rate=learning_rate)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_class_dummies,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
model = MLPClassifier()

activation= ['logistic']

solver=['lbfgs']

learning_rate=['invscaling']

hidden_layer_sizes = [(50),(10,10,10), (100)]



# define grid search

grid = dict( activation=activation, solver=solver, learning_rate=learning_rate, hidden_layer_sizes=hidden_layer_sizes)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring=scoring,error_score=0)

grid_result = grid_search.fit(Churn_train_median_class_dummies,Churn_train_risk)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
from sklearn.ensemble import AdaBoostClassifier

cross_val_score(AdaBoostClassifier(),Churn_train_median_class_dummies,Churn_train_risk, cv=cv, scoring=scoring).mean()
cross_val_score(GradientBoostingClassifier(),Churn_train_median_class_dummies,Churn_train_risk, cv=cv, scoring=scoring).mean()
from sklearn.ensemble import VotingClassifier

clf1=LogisticRegression(max_iter= 10, multi_class= 'multinomial', solver= 'saga')

clf2=DecisionTreeClassifier(criterion= 'gini', max_depth= 60, max_features= 'sqrt', max_leaf_nodes= 600)

clf3=RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False)

clf4=ExtraTreesClassifier(max_features= 'log2', n_estimators= 200, bootstrap= False)

clf5=SVC(probability=True)

clf6= GradientBoostingClassifier()

clf7=MLPClassifier(activation= 'logistic', solver= 'lbfgs',learning_rate= 'invscaling',hidden_layer_sizes= 100)

eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('rf', clf3),('ert',clf4),('svc',clf5),('gb',clf6),('MLP', clf7)],voting='soft')

cross_val_score(eclf, Churn_train_median_class_dummies,Churn_train_risk['Churn_risk'] ,scoring=scoring, cv=10).mean()
model = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state = 1)

cross_val_score(model,Churn_train_median_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
cross_val_score(model,Churn_train_median_class_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
cross_val_score(model,Churn_train_median_nv_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
cross_val_score(model,Churn_train_median_dummies_pct,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
cross_val_score(model,Churn_train_median_dummies_01,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
Churn_test = pd.read_csv(r"../input/testdata/test.csv")
Churn_test.set_index('Employee_ID', inplace=True)
Churn_complete=Churn_train.append(Churn_test)
Churn_complete.fillna(Churn_complete.median(), inplace=True)
Churn_complete_dummies = pd.get_dummies(Churn_complete)

from sklearn.preprocessing import MinMaxScaler

Churn_complete_dummies_01 = pd.DataFrame(MinMaxScaler().fit(Churn_complete_dummies).transform(Churn_complete_dummies), index=list(Churn_complete_dummies.index),columns=list(Churn_complete_dummies))
def plot_feature_importances(model):

    n_features = Churn_train_median_class_dummies.shape[1]

    plt.figure(figsize=(20,10))

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), Churn_train_median_dummies.columns)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.show()

plot_feature_importances

model.fit(Churn_train_median_class_dummies,Churn_train_risk['Churn_risk'])

plot_feature_importances(model)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import make_pipeline



# define feature selection

fs = SelectKBest(score_func=f_classif, k=18)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)



cross_val_score(rf_kbest,Churn_train_median_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import make_pipeline



# define feature selection

fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)



cross_val_score(rf_kbest,Churn_train_median_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.pipeline import make_pipeline



# define feature selection

fs = SelectKBest(score_func=f_classif, k=20)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)



cross_val_score(rf_kbest,Churn_train_median_dummies,Churn_train_risk['Churn_risk'], cv=cv, scoring=scoring).mean()
def avg_score(model):

    # apply kfold

    cv = StratifiedKFold(n_splits=5, random_state=1)

    # create lists to store the results from the different models 

    score_train = []

    score_test = []

    timer = []

   

    for train_index, test_index in cv.split(Churn_train_median_dummies_01,Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = pd.DataFrame(Churn_train_median_dummies_01).iloc[train_index], pd.DataFrame(Churn_train_median_dummies_01).iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        # start counting time

        begin = time.perf_counter()

        # fit the model to the data

        model.fit(X_train, y_train)

        # finish counting time

        end = time.perf_counter()

        # check the mean accuracy for the train

        value_train = f1_score(y_train,model.predict(X_train),average = 'micro')

        # check the mean accuracy for the test

        value_test = f1_score(y_test,model.predict(X_test),average = 'micro')

        # append the accuracies, the time and the number of iterations in the corresponding list

        score_train.append(value_train)

        score_test.append(value_test)

        timer.append(end-begin)

        

    # calculate the average and the std for each measure (accuracy, time and number of iterations)

    avg_time = round(np.mean(timer),3)

    avg_train = round(np.mean(score_train),3)

    avg_test = round(np.mean(score_test),3)

    std_time = round(np.std(timer),2)

    std_train = round(np.std(score_train),2)

    std_test = round(np.std(score_test),2)

        

    return str(avg_time) + '+/-' + str(std_time), str(avg_train) + '+/-' + str(std_train),\

str(avg_test) + '+/-' + str(std_test)
def show_results1(df, *args):

    """

    Receive an empty dataframe and the different models and call the function avg_score

    """

    count = 0

    # for each model passed as argument

    for arg in args:

        # obtain the results provided by avg_score

        time, avg_train, avg_test = avg_score(arg)

        # store the results in the right row

        df.iloc[count] = time, avg_train, avg_test

        count+=1

    return df
model = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['200_log2'])

show_results1(df, model)
depth3 = RandomForestClassifier(max_depth = 3)

depth5 = RandomForestClassifier(max_depth = 5)

depth7 = RandomForestClassifier(max_depth = 7)

leaves10 = RandomForestClassifier(max_leaf_nodes = 10)

leaves20 = RandomForestClassifier(max_leaf_nodes = 20)

leaves30 = RandomForestClassifier(max_leaf_nodes = 30)
df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['depth3','depth5','depth7','leaves10','leaves20','leaves30'])

show_results1(df, depth3,depth5,depth7,leaves10,leaves20,leaves30)
depth12 = RandomForestClassifier(max_depth = 12)

depth15 = RandomForestClassifier(max_depth = 15)

depth17 = RandomForestClassifier(max_depth = 17)
df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['depth12','depth15','depth17'])

show_results1(df, depth12,depth15,depth17)
RFDataset100=Churn_train_risk.copy()

for i in range (100):

    model = RandomForestClassifier(max_depth=10, random_state = 10*i)

    model.fit(Churn_train_median_dummies_01,Churn_train_risk['Churn_risk'])

    for j in range (3):

        RFDataset100['RF'+str(i)+str(j)] = pd.DataFrame(data=model.predict_proba(Churn_train_median_dummies_01)[:,j], index=list(Churn_train_median_dummies.index))

RFDataset100
def avg_score_mlprf(model):

    # apply kfold

    cv = StratifiedKFold(n_splits=5, random_state=1)

    # create lists to store the results from the different models 

    score_train = []

    score_test = []

    timer = []

   

    for train_index, test_index in cv.split(RFDataset100.drop('Churn_risk', axis = 1),Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = RFDataset100.drop('Churn_risk', axis = 1).iloc[train_index], RFDataset100.drop('Churn_risk', axis = 1).iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        # start counting time

        begin = time.perf_counter()

        # fit the model to the data

        model.fit(X_train, y_train)

        # finish counting time

        end = time.perf_counter()

        # check the mean accuracy for the train

        value_train = f1_score(y_train,model.predict(X_train),average = 'micro')

        # check the mean accuracy for the test

        value_test = f1_score(y_test,model.predict(X_test),average = 'micro')

        # append the accuracies, the time and the number of iterations in the corresponding list

        score_train.append(value_train)

        score_test.append(value_test)

        timer.append(end-begin)

        

    # calculate the average and the std for each measure (accuracy, time and number of iterations)

    avg_time = round(np.mean(timer),3)

    avg_train = round(np.mean(score_train),3)

    avg_test = round(np.mean(score_test),3)

    std_time = round(np.std(timer),2)

    std_train = round(np.std(score_train),2)

    std_test = round(np.std(score_test),2)

        

    return str(avg_time) + '+/-' + str(std_time), str(avg_train) + '+/-' + str(std_train),\

str(avg_test) + '+/-' + str(std_test)
def show_results2(df, *args):

    """

    Receive an empty dataframe and the different models and call the function avg_score

    """

    count = 0

    # for each model passed as argument

    for arg in args:

        # obtain the results provided by avg_score

        time, avg_train, avg_test = avg_score_mlprf(arg)

        # store the results in the right row

        df.iloc[count] = time, avg_train, avg_test

        count+=1

    return df
model = MLPClassifier(hidden_layer_sizes=50)

df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['mlp_rf'])

show_results2(df, model)
cv = StratifiedKFold(n_splits=5, random_state=1)

X = Churn_complete_dummies_01[:5200]

columns = list()

model = RandomForestClassifier(max_depth=9)





for i in range (50):

    for j in range (3):

        columns.append('RF'+str(i)+str(j))

Probas = pd.DataFrame(index=list(Churn_train.index),columns=columns)

        

for i in range (50):

    model =  RandomForestClassifier(max_depth=9, random_state=i)

    for train_index, test_index in cv.split(X,Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        model.fit(X_train, y_train)

        for j in range (3):

            Probas['RF'+str(i)+str(j)][list(X_test.index)]=pd.DataFrame(model.predict_proba(X_test), index = list(X_test.index))[j]

        

Probas
def avg_score_mlprf(model):

    # apply kfold

    cv = StratifiedKFold(n_splits=5, random_state=1)

    # create lists to store the results from the different models 

    score_train = []

    score_test = []

    timer = []

   

    for train_index, test_index in cv.split(Probas,Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = Probas.iloc[train_index], Probas.iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        # start counting time

        begin = time.perf_counter()

        # fit the model to the data

        model.fit(X_train, y_train)

        # finish counting time

        end = time.perf_counter()

        # check the mean accuracy for the train

        value_train = f1_score(y_train,model.predict(X_train),average = 'micro')

        # check the mean accuracy for the test

        value_test = f1_score(y_test,model.predict(X_test),average = 'micro')

        # append the accuracies, the time and the number of iterations in the corresponding list

        score_train.append(value_train)

        score_test.append(value_test)

        timer.append(end-begin)

        

    # calculate the average and the std for each measure (accuracy, time and number of iterations)

    avg_time = round(np.mean(timer),3)

    avg_train = round(np.mean(score_train),3)

    avg_test = round(np.mean(score_test),3)

    std_time = round(np.std(timer),2)

    std_train = round(np.std(score_train),2)

    std_test = round(np.std(score_test),2)

        

    return str(avg_time) + '+/-' + str(std_time), str(avg_train) + '+/-' + str(std_train),\

str(avg_test) + '+/-' + str(std_test)
def show_results3(df, *args):

    """

    Receive an empty dataframe and the different models and call the function avg_score

    """

    count = 0

    # for each model passed as argument

    for arg in args:

        # obtain the results provided by avg_score

        time, avg_train, avg_test = avg_score_mlprf(arg)

        # store the results in the right row

        df.iloc[count] = time, avg_train, avg_test

        count+=1

    return df
model = MLPClassifier()

df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['mlp_rf'])

show_results3(df, model)
from sklearn.ensemble import StackingClassifier



# define the base models

level0 = list()

for i in range (10):

    level0.append(('lr'+str(i), RandomForestClassifier(max_depth = 10, random_state = 10*i)))



# define meta learner model

level1 = MLPClassifier()

# define the stacking ensemble

model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['mlp_rf'])

show_results1(df, model)
Churn_test_median_class = Churn_test.copy()

med_age = Churn_test_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Age'].transform('median')

med_days = Churn_test_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Days_off'].transform('median')

med_rot = Churn_test_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Rotations'].transform('median')

med_tm = Churn_test_median_class.groupby(['Gender', 'Marital_status', 'Department'])['Satis_team'].transform('median')

Churn_test_median_class['Age'].fillna(med_age, inplace=True)

Churn_test_median_class['Days_off'].fillna(med_days, inplace=True)

Churn_test_median_class['Rotations'].fillna(med_rot, inplace=True)

Churn_test_median_class['Satis_team'].fillna(med_tm, inplace=True)

Churn_test_median_class_dummies=pd.get_dummies(Churn_test_median_class)

Churn_test_median_class_01 = pd.DataFrame(MinMaxScaler().fit(Churn_train_median_class_dummies).transform(Churn_test_median_class_dummies), index=list(Churn_test_median_class_dummies.index),columns=list(Churn_test_median_class_dummies))
model = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

model.fit(Churn_train_median_class_dummies_01,Churn_train_risk['Churn_risk'])

test_pred= pd.DataFrame(data=model.predict(Churn_test_median_class_01), index=list(Churn_test.index),columns=['Churn_risk'])

test_pred['Churn_risk'].value_counts()
fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)

rf_kbest.fit(Churn_complete_dummies_01[:5200],Churn_train_risk['Churn_risk'])

test_pred= pd.DataFrame(data=rf_kbest.predict(Churn_complete_dummies_01[5200:]), index=list(Churn_test.index),columns=['Churn_risk'])

test_pred['Churn_risk'].value_counts()
sm = SMOTE(random_state=0)

X_sm, y_sm = sm.fit_resample(Churn_complete_dummies_01[:5200],Churn_train_risk)

fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)

rf_kbest.fit(X_sm,y_sm)

test_pred_smote= pd.DataFrame(data=rf_kbest.predict(Churn_complete_dummies_01[5200:]), index=list(Churn_test.index),columns=['Churn_risk'])

test_pred_smote['Churn_risk'].value_counts()
model = RandomForestClassifier(n_estimators=100, max_features ='sqrt', bootstrap = False, random_state=2)

model.fit(Churn_train_median_class_dummies_01,Churn_train_risk['Churn_risk'])

test_pred_f1= pd.DataFrame(data=model.predict(Churn_test_median_class_01), index=list(Churn_test.index),columns=['Churn_risk'])

test_pred_f1['Churn_risk'].value_counts()
test_pred= pd.DataFrame(data=np.where(test_pred_smote['Churn_risk']=='high',test_pred_smote['Churn_risk'],test_pred_f1['Churn_risk']), index=list(Churn_test.index),columns=['Churn_risk'])

test_pred['Churn_risk'].value_counts()
fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

rf_kbest = make_pipeline(fs, clf)

rf_kbest.fit(Churn_complete_dummies_01[:5200],Churn_train_risk['Churn_risk'])

test_proba= pd.DataFrame(data=rf_kbest.predict_proba(Churn_complete_dummies_01[5200:]), index=list(Churn_test.index))

test_proba['Churn_risk']=(pd.DataFrame(data=np.where(test_proba[0]>.25,'high',np.where(test_proba[1]>.45,'low','medium')), index=list(Churn_test.index)))

test_pred=pd.DataFrame(test_proba['Churn_risk'])

test_pred['Churn_risk'].value_counts()
#test_pred.to_csv('Group5_VersionXX.csv')
cv = StratifiedKFold(n_splits=10, random_state=1)

X = Churn_complete_dummies_01[:5200]

sm = SMOTE(random_state=0)

fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

model = make_pipeline(fs, clf)







columns = ['RF_SM0','RF_SM1','RF_SM2','RF0','RF1','RF2']

Probas = pd.DataFrame(index=list(Churn_train.index),columns=columns)

        

for train_index, test_index in cv.split(X,Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        X_sm, y_sm = sm.fit_resample(X_train,y_train)

        model.fit(X_sm,y_sm)

        for i in range (3):

            Probas['RF_SM'+str(i)][list(X_test.index)]=pd.DataFrame(model.predict_proba(X_test), index = list(X_test.index))[i]

for train_index, test_index in cv.split(X,Churn_train_risk['Churn_risk']):

        # get the indexes of the observations assigned for each partition

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = Churn_train_risk['Churn_risk'].iloc[train_index], Churn_train_risk['Churn_risk'].iloc[test_index]

        model.fit(X_train,y_train)

        for i in range (3):

            Probas['RF'+str(i)][list(X_test.index)]=pd.DataFrame(model.predict_proba(X_test), index = list(X_test.index))[i]

        

Probas
model = LogisticRegression()

df = pd.DataFrame(columns = ['Time','Train','Test'], index = ['mlp_rf'])

show_results3(df, model)
X_sm, y_sm = sm.fit_resample(Churn_complete_dummies_01[:5200],Churn_train_risk)



fs = SelectKBest(score_func=f_classif, k=19)

clf = RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False, random_state=1)

model = make_pipeline(fs, clf)

model.fit(X_sm,y_sm)





columns = ['RF_SM0','RF_SM1','RF_SM2','RF0','RF1','RF2']

Probas_test = pd.DataFrame(index=list(Churn_complete_dummies_01[5200:].index),columns=columns)



for i in range (3):

        Probas_test['RF_SM'+str(i)][list(Churn_complete_dummies_01[5200:].index)]=pd.DataFrame(model.predict_proba(Churn_complete_dummies_01[5200:]), index = list(Churn_complete_dummies_01[5200:].index))[i]

model2=make_pipeline(fs, clf)

model2.fit(Churn_complete_dummies_01[:5200],Churn_train_risk)

        

for i in range (3):

        Probas_test['RF'+str(i)][list(Churn_complete_dummies_01[5200:].index)]=pd.DataFrame(model2.predict_proba(Churn_complete_dummies_01[5200:]), index = list(Churn_complete_dummies_01[5200:].index))[i]

Probas_test     
model = LogisticRegression()

model.fit(Probas,Churn_train_risk)

test_pred=pd.DataFrame(data=model.predict(Probas_test), index=list(Churn_test.index),columns=['Churn_risk'])
test_pred= pd.DataFrame(data=test_pred, index=list(Churn_test.index),columns=['Churn_risk'])

test_pred['Churn_risk'].value_counts()
print(__doc__)



import numpy as np

import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from scipy import interp

from sklearn.metrics import roc_auc_score

from sklearn.multiclass import OneVsRestClassifier



# Binarize the output

y = label_binarize(Churn_train_risk, classes=[0, 1, 2])

n_classes = y.shape[1]



#Dataset

Churn_train_risk_dummies = pd.get_dummies(Churn_train_risk)

Churn_numpy = Churn_train_risk_dummies.to_numpy()



# shuffle and split training and test sets

X_train, X_test, y_train, y_test = train_test_split(Probas, Churn_numpy, test_size=.8,

                                                    random_state=0)



# Learn to predict each class against the other



classifier = OneVsRestClassifier(LogisticRegression())

y_score =  classifier.fit(X_train, y_train).predict_proba(X_test)



# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(n_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= n_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves

plt.figure(figsize=(15,10))



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

labels = (['High', 'Low', 'Medium'])

for i, color in zip(range(n_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=2,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(labels[i], roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Churn Risk Classes ROC Curves')

plt.legend(loc="lower right")

plt.show()
# Plot average ROC curves

plt.figure(figsize=(15,10))

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)







plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Average ROC Curves')

plt.legend(loc="lower right")

plt.show()
import plotly.figure_factory as ff



cv = StratifiedKFold(n_splits=5, random_state=1)



#gives model report in dataframe

def model_report(model,train_x,train_y,name) :

    scoring_a =  make_scorer(accuracy_score)

    scoringr=make_scorer(recall_score,average='micro')

    scoringp=make_scorer(precision_score,average='micro')

    scoringf = make_scorer(f1_score,average='micro')

    accuracy     = cross_val_score(model,train_x,train_y, cv=cv, scoring=scoring_a).mean()

    recallscore  = cross_val_score(model,train_x,train_y, cv=cv, scoring=scoringr).mean()

    precision    = cross_val_score(model,train_x,train_y, cv=cv, scoring=scoringp).mean()

    f1score      = cross_val_score(model,train_x,train_y, cv=cv, scoring=scoringf).mean()

    

    

    df = pd.DataFrame({"Model"           : [name],

                       "Accuracy_score"  : [accuracy],

                       "Recall_score"    : [recallscore],

                       "Precision"       : [precision],

                       "f1_score"        : [f1score],

                                             })

    return df



#outputs for every model



clf0 = LogisticRegression()

clf1=RandomForestClassifier(n_estimators=200, max_features ='log2', bootstrap = False)

clf2=ExtraTreesClassifier(max_features= 'log2', n_estimators= 200, bootstrap= False)

clf3= GradientBoostingClassifier()

clf4=MLPClassifier(activation= 'logistic', solver= 'lbfgs',learning_rate= 'invscaling',hidden_layer_sizes= 100)

clf5=SVC(probability=True)

clf6=LogisticRegression(max_iter= 10, multi_class= 'multinomial', solver= 'saga')

clf7=DecisionTreeClassifier(criterion= 'gini', max_depth= 60, max_features= 'sqrt', max_leaf_nodes= 600)



model0 = model_report(clf0,Probas,Churn_train_risk,

                      "Logistic Reg. on top of RFs")

model1 = model_report(clf1,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Random Forest")

model2 = model_report(clf2,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Extra Randomized Trees")

model3 = model_report(clf3,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Gradient Boosting")

model4 = model_report(clf4,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "MLP")

model5 = model_report(clf5,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Support Vector")

model6 = model_report(clf6,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Logistic Regression")

model7 = model_report(clf7,Churn_train_median_class_dummies_01,Churn_train_risk,

                      "Decision Tree")





#concat all models

model_performances = pd.concat([model0,model1,model2,model3,

                                model4,model5,model6,

                                model7],axis = 0).reset_index()



model_performances = model_performances.drop(columns = "index",axis =1)



table  = ff.create_table(np.round(model_performances,4))



py.iplot(table)



model_performances

def output_tracer(metric,color) :

    tracer = go.Bar(y = model_performances["Model"] ,

                    x = model_performances[metric],

                    orientation = "h",name = metric ,

                    marker = dict(line = dict(width =.7),

                                  color = color)

                   )

    return tracer



layout = go.Layout(dict(title = "Model performances",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     title = "metric",

                                     zerolinewidth=1,

                                     ticklen=5,gridwidth=2),

                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                     zerolinewidth=1,ticklen=5,gridwidth=2),

                        margin = dict(l = 250),

                        height = 780

                       )

                  )





trace1  = output_tracer("Accuracy_score","midnightblue")

trace2  = output_tracer('Recall_score',"cornflowerblue")

trace3  = output_tracer('Precision',"lightsteelblue")

trace4  = output_tracer('f1_score',"yellow")



data = [trace1,trace2,trace3,trace4]

fig = go.Figure(data=data,layout=layout)

py.iplot(fig)