import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline



from plotly import tools

import plotly.offline as pyo

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
os.listdir('/kaggle/input')

df = pd.read_excel('/kaggle/input/Customer complaints 2014 through 2019.xlsx', sheet_name='Sheet1')

len(df.columns.values)


pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.options.mode.chained_assignment = None 
df.drop(df.filter(regex=("Ignore this column.*")).columns, axis = 1, inplace = True)

new_header = df.iloc[0]

df = df[1:]

df.columns = new_header

df.loc[:,'Total Cost'] = df['Ref. quantity']*df['Reference number']
df.head(3)
perCodeProcessingTime = df.groupby(['Coding code text'])[['Processing time']].sum().sort_values(by='Processing time',ascending=False )

perCodeProcessingTime['CumulativePercent'] = (perCodeProcessingTime['Processing time'].cumsum()/perCodeProcessingTime['Processing time'].sum())*100

perCodeProcessingTime
trace0 = go.Scatter(x = perCodeProcessingTime.index,y = perCodeProcessingTime['Processing time'],

                   name = "Defect Processing Time")

data = [trace0]

layout = go.Layout(title = "Processing time(SUM) per Different Issue")

fig  = go.Figure(data=data,layout = layout)

iplot(fig,filename = 'file1')
AssemblyRelated = df[df['Coding code text'] == 'ASSEMBLY RELATED'][['Processing time']]                    

MoldingRelated = df[df['Coding code text'] == 'MOLDING RELATED'][['Processing time']]                         

ProcedureNotFollowed = df[df['Coding code text'] == 'PROCEDURE NOT FOLLOWED '][['Processing time']]                  

UnjustifiedIssue = df[df['Coding code text'] == 'UNJUSTIFIED ISSUE'][['Processing time']]                        

OrderEntryError =  df[df['Coding code text'] == 'ORDER ENTRY ERROR'][['Processing time']]                        

InternalShippingError =  df[df['Coding code text'] == 'INTERNAL SHIPPING ERROR'][['Processing time']]                

PackingLineError =  df[df['Coding code text'] == 'PACKING LINE ERROR'][['Processing time']]                     

VendorError =  df[df['Coding code text'] == 'VENDOR ERROR'][['Processing time']]                         

IntercoRelated =  df[df['Coding code text'] == 'INTERCO RELATED'][['Processing time']]                         

DesignRelated =  df[df['Coding code text'] == 'DESIGN RELATED'][['Processing time']]                        

ExternalShippingError =  df[df['Coding code text'] == 'EXTERNAL (TRANSPORTATION) SHIPPING ERROR'][['Processing time']] 

CustomerAccommodation =  df[df['Coding code text'] == 'CUSTOMER ACCOMMODATION (SALES DECISION)'][['Processing time']] 

PartCreateError =  df[df['Coding code text'] == 'PART CREATE ERROR'][['Processing time']]                    

EngineeringRelated =  df[df['Coding code text'] == 'ENGINEERING RELATED'][['Processing time']]                  

DecorationError =  df[df['Coding code text'] == 'DECORATION ERROR'][['Processing time']]                    

MasterDataError =  df[df['Coding code text'] == 'MASTER DATA ERROR'][['Processing time']]                   

ToolRoomRelated =  df[df['Coding code text'] == 'TOOL ROOM RELATED'][['Processing time']]                    

CustomerRelated =  df[df['Coding code text'] == 'CUSTOMER RELATED'][['Processing time']]                    

x_data = ['ASSEMBLY RELATED','MOLDING RELATED','PROCEDURE NOT FOLLOWED','UNJUSTIFIED ISSUE','ORDER ENTRY ERROR',

          'INTERNAL SHIPPING ERROR','PACKING LINE ERROR','VENDOR ERROR','INTERCO RELATED','DESIGN RELATED',

          'EXTERNAL (TRANSPORTATION) SHIPPING ERROR','CUSTOMER ACCOMMODATION (SALES DECISION)',

          'PART CREATE ERROR','ENGINEERING RELATED','DECORATION ERROR','MASTER DATA ERROR',

           'TOOL ROOM RELATED','CUSTOMER RELATED']







y_data = [AssemblyRelated['Processing time'],MoldingRelated['Processing time'],ProcedureNotFollowed['Processing time'],UnjustifiedIssue['Processing time'],

          OrderEntryError['Processing time'],InternalShippingError['Processing time'],PackingLineError['Processing time'],VendorError['Processing time'],IntercoRelated['Processing time']       

,DesignRelated['Processing time']       

,ExternalShippingError['Processing time'] 

,CustomerAccommodation['Processing time'] 

,PartCreateError['Processing time']       

,EngineeringRelated['Processing time']    

,DecorationError['Processing time']       

,MasterDataError['Processing time']       

,ToolRoomRelated['Processing time']       

,CustomerRelated['Processing time']]



fig = go.Figure()



for xd, yd in zip(x_data, y_data):

        fig.add_trace(go.Box(

            y=yd,

            name=xd,            

            jitter=0.5,

            whiskerwidth=0.2,            

            marker_size=2,

            line_width=1)

        )



fig.update_layout(

    title='Processing Time Distribution For Each Issue',

   

    showlegend=False

)



fig.show()
df.loc[:,'Required Start']=pd.to_datetime(df['Required Start'])

df.loc[:,'Required End']=pd.to_datetime(df['Required End'])

df.loc[:,'Changed On']=pd.to_datetime(df['Changed On'])

df.loc[:,'Completion by date']=pd.to_datetime(df['Completion by date'])

df.loc[:,'Created On']=pd.to_datetime(df['Created On'])

df.loc[:,'Malfunction start']=pd.to_datetime(df['Malfunction start'])

df.loc[:,'Notification date']=pd.to_datetime(df['Notification date'])

df.loc[:,'Purchase Order Date']=pd.to_datetime(df['Purchase Order Date'])

df.loc[:,'Reference date']=pd.to_datetime(df['Reference date'])



timeDiff = (df['Changed On']-df['Required Start'])

timeDiff.describe()
complaintData = df[['Required Start','Complaint quantity']]

trace0 = go.Scatter(x = complaintData['Required Start'], y = complaintData['Complaint quantity'],name = 'Complaints')

data = [trace0]

layout = go.Layout(title = 'Ditribution of Customer Complaints',yaxis=dict(title='No. of Complaints'),xaxis = dict(title='Year of Complaint Made'))

fig = go.Figure(data=data,layout = layout)

iplot(fig,filename = 'complaints.html')
perCustomerComplaints = df.groupby(['Customer'])[['Complaint quantity']].sum().sort_values(by='Complaint quantity',ascending=False )

perCustomerComplaints['cumSumPercent'] = (perCustomerComplaints['Complaint quantity'].cumsum()/perCustomerComplaints['Complaint quantity'].sum())*100



customerIndex = []

for index in perCustomerComplaints.index:

    customerIndex.append("Customer "+index)

perCustomerComplaints.index = customerIndex





trace0 = dict(type='bar',

              x = perCustomerComplaints.index,

              y=perCustomerComplaints['Complaint quantity'],

     marker=dict(

        color='#2196F3'

    ),

    name='Complaint quantity',

    opacity=0.8

)



trace1 = dict(type='scatter',

              x = perCustomerComplaints.index,

              y=perCustomerComplaints['cumSumPercent'],

     marker=dict(

        color='#2196F3'

    ),

    line=dict(

        color= '#263238', 

        width= 1.5),

    name=' % of Complaints per Customer', 

    xaxis = 'x1',

    yaxis='y2'

)

trace2 =  dict(type='scatter',

              x = perCustomerComplaints.index,

              y=[80]*193,

     marker=dict(

        color='#2196F3'

    ),

               

    line=dict(

        color= 'firebrick',

        dash='dash',

        width= 1.5),

    name=' 80% cutoff', 

    xaxis = 'x1',

    yaxis='y2'

)

data = [trace0,trace1,trace2]

layout = go.Layout(

    title='[Pareto Analysis] Complaint quantity vs % of  Complaints per Customer',

    plot_bgcolor='rgba(0,0,0,0)',

    legend= dict(orientation="h",

                x=0.5,

                y=1.1),

    yaxis=dict(

        

        title='No. of Complaints',

        titlefont=dict(

            color="#2196F3"

        )

    ),

    yaxis2=dict(

        title=' % of Complaints per Customer',

        titlefont=dict(

            color='#263238'

        ),

        range=[0,105],

        overlaying='y',

        anchor='x',

        side='right'

        )

    )

    





fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="paretoCustomer")

complaintsPerCodeInfo = df[['Code group','Code group text','Coding code text','Complaint quantity']]

complaintsPerCodeInfo = complaintsPerCodeInfo.groupby(['Code group text'])[['Complaint quantity']].sum().sort_values(by='Complaint quantity',ascending=False)

complaintsPerCodeInfo['cumSumPercent'] = (complaintsPerCodeInfo['Complaint quantity'].cumsum()/complaintsPerCodeInfo['Complaint quantity'].sum())*100

complaintsPerCodeInfo
trace0 = dict(type='bar',

              x = complaintsPerCodeInfo.index,

              y=complaintsPerCodeInfo['Complaint quantity'],

     marker=dict(

        color='#2196F3'

    ),

    name='Complaint quantity',

    opacity=0.8

)



trace1 = dict(type='scatter',

              x = complaintsPerCodeInfo.index,

              y=complaintsPerCodeInfo['cumSumPercent'],

     marker=dict(

        color='#2196F3'

    ),

    line=dict(

        color= '#263238', 

        width= 1.5),

    name=' % of Complaints per Issue Type', 

    xaxis = 'x1',

    yaxis='y2'

)

trace2 =  dict(type='scatter',

              x = complaintsPerCodeInfo.index,

              y=[80]*193,

     marker=dict(

        color='#2196F3'

    ),

               

    line=dict(

        color= 'firebrick',

        dash='dash',

        width= 1.5),

    name=' 80% cutoff', 

    xaxis = 'x1',

    yaxis='y2'

)

data = [trace0,trace1,trace2]

layout = go.Layout(

    title='[Pareto Analysis] Complaint quantity vs % of  Complaints per Issue Type',

    plot_bgcolor='rgba(0,0,0,0)',

    legend= dict(orientation="h",

                x=0.5,

                y=1.1),

    yaxis=dict(

        

        title='No. of Complaints',

        titlefont=dict(

            color="#2196F3"

        )

    ),

    yaxis2=dict(

        title=' % of Complaints per Issue Type',

        titlefont=dict(

            color='#263238'

        ),

        range=[0,105],

        overlaying='y',

        anchor='x',

        side='right'

        )

    )

    





fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="paretoIssue")
df.head(2)
codeInfo = df[['Code group','Code group text','Coding Code','Coding code text','Complaint quantity']]

def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

   

    is_max = s == s.max()

    return ['background-color: yellow' if v else '' for v in is_max]

table = codeInfo.pivot_table(index = ['Code group','Code group text'],columns = ['Coding code text'],values = ['Complaint quantity'],aggfunc = 'sum')

table.style.apply(highlight_max,axis = 1)
TotalCostPerIssue = df.groupby(['Code group text'])[['Total Cost']].sum().sort_values('Total Cost',ascending = False)

TotalCostPerIssue = TotalCostPerIssue.groupby(['Code group text'])[['Total Cost']].sum().sort_values(by='Total Cost',ascending=False)

TotalCostPerIssue['cumSumPercent'] = (TotalCostPerIssue['Total Cost'].cumsum()/TotalCostPerIssue['Total Cost'].sum())*100

TotalCostPerIssue
trace0 = dict(type='bar',

              x = TotalCostPerIssue.index,

              y=TotalCostPerIssue['Total Cost'],

     marker=dict(

        color='#2196F3'

    ),

    name='Cost',

    opacity=0.8

)



trace1 = dict(type='scatter',

              x = TotalCostPerIssue.index,

              y=TotalCostPerIssue['cumSumPercent'],

     marker=dict(

        color='#2196F3'

    ),

    line=dict(

        color= '#263238', 

        width= 1.5),

    name=' % of Total Cost per Issue', 

    xaxis = 'x1',

    yaxis='y2'

)

trace2 =  dict(type='scatter',

              x = TotalCostPerIssue.index,

              y=[80]*193,

     marker=dict(

        color='#2196F3'

    ),

               

    line=dict(

        color= 'firebrick',

        dash='dash',

        width= 1.5),

    name=' 80% cutoff', 

    xaxis = 'x1',

    yaxis='y2'

)

data = [trace0,trace1,trace2]

layout = go.Layout(

    title='[Pareto Analysis] Cost vs  % of Total Cost per Issue',

    plot_bgcolor='rgba(0,0,0,0)',

    legend= dict(orientation="h",

                x=0.5,

                y=1.1),

    yaxis=dict(

        

        title='Cost',

        titlefont=dict(

            color="#2196F3"

        )

    ),

    yaxis2=dict(

        title='  % of Total Cost per Issue',

        titlefont=dict(

            color='#263238'

        ),

        range=[0,105],

        overlaying='y',

        anchor='x',

        side='right'

        )

    )

    





fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="paretoCustomer")

productIssueSubcategory = df[df['Code group text'] == 'PRODUCT ISSUE'][['Code group text','Coding code text','Total Cost']]

productIssueSubcategory = productIssueSubcategory.pivot_table(index = ['Coding code text'],values = 'Total Cost',aggfunc = 'sum').sort_values(by='Total Cost',ascending=False)

productIssueSubcategory['cumSumPercent'] = (productIssueSubcategory['Total Cost'].cumsum()/productIssueSubcategory['Total Cost'].sum())*100

productIssueSubcategory
productIssueSubcategory.index
trace0  = dict(type = 'bar',

               x = productIssueSubcategory.index,

               y = productIssueSubcategory['Total Cost'],

               name = 'productIssueSubcategory',

               marker=dict(

        color='#2196F3'

    ),opacity=0.8

)

data = [trace0]

layout = go.Layout(

    title = 'Total Cost Per productIssueSubcategory'

)

fig = go.Figure(data = data,layout = layout)

iplot(fig,filename = 'ParetoproductIssueSubcategory')