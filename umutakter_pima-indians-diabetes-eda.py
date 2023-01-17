import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as mno



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
data.head()
data.tail()
data.sample(5)
data.shape
data.info()
data.isnull().sum()
data.describe().T
data.loc[data["Glucose"] == 0.0, "Glucose"] = np.NAN

data.loc[data["BloodPressure"] == 0.0, "BloodPressure"] = np.NAN

data.loc[data["SkinThickness"] == 0.0, "SkinThickness"] = np.NAN

data.loc[data["Insulin"] == 0.0, "Insulin"] = np.NAN

data.loc[data["BMI"] == 0.0, "BMI"] = np.NAN
data.isnull().sum()
mno.matrix(data, figsize = (20, 6))
mno.heatmap(data, figsize= (15,8));
mno.dendrogram(data)
data.isnull().sum()
"|","Ortalama: ", data.Glucose.mean(),"|","Medyan: ", data.Glucose.median(), "|"
f,ax=plt.subplots(figsize=(10,6))

sns.violinplot(data["Glucose"])

plt.show()
data["Glucose"].fillna(data["Glucose"].mean(), inplace = True)

data.Glucose.isnull().sum()
"|","Ortalama: ", data.BloodPressure.mean(),"|","Medyan: ", data.BloodPressure.median(), "|"
f,ax=plt.subplots(figsize=(10,6))

sns.violinplot(data["BloodPressure"])

plt.show()
data["BloodPressure"].fillna(data["BloodPressure"].mean(), inplace = True)

data.BloodPressure.isnull().sum()
"|","Ortalama: ", data.SkinThickness.mean(),"|","Medyan: ", data.SkinThickness.median(), "|"
f,ax=plt.subplots(figsize=(10,6))

sns.violinplot(data["SkinThickness"])

plt.show()
data["SkinThickness"].fillna(data["SkinThickness"].median(), inplace = True)

data.SkinThickness.isnull().sum()
"|","Ortalama: ", data.Insulin.mean(),"|","Medyan: ", data.Insulin.median(), "|"
f,ax=plt.subplots(figsize=(10,6))

sns.violinplot(data["Insulin"])

plt.show()
data["Insulin"].fillna(data["Insulin"].median(), inplace = True)

data.Insulin.isnull().sum()
"|","Ortalama: ", data.BMI.mean(),"|","Medyan: ", data.BMI.median(), "|"
f,ax=plt.subplots(figsize=(10,6))

sns.violinplot(data["BMI"])

plt.show()
data["BMI"].fillna(data["BMI"].median(), inplace = True)

data.BMI.isnull().sum()
data.isnull().sum()
data.describe().T
data.corr()
corr = data.corr()

f,ax=plt.subplots(figsize=(25,15))

sns.heatmap(corr, annot=True, ax=ax)

plt.show()
fig = px.scatter(data, x="Age", y="Pregnancies", trendline="ols",color="Age")



fig.update_layout(

    title={

        'text': "Age-Pregnancies Regression Plot",

        'y':0.95,

        'x':0.5

})



fig.show()



fig = px.scatter(data, x="SkinThickness", y="BMI", trendline="ols",color="BMI")



fig.update_layout(

    title={

        'text': "SkinThickness-BMI Regression Plot",

        'y':0.95,

        'x':0.5

})



fig.show()
highGlucose = data[data.Glucose>140]

normalGlucose = data[(data.Glucose<=140)]
newData=data.copy()

bins=[0,140,200]

labels=['NormalGlucose','HighGlucose']

newData['GlucoseNH']=pd.cut(newData['Glucose'],bins,labels=labels)
x1 = [highGlucose[highGlucose.Outcome==0.00].Glucose.count(),

highGlucose[highGlucose.Outcome==1.00].Glucose.count()]



x2 = [normalGlucose[normalGlucose.Outcome==0.00].Glucose.count(),

normalGlucose[normalGlucose.Outcome==1.00].Glucose.count()]

f,ax=plt.subplots(figsize=(12,6))

plt.bar(["Sağlıklı Birey \n (Outcome=0)","Diyabet Hastası \n (Outcome=1)"],x1 )

plt.title("High Glucose")

plt.show()
f,ax=plt.subplots(figsize=(12,6))

plt.bar(["Sağlıklı Birey \n (Outcome=0)","Diyabet Hastası \n (Outcome=1)"],x2 )

plt.title("Normal Glucose")

plt.show()
f,ax=plt.subplots(figsize=(12,6))

sns.swarmplot(x="GlucoseNH",y="Glucose",hue="Outcome",data=newData)

plt.show()
f,ax=plt.subplots(figsize=(12,6))

sns.barplot(x="Outcome", y="Insulin", data=data)

plt.show()
trace1={

    "values":[data[data.Outcome==0].Insulin.mean(),data[data.Outcome==1].Insulin.mean()],

    "labels":[0,1],

    "domain": {"x":[0,.5]},

    "name":"Ortalamalar oranı",

    "hole":.3,

    "type":"pie"

}

data1=[trace1];

layout={

    "title":"Outcome-Insulin",

    "annotations":[

        {"font":{"size":15},

        "showarrow":False,

        "text":"Ortalamalar Oranı",

        "x":0.50,

        "y":0.9

        },

    ]

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)
sns.jointplot( data.Glucose,data.Insulin, kind="reg",height=7)

plt.show()
data1=[

    {

        "y":data.Insulin,

        "x":data.Glucose,

        "mode":"markers",

        "marker":{

            "color":data.Outcome,

            #"size":num_student_size,

            "showscale":True

        },

        "text":data.Glucose

    }

]

iplot(data1)
ageSize=data.Age/max(data.Age)*20

data1=[

    {

        "y":data.Insulin,

        "x":data.Glucose,

        "mode":"markers",

        "marker":{

            "color":data.Outcome,

            "size":ageSize,

            "showscale":True

        },

        "text":data.Glucose

    }

]

iplot(data1)
bins=[20,30,55,81]

labels=['Genç','Orta Yaş','Yaşlı']

data['YasGrp']=pd.cut(data['Age'],bins,labels=labels)
yasGrpValue=data.YasGrp.value_counts()
trace={

    "values":yasGrpValue.values,

    "labels":yasGrpValue.index,

    "hole":.2,

    "type":"pie",

}

data1=[trace];

layout={

    "title":"Yaş Dağılımları",

    

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)
plt.figure(figsize=(15,4))

sns.countplot(x="YasGrp", hue="Outcome",palette="Set2", data=data)

plt.show()
dataGenc=data[data.YasGrp=="Genç"].copy()

dataOrta=data[data.YasGrp=="Orta Yaş"].copy()

dataYasli=data[data.YasGrp=="Yaşlı"].copy()
trace={

    "values":dataGenc.Outcome.value_counts(),

    "labels":["Sağlıklı","Diyabet Hastası"],

    "hole":.2,

    "type":"pie",

    'marker': {

      'colors': [

        'rgb(173, 235, 173)',

        'rgb(153, 179, 255)'

      ]

    }

}

data1=[trace];

layout={

    "title":"Gençlerde Diyabet Oranı", 

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)
trace={

    "values":dataOrta.Outcome.value_counts(),

    "labels":["Diyabet Hastası","Sağlıklı"],

    "hole":.2,

    "type":"pie",

    'marker': {

      'colors': [

        'rgb(153, 179, 255)',

        'rgb(173, 235, 173)'             

      ]

    }

}

data1=[trace];

layout={

    "title":"Orta Yaşlılarda Diyabet Oranı", 

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)
trace={

    "values":dataYasli.Outcome.value_counts(),

    "labels":["Sağlıklı","Diyabet Hastası"],

    "hole":.2,

    "type":"pie",

    'marker': {

      'colors': [

        'rgb(173, 235, 173)',

        'rgb(153, 179, 255)'

      ]

    }

}

data1=[trace];

layout={

    "title":"Yaşlılarda Diyabet Oranı", 

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)
bins=[20,30,55,81]

labels=['Genç','Orta Yaş','Yaşlı']

highGlucose['YasGrp']=pd.cut(highGlucose['Age'],bins,labels=labels)

normalGlucose['YasGrp']=pd.cut(normalGlucose['Age'],bins,labels=labels)
plt.figure(figsize=(15,4))

sns.countplot(x="YasGrp", hue="Outcome",palette="Set1", data=highGlucose)

plt.show()
trace={

    "values":highGlucose[highGlucose.YasGrp=="Orta Yaş"].Outcome.value_counts(),

    "labels":["Diyabet Hastası","Sağlıklı"],

    "hole":.2,

    "type":"pie",

    'marker': {

      'colors': [

        'rgb(255, 102, 102)',

        'rgb(213, 128, 255)'             

      ]

    }

}

data1=[trace];

layout={

    "title":"Orta Yaşlılarda Diyabet Oranı", 

};

fig=go.Figure(data=data1,layout=layout)

iplot(fig)