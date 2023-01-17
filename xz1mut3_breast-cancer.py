



import pandas as pd 



train_df = pd.read_csv("../input/data.csv")

test_df = train_df

train_df.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'radius_mean',bins=20)



#Textura

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'texture_mean',bins=20)
#Perímetro

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'perimeter_mean',bins=20)
#Area

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'area_mean',bins=20)
#Suavidade

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'smoothness_mean',bins=20)
#Compacidade

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'compactness_mean',bins=20)


#Concavidade

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concavity_mean',bins=20)
#Pontos côncavos

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concave points_mean',bins=20)
#Simetria

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'symmetry_mean',bins=20)
#Dimensão fractal

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'fractal_dimension_mean',bins=20)
#Raio

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'radius_se',bins=20)
#Textura

g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'texture_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'perimeter_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'area_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'smoothness_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'compactness_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concavity_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concave points_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'symmetry_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'fractal_dimension_se',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'radius_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'texture_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'perimeter_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'area_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'smoothness_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'compactness_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concavity_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'concave points_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'symmetry_worst',bins=20)
g = sns.FacetGrid(train_df,col='diagnosis')

g.map(plt.hist,'fractal_dimension_worst',bins=20)
train_df.describe(include=['O'])
train_df['Unnamed: 32'] = train_df['Unnamed: 32'].fillna('0')

train_df.isnull().sum().sort_values(ascending=False)




train_df = train_df.drop("id",axis=1)

train_df.drop("area_se",axis=1)

train_df.drop("symmetry_se",axis=1)

train_df.drop("texture_worst",axis=1)

train_df.drop("smoothness_worst",axis=1)

train_df.drop("concave points_worst",axis=1)

train_df.drop("symmetry_worst",axis=1)

train_df.drop("smoothness_se",axis=1)

train_df.drop("compactness_se",axis=1)

train_df.drop("texture_se",axis=1)

train_df.drop("symmetry_mean",axis=1)

train_df.drop("Unnamed: 32",axis=1)







train_df.head()



classe = train_df['diagnosis']

atributos = train_df.drop('diagnosis',axis=1)
classe.head()

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributes_test,classe_train ,classe_test = train_test_split(atributos,classe,test_size=0.25)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, classe_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributes_test)

acc = accuracy_score(classe_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))