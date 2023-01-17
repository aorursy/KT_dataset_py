# Pacotes de análise de dados:
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pylab import rcParams

# Pacotes para aprendizagem de maquina via sklearn:
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Entrada de dados:
df = pd.read_csv("../input/iris/Iris.csv")
df.drop('Id', axis=1, inplace=True)
df.head()
## Analise exploratória de dados
# Estatística Descritiva:
round(df.describe(),3)
# Verificação da variável target:
plt.figure(figsize=(12,5))
sns.countplot('Species',data=df)
plt.title("Verificação da variável target")
plt.show()
# Histogramas com foco no target:
plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(4,1)
ax = plt.subplot()
sns.distplot(df['PetalWidthCm'][df.Species == 'Iris-setosa'], bins=10)
sns.distplot(df['PetalWidthCm'][df.Species == 'Iris-versicolor'], bins=10)
sns.distplot(df['PetalWidthCm'][df.Species == 'Iris-virginica'], bins=10)
ax.set_xlabel('')
ax.set_title('Histograma relação de variáveis com o Target: ' + str('PetalWidthCm'))
plt.show()

plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(4,1)
ax = plt.subplot()
sns.distplot(df['PetalLengthCm'][df.Species == 'Iris-setosa'], bins=10)
sns.distplot(df['PetalLengthCm'][df.Species == 'Iris-versicolor'], bins=10)
sns.distplot(df['PetalLengthCm'][df.Species == 'Iris-virginica'], bins=10)
ax.set_xlabel('')
ax.set_title('Histograma relação de variáveis com o Target: ' + str('PetalLengthCm'))
plt.show()

plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(4,1)
ax = plt.subplot()
sns.distplot(df['SepalWidthCm'][df.Species == 'Iris-setosa'], bins=10)
sns.distplot(df['SepalWidthCm'][df.Species == 'Iris-versicolor'], bins=10)
sns.distplot(df['SepalWidthCm'][df.Species == 'Iris-virginica'], bins=10)
ax.set_xlabel('')
ax.set_title('Histograma relação de variáveis com o Target: ' + str('SepalWidthCm'))
plt.show()

plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(4,1)
ax = plt.subplot()
sns.distplot(df['SepalLengthCm'][df.Species == 'Iris-setosa'], bins=10)
sns.distplot(df['SepalLengthCm'][df.Species == 'Iris-versicolor'], bins=10)
sns.distplot(df['SepalLengthCm'][df.Species == 'Iris-virginica'], bins=10)
ax.set_xlabel('')
ax.set_title('Histograma relação de variáveis com o Target: ' + str('SepalLengthCm'))
plt.show()
# Mapa de correlações
sns.set(rc={'figure.figsize':(8,6)})
sns.heatmap(df.corr(method='spearman'),fmt = '.2f',cmap='Greens')
plt.title('Correlação entre variáveis')
plt.show()
# Pacotes para AutoML:
import h2o
from h2o.automl import H2OAutoML
 
# Start cluster:
h2o.init()
# Divisão do dataset:
train, test = train_test_split(df, test_size=0.2)

# Conversão para h2o frame:
traindf = h2o.H2OFrame(train)
testdf = h2o.H2OFrame(test)

# Criação das variáveis de entrada no AutoML:
y = "Species"
x = list(traindf.columns)
x.remove(y)
 
# Tratamento na variável target:
traindf[y] = traindf[y].asfactor()
testdf[y] = testdf[y].asfactor()
# AutoML H2O:
aml = H2OAutoML(max_models = 100, max_runtime_secs = 600, seed = 7450)
aml.train(x = x, y = y, training_frame = traindf)

# Leader board:
print(aml.leaderboard)
# Previsões e tratamentos:
predict = aml.predict(testdf)
p = predict.as_data_frame()
print(" ")

# Conversão para dataframe pandas:
data = {'actual': test.Species,'Ypredict': p['predict'].tolist()}
df = pd.DataFrame(data, columns = ['actual','Ypredict'])
print(df.head(3))
# Matriz confusão:
confusion_matrix = pd.crosstab(df['actual'], df['Ypredict'], rownames=['Actual'], colnames=['Predicted'])

sns.set(rc={'figure.figsize':(8,6)})
sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix,annot=True,annot_kws={"size": 16},cmap='Greens')
plt.title("Matriz de confusão")
plt.show()
# Shutdown h2o cluster:
h2o.cluster().shutdown(prompt = False)