import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import matplotlib.gridspec as gridspec
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv', header = [0])
feature = [feat for feat in list(df) if feat not in ['id','Unnamed: 32']]
df1 = df.filter(feature)
# Ver o balanceamento das classes de diagnosticos
print(df1.diagnosis.value_counts())
print("\nCasos benignos representam {:.4f}% do dataset.\n".format((df1[df1.diagnosis == 'B'].shape[0] / df1.shape[0]) * 100))

# Gráfico de barras para as classes de diagnosticos
sns.countplot('diagnosis',data=df1)
plt.title("Verificação da variável target")
plt.show()
# Estatísticas das médias das características
df1.filter(['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
            'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']).describe()
# Estatísticas das desvio padrão das características
df1.filter(['radius_se','texture_se','perimeter_se','area_se','smoothness_se',
            'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se']).describe()
# Estatísticas das piores medidas das características
df1.filter(['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']). describe()
# Histogramas com foco no target
v_features = df1.iloc[:,1:31].columns
plt.figure(figsize=(12,31*4))
gs = gridspec.GridSpec(31, 1)
for i, cn in enumerate(df1[v_features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df1[cn][df1.diagnosis == 'B'], bins=50)
    sns.distplot(df1[cn][df1.diagnosis == 'M'], bins=50)
    ax.set_xlabel('')
    ax.set_title('Histograma relação de variáveis com o Target: ' + str(cn))
plt.show()
# Mapa de correlações
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df1.corr(method='spearman'),fmt = '.2f',cmap='Greens')
plt.title('Correlação entre variáveis')
plt.show()
# Bibliotecas para modelagem de PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Seleção das variáveis
feature = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
x = df1.filter(feature)

# Padronização das variáveis
x = StandardScaler().fit_transform(x)

# Modelagem PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
var_explicada = pca.explained_variance_ratio_
var_exp_df = pd.DataFrame({"var_exp":var_explicada})
print("A variância explicada dos quatro componentes: ",(var_exp_df['var_exp'].sum().round(2))*100,"%")

# Dataset transformado
principalDf = pd.DataFrame(data = principalComponents,columns = ['pc1', 'pc2', 'pc3', 'pc4'])
df_pca = pd.concat([principalDf, df1['diagnosis']], axis = 1)
print(" ")
print("Dataset com as componentes principais:")
print(" ")
print(df_pca.head(1))

# Gráficos PCA - PC1 e PC2
plt.figure(figsize=(10,8))
sns.scatterplot(x="pc1", y="pc2", hue="diagnosis", data=df_pca)
plt.title("Principal Components PC1 and PC2")
plt.show()

# Gráficos PCA - PC1 e PC3
plt.figure(figsize=(10,8))
sns.scatterplot(x="pc1", y="pc3", hue="diagnosis", data=df_pca)
plt.title("Principal Components PC1 and PC3")
plt.show()

# Gráficos PCA - PC1 e PC4
plt.figure(figsize=(10,8))
sns.scatterplot(x="pc1", y="pc4", hue="diagnosis", data=df_pca)
plt.title("Principal Components PC1 and PC4")
plt.show()
# Pacotes para aprendizagem de maquina via sklearn:
import random
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             auc,roc_curve,recall_score, classification_report,
                             f1_score, precision_recall_fscore_support)

# Divisão do dataset:
xtr, xval, ytr, yval = train_test_split(df_pca.drop('diagnosis',axis=1),
                                        df_pca['diagnosis'],
                                        test_size=0.2,
                                        random_state=5478)

# Modelagem estatística regressão logística:
baseline = LogisticRegression()
baseline.fit(xtr,ytr)

# Previsões:
p = baseline.predict(xval)

# Matriz de confusão:
cmx = confusion_matrix(yval, p)
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale=1.4)
sns.heatmap(cmx,annot=True,annot_kws={"size": 14},cmap='Greens')
plt.title("Matriz de confusão")
plt.show()

# Resultados:
print("Resultados da modelagem:")
print(classification_report(yval, p))
# Pacotes para AutoML:
import h2o
from h2o.automl import H2OAutoML
 
# Start cluster:
h2o.init()
# Divisão do dataset:
train, test = train_test_split(df1, test_size=0.2)

# Conversão para h2o frame:
traindf = h2o.H2OFrame(train)
testdf = h2o.H2OFrame(test)

# Criação das variáveis de entrada no AutoML:
y = "diagnosis"
x = list(traindf.columns)
x.remove(y)
 
# Tratamento na variável target:
traindf[y] = traindf[y].asfactor()
testdf[y] = testdf[y].asfactor()
# AutoML H2O:
aml = H2OAutoML(max_models = 80, max_runtime_secs = 300, seed = 247)
aml.train(x = x, y = y, training_frame = traindf)

# Leader board:
print(aml.leaderboard)
# Previsões e tratamentos:
predict = aml.predict(testdf)
p = predict.as_data_frame()
print(" ")

# Conversão para dataframe pandas:
data = {'actual': test.diagnosis,'Ypredict': p['predict'].tolist()}
df = pd.DataFrame(data, columns = ['actual','Ypredict'])
print(df.head(3))
# Matriz confusão:
confusion_matrix = pd.crosstab(df['actual'], df['Ypredict'], rownames=['Actual'], colnames=['Predicted'])

sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale=1.4)
sns.heatmap(confusion_matrix,annot=True,annot_kws={"size": 16},cmap='Greens')
plt.title("Matriz de confusão")
plt.show()

print("Resultados da modelagem:")
print(classification_report(df['actual'], df['Ypredict']))
# Shutdown h2o cluster:
h2o.cluster().shutdown(prompt = False)