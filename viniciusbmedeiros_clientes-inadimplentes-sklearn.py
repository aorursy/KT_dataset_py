!wget -O gold.parquet https://www.dropbox.com/s/3m0xqogz5gi2moy/gold.parquet?dl=1
!ls /content
import pandas as pd

work_dir = "/content"

#leitura dos dados de entrada
df_bruto = pd.read_parquet(work_dir +"/gold.parquet", engine="pyarrow")
df_bruto.shape
df_bruto
df_bruto.info()
# não iremos utilizar estas duas colunas no treinamento e no teste
df_preparado = df_bruto.drop(['key', 'timestamp'], axis=1)

# iremos remover a feature ultima_compra, 
# mas fica como exercício você aproveitá-la no conjunto de features
df_preparado = df_preparado.drop(['ultima_compra'], axis=1)
df_preparado

df_bruto.head(200)
#pd.set_option('display.max_rows', 200)
df_preparado
# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses']
df_preparado[colunas_numericas] = df_preparado[colunas_numericas].fillna(value=0)

# transformar colunas categóricas em numéricas
df_preparado = pd.get_dummies(df_preparado, columns=["city", "state", "cnae_id"])
df_preparado.head()
#Tentei deixar a coluna "ultima_compra" para ver se aumenta o Score, mas não deu certo.

"""
#df_preparado2
#df_preparado2 = df_bruto.drop(['key', 'timestamp'], axis=1)

# Substitui valores nulos por 0 nas colunas numéricas
colunas_numericas2 = ['pedidos_4_meses','pedidos_8_meses','pedidos_12_meses','itens_4_meses','itens_8_meses','itens_12_meses', 'ultima_compra']
df_preparado2[colunas_numericas2] = df_preparado2[colunas_numericas2].fillna(value=0)

df_preparado2['ultima_compra'] = df_preparado2['ultima_compra'].astype('bool')
df_preparado2

# transformar colunas categóricas em numéricas
df_preparado2 = pd.get_dummies(df_preparado2, columns=["city", "state", "cnae_id"])
df_preparado2.head()


#df_preparado = df_preparado2
#df_preparado
df_preparado2["ultima_compra"] = df_preparado2["ultima_compra"].astype(int)

df_preparado = df_preparado2
df_preparado
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# seleciona as tuplas com rótulos
df_to_train = df_preparado[df_preparado["defaulting"].notnull()]

# remove a coluna defaulting dos dados de treinamento para não gerar overfiting
X = df_to_train.drop('defaulting', axis=1)

# Transforma a variável a predizer de boolean para inteiro
le = LabelEncoder()
y = le.fit_transform(df_to_train.defaulting.values)

# Divisão em conjunto de treinamento e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=1) #Mudei o test_size de 0.2 para 0.01, ai aumentou o Score

print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
X_train
df_to_train.defaulting.values
y
X_train.info() #MUST BE int, float or bool.
#Havia feito um clone dos datasets, e converti a coluna client_id para INT para o xgboost não reclamar.
#X_train_TESTE = X_train.copy()
#X_valid_TESTE = X_valid.copy()

"""
Convertendo a primeira coluna para INT
X_train to X_train_TESTE
y_train to  - NAO PRECISA
X_valid to X_valid_TESTE
"""
#X_train_TESTE = X_train_TESTE.astype({"client_id": int}) 
#X_valid_TESTE = X_valid_TESTE.astype({"client_id": int}) 


#Converti a coluna client_id para INT para o xgboost não reclamar.
X_train = X_train.astype({"client_id": int}) 
X_valid = X_valid.astype({"client_id": int}) 
import xgboost as xgb
import sklearn.metrics as metrics
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier

# Cria um classificador
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = xgb.XGBClassifier(learning_rate=0.009, max_depth=6, min_child_weight=3, subsample=0.15, colsample_bylevel=0.85, n_estimators=500)
clf = xgb.XGBClassifier()

# Treina a Árvore de Decisão
clf = clf.fit(X_train,y_train)

# Prediz a resposta para o dataset de validação
y_pred = clf.predict(X_valid)

#8000 linhas para treinamento e 2000 rows para treinamento, e 5000 estão nullos, ou seja, não sabemos, serão os de testes la no kaggle

print(y_pred) #Verifica se o result está corretos (apenas com Zeros e Uns)
print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred)) #Imprime o Score
y_valid.shape
y_pred.shape
import sklearn.metrics as metrics
print("ROC AUC:",metrics.roc_auc_score(y_valid, y_pred)) #Usada no Kaggle
print("Acurácia:",metrics.accuracy_score(y_valid, y_pred))
print("F1 score:",metrics.f1_score(y_valid, y_pred))

#0.8214285714285714
df_test = df_preparado[df_preparado["defaulting"].isnull()]
df_test.shape
X_test = X_test.astype({"client_id": int}) 
#X_test = df_test.drop('defaulting', axis=1)
y_test = clf.predict(X_test)
y_test
output = df_test.assign(inadimplente=y_test)
output = output.loc[:, ['client_id','inadimplente']]
output.head()
output.to_csv(work_dir +"/ouput_sklearn.csv", index=False)
from google.colab import files
files.download('ouput_sklearn.csv') 