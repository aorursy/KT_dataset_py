import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
#Função de Pré-processamento
def pre_process (df):
    
    new_df = pd.DataFrame()
    
    for n,c in df.items():
                
        if pd.api.types.is_numeric_dtype(c):
            # substituindo NaN numericos pelas medianas de cada coluna
            new_df[n] = c.fillna(value=c.median())
        else:
            # interpretando o que nao for numerico como variaveis categoricas 
            # e transformando cada categoria em um numero
            new_df[n] = pd.Categorical(c.astype('category').cat.as_ordered()).codes
    
    return new_df     
#Definindo as métricas 
def rmse(x,y): 
    
    return np.sqrt(sklearn.metrics.mean_squared_error(x,y))

def display_score(m):
    
    res = [[rmse(m.predict(X_treino), y_treino), m.score(X_treino, y_treino)],
          [rmse(m.predict(X_validacao), y_validacao), m.score(X_validacao, y_validacao)]]
    
    score = pd.DataFrame(res, columns=['RMSE','R2'], index = ['Treino','Validação'])
    
    if hasattr(m, 'oob_score_'): 
        score.loc['OOB'] = [rmse(y_treino, m.oob_prediction_), m.oob_score_]
        
    display(score)
#Carregar os dados
PATH = "../input/bluebook-for-bulldozers/"

df_raw = pd.read_csv(f'{PATH}Train.zip',
                     compression='zip', 
                     low_memory=False, 
                     parse_dates=["saledate"])


#Processamento dos dados
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_proc = pre_process(df_raw)

X, y = df_proc.drop('SalePrice', axis=1), df_proc['SalePrice']

n_valid = 12000
n_trn = len(df_proc)-n_valid

X_treino, X_validacao = X[:n_trn].copy(), X[n_trn:].copy()
y_treino, y_validacao = y[:n_trn].copy(), y[n_trn:].copy()

y_treino.shape, y_validacao.shape

#Treinando o modelo
m_base = sklearn.ensemble.RandomForestRegressor(n_estimators = 35, max_features = 0.6, n_jobs=-1,
                                                min_samples_leaf = 3, oob_score = True, random_state = 0)
%time m_base.fit(X_treino, y_treino)
display_score(m_base)