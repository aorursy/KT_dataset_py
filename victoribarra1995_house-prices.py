
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor

#Leer los datos
train = pd.read_csv('../input/train.csv')
#extraer los datos en el objetivo (y) y predictores en (X)
train_y = train.SalePrice
predictor_cols=['LotArea','OverallQual','YearBuilt','TotRmsAbvGrd']
# Crear datos de predictores de entrenamiento
train_X=train[predictor_cols]

#Creamos una variable para nuestro modelo y lo entrenamos
my_model=AdaBoostRegressor()
my_model.fit(train_X,train_y)
train.head()

#Leemos los datos de prueba
test = pd.read_csv('../input/test.csv')
#Tratamos los datos de prueba de la misma manera que los datos de prueba. Utilizamos las mismas columnas(predictores).
test_X=test[predictor_cols]
#Usamos el modelo para hacer predicciones
precios_pronosticados=my_model.predict(test_X)
#Vemos los precios pronosticados
print(precios_pronosticados)
my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':precios_pronosticados})
#Se puede usar cualquier nombre para el archivo. Elegimos la presentación aquí
my_submission.to_csv('s2ubmission.csv', index=False)