import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import  datetime

#Архивированные csv-файл предварительно разархивируются (в /kaggle/working/)
!unzip /kaggle/input/springleaf-marketing-response/test.csv.zip
!unzip /kaggle/input/springleaf-marketing-response/sample_submission.csv.zip
!unzip /kaggle/input/springleaf-marketing-response/train.csv.zip
#Чтение данных в csv-файле
train = pd.read_csv("/kaggle/working/train.csv") #dtype={id: int, mean: float}
test = pd.read_csv("/kaggle/working/test.csv")
#train.describe() # вычисляет различную сводную статистику, исключая значения NaN
train.info() #используется для получения краткой сводки данных. 
train = train[:10000]
def get_data(): #функция
    
    features = train.select_dtypes(include=['float']).columns #Возвращает подмножество столбцов фрейма данных, основанных на типах float столбцов.
    features = np.setdiff1d(features,['ID','target']) #Возвращайте уникальные значения в поле ID, то есть не в том target.
        
    test_ids = test.ID
    y_train = train.target

    x_train = train[features]
    x_test = test[features]

    return x_train, y_train, x_test, test_ids
ts = datetime.now()
x_train, y_train, x_test, test_ids = get_data()

xgb_params = {"objective": "binary:logistic", "max_depth": 10, "silent": 1} # логистическая регрессия с бинарной величиной предсказания. 
                                                                            # Максимальная глубина дерева
num_rounds = 200 # количество построенных деревьев решений.

dtrain = xgb.DMatrix(x_train, label=y_train) #построение деревьев
dtest = xgb.DMatrix(x_test)
gbdt = xgb.train(xgb_params, dtrain, num_rounds)

preds = gbdt.predict(dtest) #прогноз 


submission = pd.DataFrame({"ID": test_ids, "target": preds}) #Арифметические операции выравниваются как по меткам строк, так и по меткам столбцов. 
submission = submission.set_index('ID') #Задает индекс фрейма данных
submission.to_csv('xgb_benchmark.csv')

te = datetime.now()
print('elapsed time: {0}'.format(te-ts))
#0.65177