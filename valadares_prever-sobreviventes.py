# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np



arquivos_de_treino = pd.read_csv('/kaggle/input/titanic/train.csv')

arquivos_de_teste = pd.read_csv('/kaggle/input/titanic/test.csv')
arquivos_de_treino = arquivos_de_treino.replace(np.nan, 0)

arquivos_de_teste = arquivos_de_teste.replace(np.nan, 0)



arquivos_de_treino.head()

#arquivos_de_teste.head()
arquivos_de_treino['Sex'] =  arquivos_de_treino['Sex'].replace('male',0)

arquivos_de_treino['Sex'] =  arquivos_de_treino['Sex'].replace('female',1)



arquivos_de_teste['Sex'] =  arquivos_de_teste['Sex'].replace('male',0)

arquivos_de_teste['Sex'] =  arquivos_de_teste['Sex'].replace('female',1)



arquivos_de_treino['Embarked'] =  arquivos_de_treino['Embarked'].replace('C',1)

arquivos_de_treino['Embarked'] =  arquivos_de_treino['Embarked'].replace('Q',2)

arquivos_de_treino['Embarked'] =  arquivos_de_treino['Embarked'].replace('S',3)



arquivos_de_teste['Embarked'] =  arquivos_de_teste['Embarked'].replace('C',1)

arquivos_de_teste['Embarked'] =  arquivos_de_teste['Embarked'].replace('Q',2)

arquivos_de_teste['Embarked'] =  arquivos_de_teste['Embarked'].replace('S',3)
y = arquivos_de_treino["Survived"]

x = arquivos_de_treino.drop(["Survived","Name","Cabin","Ticket"],axis=1)

clean_teste = arquivos_de_teste.drop(["Name","Cabin","Ticket"],axis=1)
# Criação dos conjuntos de treino e testes

from sklearn.model_selection import train_test_split



x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.25, random_state=5)
# Criação do modelo #

#from sklearn.ensemble import ExtraTreesClassifier



#modelo = ExtraTreesClassifier(n_estimators=100)
#y_pred = modelo.fit(x_treino, y_treino) 

#accuracy = modelo.score(x_teste, y_teste)

#print(accuracy)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier #RandomForestClassifier
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5, 25,30,40, 50, 100, 250, 500]

# Write loop to find the ideal tree size from candidate_max_leaf_nodes



best_value = 0

controle = 0

for max_leaf_nodes in candidate_max_leaf_nodes:

    my_mae = get_mae(max_leaf_nodes, x_treino, x_teste, y_treino, y_teste)

                                    

    if best_value == 0:

        controle = my_mae

        best_value = max_leaf_nodes

    elif controle > my_mae:

        controle = my_mae

        best_value = max_leaf_nodes



#Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)

best_tree_size = best_value

best_tree_size
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error





# Fill in argument to make optimal size and uncomment

final_model = RandomForestClassifier(max_leaf_nodes=25,n_estimators=100, random_state=0)



# fit the final model and uncomment the next two lines

##y_pred =  final_model.fit(x_treino, y_treino)



##accuracy = final_model.score(x_teste, y_teste)

##print(accuracy)

final_model.fit(x_treino, y_treino)

Y_pred = final_model.predict(x_teste)

accuracy = final_model.score(x_treino, y_treino)

acc_random_forest = round(accuracy * 100, 2)

acc_random_forest

print(accuracy)
resultado = final_model.predict(clean_teste)

resultado

#print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



PassengerId = arquivos_de_teste['PassengerId']
output = pd.DataFrame({'PassengerId': clean_teste.PassengerId, 'Survived': resultado})

output.to_csv('francisco_submission.csv', index=False)

#print("Your submission was successfully saved!")

#output.head()
#Concatenando os dataframe 

#Df1 = pd.DataFrame(PassengerId, columns=['PassengerId'])

#Df1.index = np.arange(0,len(PassengerId))
#Df2 = pd.DataFrame(resultado, columns=['Survived'])

#Df2.index = np.arange(0,len(resultado))



 

#presubmissionDf = pd.concat([Df1,Df2], axis=1, ignore_index=False)



#Gerar arquivo de submission

#presubmissionDf.to_csv('francisco_submission.csv',index=False)