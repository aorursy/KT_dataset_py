import pandas as pd

import sklearn
adult_teste = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=[0])



adult_treino = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?", 

        skiprows=[0])



adult_treino




pais_rpc = ['United-States', 6, 'Cambodia', 3, 'England', 5, 'Puerto-Rico', 5, 'Canada', 5,

    'Germany', 6, 'Outlying-US(Guam-USVI-etc)', 5, 'India', 3, 'Japan', 6, 'Greece', 5,

    'South', 4, 'China', 3, 'Cuba', 4, 'Iran', 4, 'Honduras', 3, 'Philippines', 3, 'Italy', 5,

    'Poland', 4, 'Jamaica', 4, 'Vietnam', 3, 'Mexico', 4.5, 'Portugal', 5, 'Ireland', 5,

    'France', 5, 'Dominican-Republic', 4, 'Laos', 3, 'Ecuador', 4, 'Taiwan', 5, 'Haiti', 3,

    'Columbia', 4, 'Hungary', 4, 'Guatemala', 4, 'Nicaragua', 3, 'Scotland', 5, 'Thailand', 4,

    'Yugoslavia', 4, 'El-Salvador', 4, 'Trinadad&Tobago', 4, 'Peru', 4, 'Hong', 5,

    'Holand-Netherlands', 5]



raca = ['White', 0, 'Asian-Pac-Islander',  1, 'Amer-Indian-Eskimo', 1, 'Other', 2, 'Black', 2]



classe = ['Private', 7, 'Self-emp-not-inc', 5, 'Self-emp-inc', 5, 'Federal-gov', 6, 'Local-gov', 4, 'State-gov', 5,

    'Without-pay', -5, 'Never-worked', -5]



relacao= ['Wife', 4, 'Own-child', 5,'Husband', 3,'Not-in-family', 1,'Other-relative', 2,'Unmarried', 0]



ocupacao=['Tech-support', 2,'Craft-repair', 2,'Other-service', 2,'Sales', 3,'Exec-managerial', 5,'Prof-specialty', 5,

   'Handlers-cleaners', 2,'Machine-op-inspct', 2,'Adm-clerical', 4,'Farming-fishing', 1,'Transport-moving', 2,

   'Priv-house-serv', 2,'Protective-serv', 4,'Armed-Forces', 5]



status = ['Married-civ-spouse', 4,'Divorced', 1,'Never-married', 0,'Separated', 2,'Widowed', 6,

          'Married-spouse-absent', 3,'Married-AF-spouse', 5]



adult_treino=adult_treino.replace(['Male', 'Female'],[1, 0])

adult_treino=adult_treino.replace(['<=50K', '>50K'],[1, 0])



adult_teste=adult_teste.replace(['Male', 'Female'],[1, 0])

adult_teste=adult_teste.replace(['<=50K', '>50K'],[1, 0])







def substitui(lista):

    l_old=[]

    l_new=[]

    for i in range(len(lista)):

        if i%2 == 0:

            l_old.append(lista[i])

        else:

            l_new.append(lista[i])

        i+=1

    return(l_old, l_new)



lista_rel, lista_num_rel = substitui(relacao)

adult_treino = adult_treino.replace(lista_rel, lista_num_rel)

adult_teste = adult_teste.replace(lista_rel, lista_num_rel)

   

lista_paises, lista_rpc=substitui(pais_rpc)

adult_treino = adult_treino.replace(lista_paises, lista_rpc)

adult_teste = adult_teste.replace(lista_paises, lista_rpc)



lista_raca, lista_num_raca=substitui(raca)

adult_treino = adult_treino.replace(lista_raca, lista_num_raca)

adult_teste = adult_teste.replace(lista_raca, lista_num_raca)



lista_classe, lista_num_classe=substitui(classe)

adult_treino=adult_treino.replace(lista_classe,lista_num_classe)

adult_teste=adult_teste.replace(lista_classe,lista_num_classe)



lista_ocup, lista_num_ocup=substitui(ocupacao)

adult_treino=adult_treino.replace(lista_ocup, lista_num_ocup)

adult_teste=adult_teste.replace(lista_ocup, lista_num_ocup)



lista_status, lista_num_status=substitui(status)

adult_treino=adult_treino.replace(lista_status, lista_num_status)

adult_teste=adult_teste.replace(lista_status, lista_num_status)



adult_treino.mean(axis = 0, skipna = True)

adult_teste.mean(axis = 0, skipna = True)



dados_treino=adult_treino.drop('Education', axis=1)

dados_teste=adult_teste.drop('Education', axis=1)

dados_treino=dados_treino.drop('Id', axis=1)

dados_teste=dados_teste.drop('Id', axis=1)

dados_treino=dados_treino.drop(0, axis=0)

dados_teste=dados_teste.drop(0, axis=0)

dados_treino.dropna(inplace=True)

dados_teste.dropna(inplace=True)



dados_treino['Workclass'].fillna(6.433653, inplace = True)

dados_treino['Occupation'].fillna(3.176476, inplace = True)

dados_treino['Country'].fillna(5.848349, inplace = True)

dados_treino.isnull().sum(axis = 0)

dados_treino.drop_duplicates(inplace=True)



dados_teste['Workclass'].fillna(6.433653, inplace = True)

dados_teste['Occupation'].fillna(3.176476, inplace = True)

dados_teste['Country'].fillna(5.848349, inplace = True)

dados_teste.isnull().sum(axis = 0)

dados_teste.drop_duplicates(inplace=True)



dados_treino.dropna(inplace=True)

dados_teste.dropna(inplace=True)





dados_treino

# KNN





from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score





Xtreino = dados_treino[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation","Race","Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Ytreino= dados_treino.Target

Xteste = dados_teste[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation","Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Yteste=dados_teste.Target





knn = KNeighborsClassifier(n_neighbors=25, p=1)

knn.fit(Xtreino,Ytreino)

scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)

scores

YtestePred = knn.predict(Xteste)

accuracy_score(Yteste,YtestePred)
# REDES NEURAIS



from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(solver="adam", alpha=0.0001, hidden_layer_sizes=(5,),

                   random_state=1, learning_rate='constant', learning_rate_init=0.01,

                   max_iter=50, activation='logistic', momentum=0.9,verbose=True,

                   tol=0.0001)



mlp.fit(Xtreino, Ytreino)

saidas = mlp.predict(Xteste)





print("Score: ", (saidas ==Yteste ).sum()/len(Xteste))





# SVM



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC



svmClassifier = SVC()

svmClassifier.fit(Xtreino,Ytreino)

prediction = svmClassifier.predict(Xteste)

print(confusion_matrix(Yteste, prediction))



print(classification_report(Yteste, prediction))

print('Score =',round(accuracy_score(Yteste, prediction), 2))


