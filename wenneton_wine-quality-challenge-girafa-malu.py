# IMPORTS

from sklearn.neural_network import MLPClassifier # Multilayer Perceptron Classifier
from sklearn.model_selection import train_test_split # Separador do dataset para teste e treinamento
from sklearn.preprocessing import scale # Ferramenta para normalização dos dados
import pandas as pd #data processing, CSV file I/O
from sklearn.metrics import f1_score as f1 # métrica f-score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# ".../input/" necessário antes do nome do arquivo para funcionamento do Kernel
dataset = pd.read_csv('../input/train.csv',sep=',')

#determinando atributos e alvo e normalizando
y = dataset['quality']
x = dataset.drop(['id','quality','total sulfur dioxide'],axis=1)
#Vários testes realizados demostraram melhor desempenho sem o atributo 'total sulfur dioxide'
x = pd.DataFrame(scale(x)) #normalização dos dados de treinamento

print(x.head())
#separando conjunto de dados para treino e teste
xTreino, xTeste, yTreino, yTeste = train_test_split(x,y,test_size=0.3)
MLP = {

#camadas escondidas e quantidade de neurônios - tupla
#quantidade 1 ou 2 camadas 
'hidden_layer_sizes':1,

#função de ativação - {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
'activation':'relu',

#tipo de otimização - {‘lbfgs’, ‘sgd’, ‘adam’}
'solver':'sgd',

#não sei - float
'alpha':1e-5,

#quantidade de neurônios na otimização - int
#funciona somente para solver=‘sgd’ ou solver=‘adam’
#'auto' significa batch_size=min(200, n_samples)
'batch_size':'auto', 

#forma que a taxa de aprendizado vai ser atualizada - {‘constant’, ‘invscaling’, ‘adaptive’}
#funciona somente para solver=‘sgd’
'learning_rate':'adaptive',

#taxa de aprendizado inicial - double
#funciona somente para solver=‘sgd’ ou solver=‘adam’
'learning_rate_init':0.001,

#expoente para taxa de aprendizado de escala inversa - double
#funciona somente para solver=‘sgd’ e para learning_rate=‘invscaling’
'power_t':0.5,
    
#quantidade máxima de épocas - int
'max_iter':10000,

#Exemplos randomizados em cada época - bool
#funciona somente para solver=‘sgd’ ou solver=‘adam’
'shuffle':True,

#número para auxiliar na randomização - int ou None
'random_state':None,

#erro mínimo tolerado - float
#não funciona para learning_rate=‘adaptive’
'tol':1e-6,

#durante o treinamento vai printar algum resultado - bool
'verbose':False,
    
#reultiliza os resultados do treinamento anterior como inicialização - bool
'warm_start':False,
    
#peteleco para encontrar mínimos globais - float
#funciona somente para solver=‘sgd’
#deve estar entre 0 e 1
'momentum':0.9,

#um tipo específico de momentum - bool
#funciona somente para solver=‘sgd’ and momentum > 0
'nesterovs_momentum':True,

#parada antecipada caso não ocorra uma melhora no rendimento dos dados de validação - bool
#se o valor é True seta automaticamente validation_fraction=0.1
#funciona somente para solver=‘sgd’ ou solver=‘adam’
'early_stopping':False,

#fração de dados de validação - float
#deve estar entre 0 e 1
#funciona somente para early_stopping=True,
'validation_fraction':0.1,

#os três parâmetros abaixo têm a ver com o solver=‘adam’ - float
#não entendi nenhum deles 
'beta_1':0.9,
'beta_2':0.999,
'epsilon':1e-8
}
#função que cria as redes MLP no sci-kit learn
def create_network(MLP):
    rede = MLPClassifier(
    hidden_layer_sizes=MLP['hidden_layer_sizes'],
    activation=MLP['activation'],
    solver=MLP['solver'],
    alpha=MLP['alpha'],
    batch_size=MLP['batch_size'], 
    learning_rate=MLP['learning_rate'],
    learning_rate_init=MLP['learning_rate_init'],
    power_t=MLP['power_t'],
    max_iter=MLP['max_iter'],
    shuffle=MLP['shuffle'],
    random_state=MLP['random_state'],
    tol=MLP['tol'],
    verbose=MLP['verbose'],
    warm_start=MLP['warm_start'],
    momentum=MLP['momentum'],
    nesterovs_momentum=MLP['nesterovs_momentum'],
    early_stopping=MLP['early_stopping'],
    validation_fraction=MLP['validation_fraction'], 
    #beta_1=MLP['beta_1'],
    #beta_2=MLP['beta_2'],
    #epsilon=MLP['epsilon']
    )
    return rede
#exemplo de parâmetros redes
redes_parametros = [] #Parâmetros das redes a serem testadas
qtde_neuronios = [] #Qtde de neurônios em cada camada da rede correspondente ao índice
for i in range(6,10):
    aux = MLP.copy()
    aux['hidden_layer_sizes']=(i+1)
    redes_parametros.append(aux)
    qtde_neuronios.append(i+1)
    for j in range(3,6):
        aux = MLP.copy()
        aux['hidden_layer_sizes']=(i+1, j+1)
        redes_parametros.append(aux)        
        qtde_neuronios.append((i+1,j+1))
print(len(qtde_neuronios)) #Imprime quantidade total de redes
#criando as redes MLP
redes_MLP = []
for i in redes_parametros:
    redes_MLP.append(create_network(i))
#treinando as redes
metricas = []
for rede in redes_MLP:
    rede.fit(xTreino,yTreino)
    metrica = f1(yTeste, rede.predict(xTeste),average='micro')
    metricas.append(metrica)
#mostra as métricas de cada rede
for i in range(len(metricas)):
    print("Rede " + str(i) + ": Neurônios nas camadas " + str(qtde_neuronios[i]) + ", F-score " + str(metricas[i]))

#Importando o conjunto de testes
test = pd.read_csv("../input/test.csv")

idd = test['id']
test = test.drop(['id','total sulfur dioxide'], axis=1)
test = pd.DataFrame(scale(test)) #normalização

test.head() #visualização
#Treinamento e nova métrica
melhorRede = redes_MLP[12]

melhorRede.fit(x,y)

result = melhorRede.predict(test)
metrica = f1(yTeste, melhorRede.predict(xTeste),average='micro')
print(metrica) #Visualização da nova métrica obtida utilizando yTeste e xTeste
final = pd.DataFrame()

final['id'] = idd
final['quality'] = result

final.to_csv("submit.csv", sep=',', index=False)

final.head()
#Visualização da melhor rede encontrada nesse teste
print(melhorRede)
