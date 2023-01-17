# importando as bibliotecas necessarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.metrics import mean_absolute_error
import tensorflow as tf
import datetime, os
# fazendo a leitura dos dados nos diretórios
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
        #print(os.path.join(dirname, filename))
# criando os dataframes de teste e de treino
df_train = pd.read_csv('/kaggle/input/i2a2-bone-age-regression/train.csv')
df_test = pd.read_csv('/kaggle/input/i2a2-bone-age-regression/test.csv')

# visualizando o formato linhas por colunas
print("treino:" + str(df_train.shape))
print("teste:" + str(df_test.shape))
# visualizando uma amostra da base de treino, mudando a variavel do .sample() é possivel plotar mais imagens
for fileName, patientSex, boneage in df_train[['fileName','patientSex','boneage']].sample(1).values:
    img_name = str(fileName)
    img = mpimg.imread("/kaggle/input/i2a2-bone-age-regression/images/"+img_name)
    plt.imshow(img)
    plt.title('Image: {} Boneage: {} Male: {}'.format(fileName, patientSex, boneage))
    plt.show()
# isualizando os 5primeiros itens do dataframe
df_train.head()
# visualizando uma amostra da base de teste, mudando a variavel do .sample() é possivel plotar mais imagens
# possivel ver que em alguns momentos apareceram duas mãos nas imagens e algumas de cabeça para baixo
for fileName, patientSex in df_test[['fileName','patientSex']].sample(1).values:
    img_name = str(fileName)
    img = mpimg.imread("/kaggle/input/i2a2-bone-age-regression/images/"+img_name)
    plt.imshow(img)
    plt.title('Image: {} Sex: {}'.format(fileName, patientSex))
    plt.show()
df_test.head()
# criando um diretório de saída para as imagens de teste
# caso a pasta já exista ira gerar um erro... FileExistsError: [Errno 17] File exists: '../output/working/preview'
# então rode uma vez apenas...
os.makedirs('../output/working/preview')
# Fazendo o processo de corte das imagens para tentar obter apenas as imagens da mão esquerda, ou do lado esquerdo.
# é uma abordagem bem simplista...
for row in df_test.iterrows():
    img_name = row[1][0]
    img = mpimg.imread("/kaggle/input/i2a2-bone-age-regression/images/"+img_name)
    h, w = img.shape
    
    # se a largura da imagem estiver entre 800 e 900 utilizei dois terços da largura
    if w > 800 and w < 900:
        img = img[0:int(h),0:int(w-(w/3))]
        #plt.imshow(img)
        #plt.title('Image: {} Sex: {}'.format(row[1][0], row[1][1]))
        #plt.show()
        #print(img.shape)
        mpimg.imsave('../output/working/preview/'+img_name, img)
        
    # se a largura da imagem estiver acima 900 utilizei metade    
    elif w > 900:
        img = img[0:int(h),0:int(w/2)]
        #plt.imshow(img)
        #plt.title('Image: {} Sex: {}'.format(row[1][0], row[1][1]))
        #plt.show()
        #print(img.shape)
        mpimg.imsave('../output/working/preview/'+img_name, img)
        
    # caso seja menor que 800 é utilizada inteira
    else:
        #plt.imshow(img)
        #plt.title('Image: {} Sex: {}'.format(row[1][0], row[1][1]))
        #plt.show()
        #print(img.shape)
        mpimg.imsave('../output/working/preview/'+img_name, img)
# fazendo leitura dos dados no diretório criado para armazenar as imagens
for dirname, _, filenames in os.walk('/kaggle/output'):
    for filename in filenames:
        os.path.join(dirname, filename)
        #print(os.path.join(dirname, filename))
# total de arquivos... 249 
file_count = len(filenames)
file_count
# visualizando os dados na pasta que acabamos de salvar
for fileName, patientSex in df_test[['fileName','patientSex']].sample(3).values:
    img_name = str(fileName)
    img = mpimg.imread("/kaggle/output/working/preview/"+img_name)
    plt.imshow(img)
    plt.title('Image: {} Sex: {}'.format(fileName, patientSex))
    plt.show()
# maior idade do dataset
print('MAX age: ' + str(df_train['boneage'].max()) + ' months')

# menor idade do dataset
print('MIN age: ' + str(df_train['boneage'].min()) + ' months')

# idade media
mean_bone_age = df_train['boneage'].mean()
print('mean: ' + str(mean_bone_age))

# mediana
print('median: ' +str(df_train['boneage'].median()))

# desvio padrão das idades
std_bone_age = df_train['boneage'].std()

# modelos podem performar melho quando são normalizados os dados
df_train['bone_age_z'] = (df_train['boneage'] - mean_bone_age)/(std_bone_age)

# visualizando o dataset novamente
print(df_train.head())
# dividindo o dataframe em uma base de treino e uma de validação, 20% para validar e 80% para treinar
dfTrain, dfValid = train_test_split(df_train, test_size = 0.2, random_state = 0)
# tamanho da imagem a ser utilizado
img_size = 256

# criando objetos para receber a função ImageDataGenerator() a serem usados no treinamento e validação
train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)
val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

# treino - data generator
train_generator = train_data_generator.flow_from_dataframe(
    dataframe = dfTrain,
    directory = '/kaggle/input/i2a2-bone-age-regression/images/', 
    x_col= 'fileName',
    y_col= 'bone_age_z',
    batch_size = 32,
    seed = 42,
    shuffle = True,
    class_mode= 'other',
    flip_vertical = True,
    color_mode = 'rgb',
    target_size = (img_size, img_size))

# validador - data generator
val_generator = val_data_generator.flow_from_dataframe(
    dataframe = dfValid,
    directory = '/kaggle/input/i2a2-bone-age-regression/images/',
    x_col = 'fileName',
    y_col = 'bone_age_z',
    batch_size = 32,
    seed = 42,
    shuffle = True,
    class_mode = 'other',
    flip_vertical = True,
    color_mode = 'rgb',
    target_size = (img_size, img_size))

# test - data generator
test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

# é daqui que vai sair a base de teste para gerar o data set de submissão
test_generator = val_data_generator.flow_from_dataframe(
    dataframe = df_test,
    directory = '/kaggle/output/working/preview/',
    x_col = 'fileName',
    y_col = 'patientSex',
    batch_size = 32,
    #seed = 42,
    shuffle = True,
    class_mode = 'other',
    flip_vertical = True,
    color_mode = 'rgb',
    target_size = (img_size, img_size))
# separando o validador em x e y
test_X, test_Y = next(val_data_generator.flow_from_dataframe( 
                            dfValid, 
                            directory = '/kaggle/input/i2a2-bone-age-regression/images/',
                            x_col = 'fileName',
                            y_col = 'bone_age_z', 
                            target_size = (img_size, img_size),
                            batch_size = 2523,
                            class_mode = 'other'
                            )) 
# visualizando o formato de X
print(test_X.shape)

# visualizando o formato de Y
print(test_Y.shape)
# função para plotar o histórico do aprendizado
def plot_it(history):
    '''function to plot training and validation error'''
    fig, ax = plt.subplots( figsize=(20,10))
    ax.plot(history.history['mae_in_months'])
    ax.plot(history.history['val_mae_in_months'])
    plt.title('Model Error')
    plt.ylabel('error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    ax.grid(color='black')
    plt.show()
# função que retorna o erro medio absoluto em meses
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 
# importando mais algumas libs necessarias
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential

# importando a Xception pré-treinada
model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),
                                           include_top = False,
                                           weights = 'imagenet')

# nesta etapa foi combinada a rede pré-treinada model1 com as camadas model2
model_1.trainable = True
model_2 = Sequential()
# adiciona a rede pré-treinada na nova rede model2
model_2.add(model_1)
# criando ultimas camadas 
model_2.add(GlobalMaxPooling2D())
model_2.add(Flatten())
model_2.add(Dense(10, activation = 'relu'))
model_2.add(Dense(1, activation = 'linear'))


# compilando o modelo
model_2.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months] )

# visualizando o sumario da rede
model_2.summary()
# armazenando saidas, comandos para visuaizar o tensorboard que não estão fucnionando
# https://www.tensorflow.org/tensorboard/get_started

#%load_ext tensorboard
logs_dir = '.\logs'
#%tensorboard --logdir {logs_dir}
# early stopping - efetuada a parada do treinamento quando o value_loss não obtem mais melhorias
early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience= 5,
                              verbose=0, mode='auto')

# model checkpoint - armazena o melhor modelo ou peso treinado para ser usado no teste final
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

#tensorboard callback -->> não entendo bem de como fucniona o tensor board mas foi necessario manter aqui no código para mantero o callback
# 
logdir = os.path.join(logs_dir,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(logdir, histogram_freq = 1)

#reduce lr on plateau - aplica a redução da taxa de aprendizado quando a metrica para de ser melhorada, é aplicado a cada 10 epocas
# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau
red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
callbacks = [tensorboard_callback,early_stopping,mc, red_lr_plat]


#fit model
history = model_2.fit_generator(train_generator,
                            steps_per_epoch = 300,
                            validation_data = val_generator,
                            validation_steps = 1,
                            epochs = 50,
                            callbacks= callbacks)

# mostrando o treinamento
history
#%tensorboard --logdir logs
plot_it(history)
# model_2.load_weights('best_model.h5')
# calculo para retornar o valor correto da predição devido a normalização feita anteriormente
pred = mean_bone_age + std_bone_age*(model_2.predict(test_X, batch_size = 32, verbose = True))
test_months = mean_bone_age + std_bone_age*(test_Y)

# obs.: algumas etapas neste trecho não comprendi muito bem estou estudando ainda.
ord_ind = np.argsort(test_Y)
ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)] # take 8 evenly spaced ones
fig, axs = plt.subplots(4, 2, figsize = (15, 30))
for (ind, ax) in zip(ord_ind, axs.flatten()):
    ax.imshow(test_X[ind, :,:,0], cmap = 'bone')
    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, 
                                                           pred[ind]/12.0))
    ax.axis('off')
fig.savefig('trained_image_predictions.png', dpi = 300)
# Plotando algumas imagens para visualização dos resultados
fig, ax = plt.subplots(figsize = (7,7))
ax.plot(test_months, pred, 'r.', label = 'predictions')
ax.plot(test_months, test_months, 'b-', label = 'actual')
ax.legend(loc = 'upper right')
ax.set_xlabel('Actual Age (Months)')
ax.set_ylabel('Predicted Age (Months)')
# utilizando o test_generator criado etapas acima
test_generator.reset()
y_pred = model_2.predict_generator(test_generator)
predicted = y_pred.flatten()
predicted_months = mean_bone_age + std_bone_age*(predicted)
filenames=test_generator.filenames
results=pd.DataFrame({"fileName":filenames,
                      "boneage": predicted_months})
# salvando o resultado para submeter no kaggle, o arquivo ficara no outputs basta efetuar o download.
results.to_csv("results.csv",index=False)
# caso queira remover alguma coisa do diretório de saida

# os.remove("/kaggle/working/results2.csv")
# Visualizando o dataframe de results para ter certeza que esta com nomes de coluns corretos
results.head()
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics

# raiz do erro médio quadrado
rmse = sqrt(mean_squared_error(test_months, pred))
rmse
# erro maximo
metrics.max_error(test_months, pred)
# r score
metrics.r2_score(test_months, pred)
# erro medio quadrado
metrics.mean_squared_error(test_months, pred)
# erro medio absoluto
metrics.mean_absolute_error(test_months, pred)