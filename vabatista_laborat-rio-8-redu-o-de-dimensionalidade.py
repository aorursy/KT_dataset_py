import pandas as pd
df_dados = pd.read_csv('../input/lab8_neuro.txt', header=None, sep=' ' ) # nosso dado não possui cabeçalho.
df_dados.head()
df_dados.tail()
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Essa função apenas prepara a exibição de nossos gráficos
def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999',
                gridWidth=1.0):

    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax
# recuperamos a coluna de indice 2, que contem os dados em t0 e depois 
# redimensionamos para o formato original da imagem 230 x 202
imagem_t0 = df_dados[2].values.reshape(230,202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Imagem em t=0', color='#888888')
image = plt.imshow(imagem_t0,interpolation='nearest', aspect='auto', cmap=cm.gray)
pass
#calculamos o desvio padrão da luminosidade em cada linha (axis=1).
std_por_pixel = np.std(df_dados.values[:,2:], axis=1) 
#aqui selecionamos os pontos cujo desvio padrão é maior que 100. a Variável guarda os indices.
pontos_alto_std = np.where(std_por_pixel>100)[0]
pixel_escolhido =  df_dados.iloc[np.random.choice(pontos_alto_std)][2:].values
# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(300, 800, 100))
ax.set_xlabel(r'tempo'), ax.set_ylabel(r'luminosidade')
ax.set_xlim(-20, 270), ax.set_ylim(270, 900)
plt.plot(range(len(pixel_escolhido)), pixel_escolhido, c='#8cbfd0', linewidth='3.0')
pass
dados = df_dados.values[:,2:]
print(dados)
print(dados.shape)
#calculamos a média da luminosidade em cada linha (axis=1).
media_por_pixel = np.mean(dados, axis=1) 

#Agora ajustamos os dados
dados_reescalonados = ((dados.T - media_por_pixel)/media_por_pixel).T
pixel_escolhido =  dados_reescalonados[np.random.choice(pontos_alto_std)]
# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 300, 50), np.arange(-.1, .6, .1))
ax.set_xlabel(r'tempo'), ax.set_ylabel(r'luminosidade')
ax.set_xlim(-20, 260), ax.set_ylim(-.30, .80)
plt.plot(range(len(pixel_escolhido)), pixel_escolhido, c='#8cbfd0', linewidth='3.0')
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
componentes = pca.fit(dados_reescalonados).transform(dados_reescalonados)
print(pca.explained_variance_ratio_) 
primeiro_componente = componentes[:,0].reshape(230,202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Primeiro componente principal', color='#888888')
image = plt.imshow(primeiro_componente,interpolation='nearest', aspect='auto', cmap=cm.gray)
segundo_componente = componentes[:,1].reshape(230,202).T

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
ax.set_title('Segundo componente principal', color='#888888')
image = plt.imshow(segundo_componente,interpolation='nearest', aspect='auto', cmap=cm.gray)
pass
def polarTransform(scale, img):
    """Convert points from cartesian to polar coordinates and map to colors."""
    from matplotlib.colors import hsv_to_rgb

    img = np.asarray(img)
    dims = img.shape

    phi = ((np.arctan2(-img[0], -img[1]) + np.pi/2) % (np.pi*2)) / (2 * np.pi)
    rho = np.sqrt(img[0]**2 + img[1]**2)
    saturation = np.ones((dims[1], dims[2]))

    out = hsv_to_rgb(np.dstack((phi, saturation, scale * rho)))

    return np.clip(out * scale, 0, 1)

# Show the polar mapping from principal component coordinates to colors.
x1AbsMax = np.max(np.abs(primeiro_componente))
x2AbsMax = np.max(np.abs(segundo_componente))

numOfPixels = 300
x1Vals = np.arange(-x1AbsMax, x1AbsMax, (2 * x1AbsMax) / numOfPixels)
x2Vals = np.arange(x2AbsMax, -x2AbsMax, -(2 * x2AbsMax) / numOfPixels)
x2Vals.shape = (numOfPixels, 1)

x1Data = np.tile(x1Vals, (numOfPixels, 1))
x2Data = np.tile(x2Vals, (1, numOfPixels))

# Try changing the first parameter to lower values
polarMap = polarTransform(2.0, [x1Data, x2Data])

gridRange = np.arange(0, numOfPixels + 25, 25)
fig, ax = preparePlot(gridRange, gridRange, figsize=(9.0, 7.2), hideLabels=True)
image = plt.imshow(polarMap, interpolation='nearest', aspect='auto')
ax.set_xlabel('Primeiro componente principal'), ax.set_ylabel('Segundo componente principal')
gridMarks = (2 * gridRange / float(numOfPixels) - 1.0)
x1Marks = x1AbsMax * gridMarks
x2Marks = -x2AbsMax * gridMarks
ax.get_xaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x1Marks))
ax.get_yaxis().set_ticklabels(map(lambda x: '{0:.1f}'.format(x), x2Marks))
brainmap = polarTransform(2.0, [primeiro_componente, segundo_componente])

# generate layout and plot data
fig, ax = preparePlot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hideLabels=True)
ax.grid(False)
image = plt.imshow(brainmap,interpolation='nearest', aspect='auto')
pass

