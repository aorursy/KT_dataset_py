# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import numpy as np 
import pandas as pd
import datetime as dt
from sklearn import preprocessing as prep
import librosa as lb
import librosa.display as lbd
import librosa.feature as lbf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from collections import Counter
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib import rcParams
import plotly.offline
sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (16,8)
import IPython.display as ipd
import ipywidgets as ipw
import warnings
warnings.filterwarnings('ignore')

link = 'https://ebird.org/species/'
PATH_AUDIO = '../input/birdsong-recognition/train_audio/'
## configuring setup, constants and parameters
PATH_TRAIN = "../input/birdsong-recognition/train.csv"
PATH_TEST = "../input/birdsong-recognition/test.csv"

# PATH_TRAIN_EXTENDED = "../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv"


train = pd.read_csv(PATH_TRAIN)
train.head()
train.columns
len(set(train.ebird_code))
# zero_crossings = lb.zero_crossings(x[n0:n1], pad=False)
# print(sum(zero_crossings))
df_bird_map = train[["ebird_code", "species"]].drop_duplicates()

for ebird_code in os.listdir(PATH_AUDIO)[:20]:
    species = df_bird_map[df_bird_map.ebird_code == ebird_code].species.values[0]
    audio_file = os.listdir(f"{PATH_AUDIO}/{ebird_code}")[0]
    audio_path = f"{PATH_AUDIO}/{ebird_code}/{audio_file}"
    ipd.display(ipd.HTML(f"<h2>{ebird_code} ({species})</h2>"))
    ipd.display(ipd.Audio(audio_path))
    
def plot_for_one_species(Audio_path):
    values = Audio_path.split("/")
    ipd.display(ipd.HTML(f"<h2>{values[5]}</h2>"))
    ipd.display(ipd.Audio(Audio_path))
    data , samplingrate = lb.load(Audio_path)
    plt.figure(figsize=(12, 4))
    plt.title("Visuvalizing the Audio")
    lb.display.waveplot(data, sr=samplingrate)
    X = lb.stft(data)
    Xdb = lb.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    lb.display.specshow(Xdb, sr=samplingrate, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title("Spectrogram of the wave")
#     # Zooming in
#     n0 = 9000
#     n1 = 9100
#     plt.figure(figsize=(14, 5))
#     plt.plot(X[n0:n1])
#     plt.grid()
#     plt.title("Zero Crossing rate")
path = '/kaggle/input/birdsong-recognition/train_audio/nutwoo/XC462016.mp3'
plot_for_one_species(path)
fig = px.scatter(data_frame=train, x='longitude', y='latitude', color='ebird_code')
fig.show()
fig = px.choropleth(data_frame=train,locations="country",locationmode="country names",hover_name="species",title="Birds Location")
fig.show()
# displaying only the top 30 countries
country = train.country.value_counts()
country_df = pd.DataFrame({'country':country.index, 'frequency':country.values}).head(35)

fig = px.bar(country_df, x="frequency", y="country",color='country', orientation='h',
             hover_data=["country", "frequency"],
             height=1000,
             title='Number of audio samples besed on country of recording')
fig.show()


# displaying only the top 30 countries
authors = train.author.value_counts()
authors_df = pd.DataFrame({'authors':authors.index, 'frequency':authors.values}).head(35)
fig = px.bar(authors_df, x="frequency", y="authors",color='authors', orientation='h',
             hover_data=["authors", "frequency"],
             height=1000,
             title='Authors Contribution')
fig.show()


rcParams["figure.figsize"] = 20,8
train['ebird_code'].value_counts().plot(kind='hist')
plt.figure(figsize=(20, 8))
train['date'].value_counts().sort_index().plot();

# def parser(row):
#    # function to load files and extract features
#    file_name = os.path.join(os.path.abspath(data_dir), str(row.ebird_code), str(row.filename))

#    # handle exception to check if there isn't a file which is corrupted
#    try:
#         for i in tqdm(range(0,10)):
#             with joblib.parallel_backend('dask'):
#               # here kaiser_fast is a technique used for faster extraction
#               X, sample_rate = lb.load(file_name, res_type='kaiser_fast') 
#               # we extract mfcc feature from data
#               mfccs = np.mean(lb.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
#    except Exception as e:
#       print("Error encountered while parsing file: ", file)
#       return None, None
 
#    feature = mfccs
#    label = row.ebird_code
 
#    return [feature, label]

# temp = train.apply(parser, axis=1)
# temp.columns = ['feature', 'label']

