from google.colab import drive

drive.mount('/content/drive/')
path = "/content/drive/My Drive/Colab Notebooks/.csv"

train = pd.read_csv(path)