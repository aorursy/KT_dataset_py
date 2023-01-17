from google.colab import drive
drive.mount('/content/drive')
% cd /content/drive/My Drive/Colab_Notebooks/dog-cat
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!rm kaggle.json
!kaggle datasets download -d biaiscience/dogs-vs-cats
from zipfile import ZipFile
file_name="/content/drive/My Drive/Colab_Notebooks/dog-cat/dogs-vs-cats.zip"

with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('Done')