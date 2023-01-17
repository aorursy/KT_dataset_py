# pip installation of torch 1.6
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#Pip installation of torch 1.6 indicates incompatibility of kornia and allennlp. Hence upgrading them
!pip install --upgrade kornia
!pip install allennlp==1.1.0.rc4
# installing/upgrading fastai2 
!pip install --upgrade fastai
#Importing Torch and checking CUDA availability and version
import torch
print(torch.__version__)
print(torch.cuda.is_available())
#Importing the Fast AI library
import fastai
fastai.__version__