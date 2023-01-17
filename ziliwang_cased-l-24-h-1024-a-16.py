!wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
!unzip cased_L-24_H-1024_A-16.zip
!rm cased_L-24_H-1024_A-16.zip
!wget https://raw.githubusercontent.com/huggingface/pytorch-pretrained-BERT/master/pytorch_pretrained_bert/convert_tf_checkpoint_to_pytorch.py
!pip install pytorch-pretrained-bert -q
!python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path cased_L-24_H-1024_A-16/bert_model.ckpt --bert_config_file cased_L-24_H-1024_A-16/bert_config.json --pytorch_dump_path pytorch_model.bin
!ls -l
!cp cased_L-24_H-1024_A-16/bert_config.json ./

!cp cased_L-24_H-1024_A-16/vocab.txt ./
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
m = BertModel.from_pretrained('./')
m
!rm -rf cased_L-24_H-1024_A-16/
!rm convert_tf_checkpoint_to_pytorch.py
!ls -l