import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!pip install bert-serving-server

!pip install bert-serving-client
bert_server_cmd = 'bert-serving-start -model_dir /kaggle/input'

# !{bert_server_cmd}
# Start the BERT server

import subprocess

process = subprocess.Popen(bert_server_cmd.split(), stdout=subprocess.PIPE)
# Start the BERT client

from bert_serving.client import BertClient

bc = BertClient(ip='0.0.0.0')
texts = ['まずスタートして',

         'それから正しく作って',

         'そして、改善を行う']



bc.encode(texts)