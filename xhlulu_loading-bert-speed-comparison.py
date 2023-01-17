MODEL_NAME = "bert-large-uncased"
%%time

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
%%time

from transformers import AutoTokenizer, AutoModelForMaskedLM

file_path = '/kaggle/input/huggingface-bert/'

tokenizer = AutoTokenizer.from_pretrained(file_path + MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(file_path + MODEL_NAME)