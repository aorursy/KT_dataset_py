!git clone https://github.com/huggingface/pytorch-transformers.git
%cd pytorch-transformers/
# !pip install -U torch
!pip install -U .
!pip install -r ./examples/requirements.txt
!python ./examples/run_generation.py  --prompt=Hello --length=10 --model_type=xlnet --model_name_or_path=xlnet-large-cased
!python ./examples/run_generation.py  --prompt=Hello --length=10 --model_type=transfo-xl --model_name_or_path=transfo-xl-wt103
# !python ./examples/run_generation.py  --prompt=Hello --length=10 --model_type=gpt2 --model_name_or_path=gpt2-large

!python ./examples/run_generation.py  --prompt=Hello --length=10 --model_type=gpt2 --model_name_or_path=gpt2
!python ./examples/run_generation.py  --prompt=Hello --length=10 --model_type=openai-gpt --model_name_or_path=openai-gpt
!ls ..
!rm -Rf * .??*