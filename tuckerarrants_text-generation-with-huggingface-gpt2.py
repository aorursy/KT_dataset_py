#!pip install transformers
#for reproducability

SEED = 34



#maximum number of words in output text

MAX_LEN = 70
input_sequence = "I don't know about you, but there's only one thing I want to do after a long day of work"
#get transformers

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer



#get large GPT2 tokenizer and GPT2 model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)



#tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id)



#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)



#view model parameters

GPT2.summary()
#get deep learning basics

import tensorflow as tf

tf.random.set_seed(SEED)
# encode context the generation is conditioned on

input_ids = tokenizer.encode(input_sequence, return_tensors='tf')



# generate text until the output length (which includes the context length) reaches 50

greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)



print("Output:\n" + 100 * '-')

print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))
# set return_num_sequences > 1

beam_outputs = GPT2.generate(

    input_ids, 

    max_length = MAX_LEN, 

    num_beams = 5, 

    no_repeat_ngram_size = 2, 

    num_return_sequences = 5, 

    early_stopping = True

)



print('')

print("Output:\n" + 100 * '-')



# now we have 3 output sequences

for i, beam_output in enumerate(beam_outputs):

      print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
# use temperature to decrease the sensitivity to low probability candidates

sample_output = GPT2.generate(

                             input_ids, 

                             do_sample = True, 

                             max_length = MAX_LEN, 

                             top_k = 0, 

                             temperature = 0.8

)



print("Output:\n" + 100 * '-')

print(tokenizer.decode(sample_output[0], skip_special_tokens = True))
#sample from only top_k most likely words

sample_output = GPT2.generate(

                             input_ids, 

                             do_sample = True, 

                             max_length = MAX_LEN, 

                             top_k = 50

)



print("Output:\n" + 100 * '-')

print(tokenizer.decode(sample_output[0], skip_special_tokens = True), '...')
#sample only from 80% most likely words

sample_output = GPT2.generate(

                             input_ids, 

                             do_sample = True, 

                             max_length = MAX_LEN, 

                             top_p = 0.8, 

                             top_k = 0

)



print("Output:\n" + 100 * '-')

print(tokenizer.decode(sample_output[0], skip_special_tokens = True), '...')
#combine both sampling techniques

sample_outputs = GPT2.generate(

                              input_ids,

                              do_sample = True, 

                              max_length = 2*MAX_LEN,                              #to test how long we can generate and it be coherent

                              #temperature = .7,

                              top_k = 50, 

                              top_p = 0.85, 

                              num_return_sequences = 5

)



print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):

    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

    print('')
MAX_LEN = 150
prompt1 = 'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.'



input_ids = tokenizer.encode(prompt1, return_tensors='tf')
sample_outputs = GPT2.generate(

                              input_ids,

                              do_sample = True, 

                              max_length = MAX_LEN,                              #to test how long we can generate and it be coherent

                              #temperature = .8,

                              top_k = 50, 

                              top_p = 0.85 

                              #num_return_sequences = 5

)



print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):

    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

    print('')
prompt2 = 'Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.'



input_ids = tokenizer.encode(prompt2, return_tensors='tf')
sample_outputs = GPT2.generate(

                              input_ids,

                              do_sample = True, 

                              max_length = MAX_LEN,                              #to test how long we can generate and it be coherent

                              #temperature = .8,

                              top_k = 50, 

                              top_p = 0.85

                              #num_return_sequences = 5

)



print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):

    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

    print('')
prompt3 = 'Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.'



input_ids = tokenizer.encode(prompt3, return_tensors='tf')
sample_outputs = GPT2.generate(

                              input_ids,

                              do_sample = True, 

                              max_length = MAX_LEN,                              #to test how long we can generate and it be coherent

                              #temperature = .8,

                              top_k = 50, 

                              top_p = 0.85 

                              #num_return_sequences = 5

)



print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):

    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

    print('')
prompt4 = "For today’s homework assignment, please describe the reasons for the US Civil War."



input_ids = tokenizer.encode(prompt4, return_tensors='tf')
sample_outputs = GPT2.generate(

                              input_ids,

                              do_sample = True, 

                              max_length = MAX_LEN,                              #to test how long we can generate and it be coherent

                              #temperature = .8,

                              top_k = 50, 

                              top_p = 0.85 

                              #num_return_sequences = 5

)



print("Output:\n" + 100 * '-')

for i, sample_output in enumerate(sample_outputs):

    print("{}: {}...".format(i, tokenizer.decode(sample_output, skip_special_tokens = True)))

    print('')