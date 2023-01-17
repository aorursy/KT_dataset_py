from transformers import pipeline
nlp = pipeline("sentiment-analysis")
result = nlp("Everyone hates you")

result
result = nlp("Your family loves you")

result
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
classes = ["Not_Paraphrase", "Paraphrase"]
sentence1 = "A reply from you is what I'm expecting"

sentence2 = "You are very cheerful today"

sentence3 = "I am awaiting a response from you"
paraphrase = tokenizer(sentence1, sentence3, return_tensors="pt")

not_paraphrase = tokenizer(sentence1, sentence2, return_tensors="pt")
paraphrase
paraphrase_model = model(**paraphrase)

nonparaphrase_model = model(**not_paraphrase)
paraphrase_model, nonparaphrase_model
paraphrase_result = torch.softmax(paraphrase_model[0], dim=1).tolist()[0]

nonparaphrase_result = torch.softmax(nonparaphrase_model[0], dim=1).tolist()[0]
paraphrase_result
# Paraphrase output

for i in range(len(classes)):

    print(f"{classes[i]}: {paraphrase_result[i] * 100:.2f}%")
# Non Paraphrase output

for i in range(len(classes)):

    print(f"{classes[i]}: {paraphrase_result[i] * 100:.2f}%")
nlp = pipeline("question-answering")



context = r'''Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 

to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. 

Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, 

which consisted of Skylab, a space station that supported three manned missions in 1973–74, and the Apollo–Soyuz Test Project, 

a joint Earth orbit mission with the Soviet Union in 1975'''
result = nlp(question="What space station supported three manned missions in 1973–1974?", context=context)

result
result = nlp(question="What is Apollo–Soyuz Test Project?", context=context)

result
from transformers import AutoModelForQuestionAnswering



tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
questions = ["What space station supported three manned missions in 1973–1974?", "What is Apollo–Soyuz Test Project?",

             "What are Gemini missions?"]
for question in questions:

    inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")

    input_ids = inputs["input_ids"].tolist()[0]

    

    answer_start_scores, answer_end_scores = model(**inputs)

    

    # Get the likely beginning of answer

    answer_start = torch.argmax(answer_start_scores) 

    # Get the likely end of answer

    answer_end = torch.argmax(answer_end_scores) + 1

    

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))



    print(f"Question: {question}")

    print(f"Answer: {answer}")
from transformers import pipeline

nlp = pipeline("fill-mask")
from pprint import pprint

pprint(nlp(f"Learning another {nlp.tokenizer.mask_token} is like becoming another person"))
pprint(nlp(f"I love Kaggle because it gives me {nlp.tokenizer.mask_token}"))
from transformers import AutoModelWithLMHead, AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")



sequence = f"Masked language modeling is the task of masking tokens in a sequence with a masking token, and prompting the model to fill {tokenizer.mask_token} with an appropriate token"



input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_id = torch.where(input == tokenizer.mask_token_id)[1]

model_output = model(input)[0]
mask_token_logits = model_output[0, mask_token_id, :]

print(mask_token_logits)
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:

    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering



from torch.nn import functional as F



tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelWithLMHead.from_pretrained("gpt2")

 

sequence = f"Psychology is the study of human "



input_ids = tokenizer.encode(sequence, return_tensors="pt")
model(input_ids)[0][:, -1, :]
# get logits of last hidden state

next_token_logits = model(input_ids)[0][:, -1, :]



filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)



probs = F.softmax(filtered_next_token_logits, dim=-1)

next_token = torch.multinomial(probs, num_samples=1)



generated = torch.cat([input_ids, next_token], dim=-1)



resulting_string = tokenizer.decode(generated.tolist()[0])



print(resulting_string)