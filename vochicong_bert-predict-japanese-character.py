!pip install pytorch-transformers
import torch

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM



def load_pretrained(model='bert-base-uncased'):

    # Load pre-trained model tokenizer (vocabulary)

    tokenizer = BertTokenizer.from_pretrained(model)

    # Load pre-trained model (weights)

    model = BertForMaskedLM.from_pretrained(model)

    model.eval()

    return tokenizer, model



en_tokenizer, en_model = load_pretrained('bert-base-uncased')



def predict(text, tokenizer=en_tokenizer, model=en_model):

    text = "[CLS] " + text + " [SEP]"

    tokenized_text = tokenizer.tokenize(text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)



    # Create the segments tensors.

    segments_ids = [0] * len(tokenized_text)



    # Convert inputs to PyTorch tensors

    tokens_tensor = torch.tensor([indexed_tokens])

    segments_tensors = torch.tensor([segments_ids])



    # If you have a GPU, put everything on cuda

    tokens_tensor = tokens_tensor.to('cuda')

    segments_tensors = segments_tensors.to('cuda')

    model.to('cuda')



    # Predict all tokens

    with torch.no_grad():

        predictions = model(tokens_tensor, segments_tensors)



    masked_index = tokenized_text.index('[MASK]') 



    predicted_index = torch.argmax(predictions[0][0][masked_index]).item()

    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]



    return predicted_token
predict('I want to [MASK] the car')
predict('I want to [MASK] the car because it is cheap')
predict('I want to [MASK] the car because it is cheap.')
multilingual_tokenizer, multilingual_model = load_pretrained('bert-base-multilingual-cased')

def multilingual_predict(text):

    return predict(text, multilingual_tokenizer, multilingual_model)
multilingual_predict("安いからこの車を[MASK]いたい")
multilingual_predict("彼は日本[MASK]を母国語として話す")
multilingual_predict("彼は優しくて可愛い彼女をとても[MASK]きになりました。")
multilingual_predict("彼は醜くてわがままな彼女をとても[MASK]いになりました。")
multilingual_predict('I want to [MASK] the car because it is cheap.')
multilingual_predict("Tôi muốn [MASK] cái ô tô vì nó rất rẻ.")
multilingual_predict("Tôi yêu tiếng [MASK] vì tôi là người Việt.")
multilingual_predict("Tôi rất [MASK] mến cô gái ấy.")
cn_tokenizer, cn_model = load_pretrained('bert-base-chinese')



def cn_predict(text):

    return predict(text, cn_tokenizer, cn_model)
cn_predict("我[MASK]你，因为你很漂亮") # 爱
cn_predict("我[MASK]你")