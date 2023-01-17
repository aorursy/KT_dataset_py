import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils import clip_grad_norm_

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



import pandas as pd

from gensim.models import FastText



from collections import Counter

from operator import itemgetter
# CPU

#device = torch.device("cpu")



# GPU

device = torch.device("cuda:0")
class PairDataset(Dataset):

    """

    Pair data를 위한 Custom dataset 클래스

    """

    

    def __init__(self, df):

        self.data = df

        self.is_test = 'label' not in df.columns

            

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        return (self.sen2idx(self.data['sentence1'][idx]),

                self.sen2idx(self.data['sentence2'][idx]),

                self.data['label'][idx] if not self.is_test else None)

    

    def sen2idx(self, sen):

        return [word2idx[word] if word in word2idx else word2idx['<UNK>'] for word in sen]
def preprocessing(data_path, min_vocab_freq=1, is_train=False, valid_size=5000):

    """

    Pandas를 통해 데이터를 로드하고 Vocab 생성

    """

    df = pd.read_csv(data_path,

                     header=0, delimiter=',')



    df['sentence1'] = [[int(idx) for idx in sen.split()] for sen in df['sentence1']]

    df['sentence2'] = [[int(idx) for idx in sen.split()] for sen in df['sentence2']]

    

    if is_train:

        df['label'] = [int(label) for label in df['label']]

        

        df_len = len(df)

        train_df = df.iloc[:df_len-valid_size, :]

        valid_df = df.iloc[df_len-valid_size:, 1:]

        valid_df = valid_df.reset_index().rename(columns={'index':'id'})

        valid_df['id'] = range(1, 5001)

        

        # 받은 데이터 자체가 Index이긴 하지만,

        # 출현빈도 순으로 다시 Index화 시킬 것이기 때문에 Word라 칭함.

        # Unknown token를 Padding token를 위해 +2에 Mapping

        cnt = Counter()

        for idx_list in train_df['sentence1']:

            cnt.update(idx_list)

        for idx_list in train_df['sentence2']:

            cnt.update(idx_list)

        

        vocab_dict = [(word, cnt) for word, cnt in cnt.most_common() if cnt >= min_vocab_freq]

        word2idx = {word:i+2 for i, (word, cnt) in enumerate(vocab_dict)}

        word2idx["<PAD>"] = 0

        word2idx["<UNK>"] = 1

        

        return train_df, valid_df, word2idx

    else:

        return df
def pair_collate_fn(chunk):

    """

    Custom Dataset을 위한 DataLoader에 사용할 Collate function

    """

    sen1_list, sen2_list, label_list = zip(*chunk)



    sen1_len_tensor = torch.tensor([len(sen) for sen in sen1_list])

    sen2_len_tensor = torch.tensor([len(sen) for sen in sen2_list])



    sen1_list = [torch.tensor(sen) for sen in sen1_list]

    sen2_list = [torch.tensor(sen) for sen in sen2_list]



    sen1_tensor = pad_sequence(sen1_list, batch_first=True, padding_value=0)

    sen2_tensor = pad_sequence(sen2_list, batch_first=True, padding_value=0)

    label_list = torch.tensor(label_list) if None not in label_list else None

    

    return (sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor, label_list)
data_dir = '/kaggle/input/hlr79wbwd822zke/{}'

train_path = data_dir.format('train.csv')

test_path = data_dir.format('test.csv')



train_df, valid_df, word2idx = preprocessing(train_path, min_vocab_freq=1, is_train=True)

test_df = preprocessing(test_path)



train = PairDataset(train_df)

valid = PairDataset(valid_df)

test = PairDataset(test_df)



train_data = DataLoader(train,

                        batch_size=16,

                        shuffle=True,

                        collate_fn=pair_collate_fn)



valid_data = DataLoader(valid,

                        batch_size=16,

                        shuffle=False,

                        collate_fn=pair_collate_fn)



test_data = DataLoader(test,

                       batch_size=16,

                       shuffle=False,

                       collate_fn=pair_collate_fn)
# Fasttext 적용

# data = pd.read_csv(train_path, header=0, delimiter=',')

# data['sentence1'] = [d.split() for d in data['sentence1']]

# data['sentence2'] = [d.split() for d in data['sentence2']]

# total_sentences = data['sentence1'] + data['sentence2']



# ft = FastText(total_sentences, size=128, sg=1, hs=1, sorted_vocab=1, window=3, min_count=1, workers=4)

# word2idx = {int(w):i+2 for i, w in enumerate(ft.wv.index2word)}

# word2idx["<PAD>"] = 0

# word2idx["<UNK>"] = 1
mode_list = ["manhattan", "logistic", "dot", "attn"]



class SiamaseNetwork(nn.Module):

    """

    4가지 종류의 Siamese network

    """

    

    def __init__(self, vocab_size, input_size, hidden_size, dropout_p, mode,

                 pretrained_word_embedding=None, is_freeze=False):

        super(SiamaseNetwork, self).__init__()

        

        self.embed = nn.Embedding(vocab_size, input_size, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout_p)

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)

        

        assert mode in mode_list

        self.mode = mode

        if mode == "logistic":

            self.out = nn.Linear(hidden_size*2*3, 1)

        elif mode == "attn":

            self.w_in = nn.Linear(hidden_size*2, hidden_size*2)

            self.w_attn = nn.Linear(hidden_size*4, hidden_size*2)

            self.out1 = nn.Linear(hidden_size*4, hidden_size*2)

            self.out2 = nn.Linear(hidden_size*2, 1)

        

        if pretrained_word_embedding is not None:

            self.embed.from_pretrained(torch.tensor(pretrained_word_embedding))

            self.embed.weight.requires_grad = not is_freeze

        

    def forward(self, sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor):

        sen1_outputs, sen1_hidden = self._forward_single(sen1_tensor, sen1_len_tensor)

        sen2_outputs, sen2_hidden = self._forward_single(sen2_tensor, sen2_len_tensor)

        

        if mode == "manhattan":

            out = self._manhattan_dist(sen1_hidden, sen2_hidden)

        elif mode == "logistic":

            final_vec = torch.cat((sen1_hidden, sen2_hidden, torch.abs(sen1_hidden-sen2_hidden)), dim=-1)

            out = torch.sigmoid(self.out(final_vec))

        elif mode == "dot":

            out = torch.bmm(sen1_hidden.unsqueeze(1), sen2_hidden.unsqueeze(2))

            out = torch.sigmoid(out)

        elif mode == "attn":

            sen1_attn = self._attention(sen2_hidden, sen1_outputs, sen1_tensor).squeeze()

            sen1_final = self.w_attn(torch.cat((sen1_hidden, sen1_attn), dim=-1))

            sen1_final = torch.tanh(sen1_final)



            sen2_attn = self._attention(sen1_hidden, sen2_outputs, sen2_tensor).squeeze()

            sen2_final = self.w_attn(torch.cat((sen2_hidden, sen2_attn), dim=-1))

            sen2_final = torch.tanh(sen2_final)



            out = torch.tanh(self.out1(torch.cat((sen1_final, sen2_final), dim=-1)))

            out = torch.sigmoid(self.out2(out))

        

        return out.squeeze()

        

    def _forward_single(self, sen_tensor, len_tensor):

        sen_tensor, len_tensor, sorted_idx = self._sort_batch(sen_tensor, len_tensor)

        

        embedded = self.dropout(self.embed(sen_tensor))

        packed = pack_padded_sequence(embedded, len_tensor, batch_first=True)

        outputs, hidden = self.gru(packed)

        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        

        # Bidirectional GRU hidden을 Concat 시킴

        hidden = torch.cat((hidden[:1], hidden[1:]), dim=-1).squeeze()

        

        outputs, hidden, len_tensor = self._unsort_batch(outputs, hidden, len_tensor, sorted_idx)

        return outputs, hidden

        

    def _manhattan_dist(self, tensor1, tensor2):

        return torch.exp(-torch.norm((tensor1 - tensor2), dim=-1))

    

    def _attention(self, hidden, outputs, sen_tensor):

        hidden_ = self.w_in(hidden)

        align = torch.bmm(hidden_.unsqueeze(1), outputs.transpose(1, 2))



        mask = (sen_tensor == 0).unsqueeze(1)

        align = torch.softmax(align.masked_fill(mask, float('-inf')), dim=-1)

        return torch.bmm(align, outputs)

        

    def _sort_batch(self, seq_tensor, len_tensor):

        len_tensor, sorted_idx = len_tensor.sort(dim=0, descending=True)

        seq_tensor = seq_tensor[sorted_idx]

        return (seq_tensor, len_tensor, sorted_idx)



    def _unsort_batch(self, outputs, hidden, len_tensor, sorted_idx):

        _, org_idx = sorted_idx.sort(dim=0)

        outputs = outputs[org_idx]

        hidden = hidden[org_idx]

        len_tensor = len_tensor[org_idx]

        return (outputs, hidden, len_tensor)
def train_model(net, num_epoch, criterion, optimizer, report_every=500):

    for ep in range(1, num_epoch+1):

        # Train

        net.train()

        accum_loss = accum_correct = accum_total = 0

        for i, batch_data in enumerate(train_data, start=1):

            optimizer.zero_grad()



            sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor, label_list = (ele.to(device) for ele in batch_data)

            dist = net(sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor)



            loss = criterion(dist, label_list.float())

            loss.backward()



            accum_loss += loss.item()

            accum_correct += ((dist>0.5).long() == label_list).sum().item()

            accum_total += label_list.numel()



            optimizer.step()



            if i % report_every == 0:

                avg_loss = accum_loss / report_every

                avg_acc = accum_correct / accum_total

                print('Epoch: {:2d} | Training Batch: {:4d} | Avg loss : {:.4f} | Avg acc : {:.4f} '.format

                      (ep, i, avg_loss, avg_acc))

                accum_loss = accum_correct = accum_total = 0



        # Valid

        net.eval()

        accum_loss = accum_correct = accum_total = 0

        for i, batch_data in enumerate(valid_data, start=1):

            sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor, label_list = (ele.to(device) for ele in batch_data)

            dist = net(sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor)



            loss = criterion(dist, label_list.float())



            accum_loss += loss.item()

            accum_correct += ((dist>0.5).long() == label_list).sum().item()

            accum_total += label_list.numel()



        avg_loss = accum_loss / len(valid_data)

        avg_acc = accum_correct / accum_total

        print('Valid Epoch: {:2d} | Avg loss : {:.4f} | Avg acc : {:.4f} '.format

              (ep, avg_loss, avg_acc))
def test_model(net, output_name):

    results = []



    net.eval()

    for i, batch_data in enumerate(test_data, start=40001):

        sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor = (ele.to(device) for ele in batch_data[:-1])

        dist = net(sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor)



        results.extend([1 if b else 0 for b in (dist>0.5).tolist()])



    col = ["id", "label"]

    df = pd.DataFrame({"id":range(40001,50001),

                       "label":results})

    df.to_csv("/kaggle/working/{}".format(output_name), mode="w", index=False)
vocab_size = len(word2idx)

input_size = 128

hidden_size = 128

dropout_p = 0.2

mode = "manhattan"



num_epoch = 5

lr = 0.001

report_every = 500



net = SiamaseNetwork(vocab_size, input_size, hidden_size, dropout_p, mode=mode).to(device)



criterion = nn.MSELoss().to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer)
test_model(net, "siamese_manhattan_final_dropout02_min4_vocab.csv")
vocab_size = len(word2idx)

input_size = 128

hidden_size = 128

dropout_p = 0.2

mode = "logistic"



num_epoch = 2

lr = 0.001

report_every = 500



net = SiamaseNetwork(vocab_size, input_size, hidden_size, dropout_p, mode=mode).to(device)



criterion = nn.BCELoss().to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer)
test_model(net, "siamese_logistic_final_dropout02_full_vocab.csv")
vocab_size = len(word2idx)

input_size = 128

hidden_size = 128

dropout_p = 0.0

mode = "logistic"



num_epoch = 2

lr = 0.001

report_every = 500



net = SiamaseNetwork(vocab_size, input_size, hidden_size, dropout_p, mode=mode).to(device)



criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer)
test_model(net, "siamese_sim_regression_final_min4cut_vocab.csv")
vocab_size = len(word2idx)

input_size = 128

hidden_size = 128

dropout_p = 0.0

mode = "dot"



num_epoch = 3

lr = 0.001

report_every = 500



net = SiamaseNetwork(vocab_size, input_size, hidden_size, dropout_p, mode=mode).to(device)



criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer)
test_model(net, "siamese_dot_final.csv")
vocab_size = len(word2idx)

input_size = 128

hidden_size = 128

dropout_p = 0.0

mode = "attn"



num_epoch = 10

lr = 0.001

report_every = 500



net = SiamaseNetwork(vocab_size, input_size, hidden_size, dropout_p, mode=mode).to(device)



criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer)
test_model(net, "siamese_attn_final_min4cut_vocab.csv")
class BiMPM(nn.Module):

    """

    BiMPM 모델 구현

    """

    def __init__(self, vocab_size, input_size, rnn_size, num_perspective, dropout_p=0.0,

                 pretrained_word_embedding=None, is_freeze=False):

        super(BiMPM, self).__init__()

        

        self.rnn_size = rnn_size

        

        self.embed = nn.Embedding(vocab_size, input_size, padding_idx=0)

        self.dropout = nn.Dropout(p=dropout_p)

        self.ctx_gru = nn.GRU(input_size, rnn_size, bidirectional=True, batch_first=True)

        self.agg_gru = nn.GRU(num_perspective*8, rnn_size, bidirectional=True, batch_first=True)

        

        self.fc_pred1 = nn.Linear(rnn_size*4, rnn_size*2)

        self.fc_pred2 = nn.Linear(rnn_size*2, 1)

        

        for i in range(1, 9):

            setattr(self, "W_{}".format(i),

                    nn.Parameter(torch.randn((num_perspective, rnn_size))))

            

        if pretrained_word_embedding is not None:

            self.embed.from_pretrained(torch.tensor(pretrained_word_embedding))

            self.embed.weight.requires_grad = not is_freeze

 

    def forward(self, sen1_tensor, sen1_len_tensor, sen2_tensor, sen2_len_tensor):

        sen1_outputs = self._context_representation(sen1_tensor, sen1_len_tensor)

        sen2_outputs = self._context_representation(sen2_tensor, sen2_len_tensor)

                

        pq_cat, qp_cat = self._matching(sen1_outputs, sen2_outputs)

        

        pq_hidden = self._aggrigation(pq_cat, sen1_len_tensor)

        qp_hidden = self._aggrigation(qp_cat, sen2_len_tensor)

        

        final_cat = torch.cat((pq_hidden, qp_hidden), dim=-1)

        prob = torch.tanh(self.fc_pred1(final_cat))

        prob = torch.sigmoid(self.fc_pred2(prob)).squeeze()

        return prob

        

    def _context_representation(self, sen_tensor, len_tensor):

        sen_tensor, len_tensor, sorted_idx = self._sort_batch(sen_tensor, len_tensor)

        

        embedded = self.embed(sen_tensor)

        packed = pack_padded_sequence(embedded, len_tensor, batch_first=True)

        outputs, _ = self.ctx_gru(packed)

        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        

        outputs, len_tensor = self._unsort_batch(outputs, len_tensor, sorted_idx)

        outputs = self.dropout(outputs)

        return outputs

    

    def _matching(self, sen1_outputs, sen2_outputs):

        sen1_forward, sen1_backward = sen1_outputs[:, :, :self.rnn_size], sen1_outputs[:, :, self.rnn_size:]

        sen2_forward, sen2_backward = sen2_outputs[:, :, :self.rnn_size], sen2_outputs[:, :, self.rnn_size:]

                

        m_full_pq_fw, m_full_qp_fw = self._full_matching(sen1_forward, sen2_forward, self.W_1)

        m_full_pq_fw, m_full_qp_fw = self._full_matching(sen1_backward, sen2_backward, self.W_2)

        

        m_maxpool_pq_fw, m_maxpool_qp_fw = self._maxpooling_matching(sen1_forward, sen2_forward, self.W_3)

        m_maxpool_pq_bw, m_maxpool_qp_bw = self._maxpooling_matching(sen1_backward, sen2_backward, self.W_4)

        

        m_attn_pq_fw, m_attn_qp_fw = self._attentive_matching(sen1_forward, sen2_forward, self.W_5)

        m_attn_pq_bw, m_attn_qp_bw = self._attentive_matching(sen1_backward, sen2_backward, self.W_6)

        

        m_max_attn_pq_fw, m_max_attn_qp_fw = self._max_attentive_matching(sen1_forward, sen2_forward, self.W_5)

        m_max_attn_pq_bw, m_max_attn_qp_bw = self._max_attentive_matching(sen1_backward, sen2_backward, self.W_6)

        

        pq_list = [m_full_pq_fw, m_full_pq_fw,

                   m_maxpool_pq_fw, m_maxpool_pq_bw,

                   m_attn_pq_fw, m_attn_pq_bw,

                   m_max_attn_pq_fw, m_max_attn_pq_bw]

        

        qp_list = [m_full_qp_fw, m_full_qp_fw,

                   m_maxpool_qp_fw, m_maxpool_qp_bw,

                   m_attn_qp_fw, m_attn_qp_bw,

                   m_max_attn_qp_fw, m_max_attn_qp_bw]

        

        cat_pq = self.dropout(torch.cat(pq_list, dim=-1))

        cat_qp = self.dropout(torch.cat(qp_list, dim=-1))

        

        return cat_pq, cat_qp

    

    def _aggrigation(self, sen_tensor, len_tensor):

        sen_tensor, len_tensor, sorted_idx = self._sort_batch(sen_tensor, len_tensor)

        

        packed = pack_padded_sequence(sen_tensor, len_tensor, batch_first=True)

        _, hidden = self.agg_gru(packed)

              

        hidden = self.dropout(torch.cat((hidden[0], hidden[1]), dim=-1))

        hidden, len_tensor = self._unsort_batch(hidden, len_tensor, sorted_idx)

        

        return hidden

    

    def _full_matching(self, v1, v2, w_k):

        v1_ = w_k * v1.unsqueeze(dim=2)

        v2_ = w_k * v2.unsqueeze(dim=2)



        v1_final_ = v1_[:, -1].unsqueeze(dim=1)

        v2_final_ = v2_[:, -1].unsqueeze(dim=1)

        

        m_full_pq = self.dropout(torch.cosine_similarity(v1_, v2_final_, dim=-1))

        m_full_qp = self.dropout(torch.cosine_similarity(v2_, v1_final_, dim=-1))

        return m_full_pq, m_full_qp

        

    def _maxpooling_matching(self, v1, v2, w_k):

        v1_ = w_k * v1.unsqueeze(dim=2)

        v2_ = w_k * v2.unsqueeze(dim=2)

        

        cos_mat = self._cosine_matrix(v1_, v2_)

        m_maxpool_pq = self.dropout(cos_mat.max(dim=2)[0])

        m_maxpool_qp = self.dropout(cos_mat.max(dim=1)[0])

        return m_maxpool_pq, m_maxpool_qp

        

    def _attentive_matching(self, v1, v2, w_k):

        v1_ = w_k * v1.unsqueeze(dim=2)

        v2_ = w_k * v2.unsqueeze(dim=2)

        

        align = self._cosine_matrix(v1, v2).unsqueeze(-1)

        

        weighted_sum = (align * v2.unsqueeze(dim=1)).sum(dim=2)

        sum_cosine = align.sum(dim=2) + 1e-8

        weighted_sum.size(), sum_cosine.size()

        h_mean_pq = weighted_sum / sum_cosine

        h_mean_pq_ = w_k * h_mean_pq.unsqueeze(dim=2)



        weighted_sum = (align.transpose(1, 2) * v1.unsqueeze(dim=1)).sum(dim=2)

        sum_cosine = align.sum(dim=1) + 1e-8

        h_mean_qp = weighted_sum / sum_cosine

        h_mean_qp_ = w_k * h_mean_qp.unsqueeze(dim=2)



        m_attn_pq = self.dropout(torch.cosine_similarity(v1_, h_mean_pq_, dim=-1))

        m_attn_qp = self.dropout(torch.cosine_similarity(v2_, h_mean_qp_, dim=-1))

        return m_attn_pq, m_attn_qp

        

    def _max_attentive_matching(self, v1, v2, w_k):

        batch_size, sen1_len, hidden_size = v1.size()

        batch_size, sen2_len, hidden_size = v2.size()

        

        v1_ = w_k * v1.unsqueeze(dim=2)

        v2_ = w_k * v2.unsqueeze(dim=2)

        

        align = self._cosine_matrix(v1, v2)

        

        _, max_idx = align.max(dim=2)

        v2 = v2.view(-1, hidden_size)

        max_idx = max_idx.view(-1)

        max_v2 = v2[max_idx]

        h_max_pq = max_v2.view(batch_size, sen1_len, hidden_size)

        h_max_pq_ = w_k * h_max_pq.unsqueeze(dim=2)

        

        _, max_idx = align.max(dim=1)

        v1 = v1.view(-1, hidden_size)

        max_idx = max_idx.view(-1)

        max_v1 = v1[max_idx]

        h_max_qp = max_v1.view(batch_size, sen2_len, hidden_size)

        h_max_qp_ = w_k * h_max_qp.unsqueeze(dim=2)



        m_max_attn_pq = self.dropout(torch.cosine_similarity(v1_, h_max_pq_, dim=-1))

        m_max_attn_qp = self.dropout(torch.cosine_similarity(v2_, h_max_qp_, dim=-1))

        return m_max_attn_pq, m_max_attn_qp

        

    def _sort_batch(self, data_tensor, len_tensor):

        len_tensor, sorted_idx = len_tensor.sort(dim=0, descending=True)

        data_tensor = data_tensor[sorted_idx]

        return (data_tensor, len_tensor, sorted_idx)



    def _unsort_batch(self, data_tensor, len_tensor, sorted_idx):

        _, org_idx = sorted_idx.sort(dim=0)

        data_tensor = data_tensor[org_idx]

        len_tensor = len_tensor[org_idx]

        return (data_tensor, len_tensor)

    

    def _cosine_matrix(self, x1, x2):

        x1 = x1.unsqueeze(2)

        x2 = x2.unsqueeze(1)

        cos_matrix = torch.cosine_similarity(x1, x2, dim=-1)

        return cos_matrix
vocab_size = len(word2idx)

input_size = 128

rnn_size = 128

num_perspective = 20

dropout_p = 0.1



num_epoch = 3

lr = 0.001

report_every = 500



net = BiMPM(vocab_size, input_size, rnn_size, num_perspective, dropout_p).to(device)



criterion = nn.BCELoss().to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)



train_model(net, num_epoch, criterion, optimizer, report_every=report_every)
test_model(net, "BiMPM_dropout01_min4cut_vocab_final.csv")