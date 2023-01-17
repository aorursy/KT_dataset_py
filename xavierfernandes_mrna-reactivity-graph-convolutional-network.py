!pip install torch-geometric
!pip install networkx
!pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
!pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
import torch_geometric.nn as gnn
train_file = open("/kaggle/input/stanford-covid-vaccine/train.json", "r")
train_data_raw = train_file.read()
train_data = list(map(lambda l: json.loads(l), list(filter(lambda l: len(l) > 0, train_data_raw.split("\n")))))
train_file.close()

test_file = open("/kaggle/input/stanford-covid-vaccine/test.json", "r")
test_data_raw = test_file.read()
test_data = list(map(lambda l: json.loads(l), list(filter(lambda l: len(l) > 0, test_data_raw.split("\n")))))
test_file.close()
train_data[0].keys()
train_data[0]["sequence"]
train_data[0]["structure"]
train_data[0]["predicted_loop_type"]
train_data[0]["id"], len(train_data[0]["sequence"]), len(train_data[0]["structure"]), len(train_data[0]["predicted_loop_type"])
input_sequence = list(zip(train_data[0]["sequence"], train_data[0]["structure"], train_data[0]["predicted_loop_type"]))
len(input_sequence)
input_sequence[0], input_sequence[-1]
train_data[0]["reactivity"][0], train_data[0]["deg_Mg_pH10"][0], train_data[0]["deg_pH10"][0], train_data[0]["deg_Mg_50C"][0], train_data[0]["deg_50C"][0] 
output_sequence = list(zip(train_data[0]["reactivity"], train_data[0]["deg_Mg_pH10"], train_data[0]["deg_pH10"], train_data[0]["deg_Mg_50C"], train_data[0]["deg_50C"]))
evaluation_length = train_data[0]["seq_scored"]
output_sequence[0], output_sequence[-1]
def build_categorical_encoder(input_attribute_index_or_key, categories, total_dimensions, start_index, end_index):
    if not (end_index - start_index + 1 == len(categories)):
        raise Exception("Mismatch between number of categories and dimensions assigned.")
    def encoder(data_row, data_vector):
        if len(data_vector) != total_dimensions:
            raise Exception(f"Data vector is of size {len(data_vector)}, but should be of size {total_dimensions}.")
        
        category_index = categories.index(data_row[input_attribute_index_or_key])
        encoding_index = start_index + category_index
        data_vector[encoding_index] = 1
    
    return encoder

def build_continuous_encoder(input_attribute_index_or_key, total_dimensions, attribute_index):
    def encoder(data_row, data_vector):
        if len(data_vector) != total_dimensions:
            raise Exception(f"Data vector is of size {len(data_vector)}, but should be of size {total_dimensions}.")
        data_vector[attribute_index] = float(data_row[input_attribute_index_or_key])
    
    return encoder

def encode_data(rows, encoders, total_dimensions):
    encoded_rows = []
    for row in rows:
        encoded_row = [0] * total_dimensions
        for encoder in encoders:
            encoder(row, encoded_row)
        encoded_rows.append(encoded_row)
    
    return encoded_rows
total_dim_input = 11
total_dim_output = 5
sequence_classes = ["A", "G", "U", "C"]
structure_classes = ["(", ".", ")"]
predicted_loop_type_classes = ["S", "M", "I", "B", "H", "E", "X"]

sequence_encoder = build_categorical_encoder(0, sequence_classes, total_dim_input, 0, 3)
predicted_loop_encoder = build_categorical_encoder(2, predicted_loop_type_classes, total_dim_input, 4, 10)


data_encoders = [
    sequence_encoder,
    predicted_loop_encoder
]
encoded_input_sequence = encode_data(input_sequence, data_encoders, total_dim_input)
input_sequence[5]
encoded_input_sequence[5]
def build_rna_graph_structure(input_sequence, target=None, target_errors=None):
    encoded_input_sequence = torch.tensor(
        encode_data(input_sequence, data_encoders, total_dim_input),
        dtype=torch.float32)
    edges = []
    node_features = []
    G = nx.Graph()
    stack = []
    prev_id = None
    for (node_id, (base, structure_class, predicted_loop_type)) in enumerate(input_sequence):
        G.add_node(node_id, base=base, predicted_loop_type=predicted_loop_type)
        if structure_class == "(":
            stack.append(node_id)
        elif structure_class == ")":
            neighbour_id = stack.pop()
            G.add_edge(node_id, neighbour_id)
            edges.append([node_id, neighbour_id])
            edges.append([neighbour_id, node_id])
        
        if prev_id is not None:
            G.add_edge(node_id, prev_id)
            edges.append([node_id, prev_id])
            edges.append([prev_id, node_id])

        prev_id = node_id
    
    edge_index = torch.transpose(torch.tensor(edges, dtype=torch.long), 0, 1)
    
    if target is not None and target_errors is not None:
        weights = []
        for error_row in target_errors:
            weights_row = []
            for error in error_row:
                weights_row.append(1/(error + 1))
            weights.append(weights_row)
        target = torch.tensor(target, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        graph_data = Data(x=encoded_input_sequence, edge_index=edge_index, y=target, weights=weights)
    elif target is not None:
        target = torch.tensor(target, dtype=torch.float32)
        graph_data = Data(x=encoded_input_sequence, edge_index=edge_index, y=target)
    else:
        graph_data = Data(x=encoded_input_sequence, edge_index=edge_index)
    
    return G, graph_data
G, graph_data = build_rna_graph_structure(input_sequence)
nx.draw(G, node_size=20)
G.nodes[0]
filtered_train_data = list(filter(lambda d: d["SN_filter"] == 1, train_data))
sn = list(map(lambda r: r["SN_filter"], train_data))

graphs_train = []
dataset = []
for row in filtered_train_data:
    input_sequence = list(zip(row["sequence"], row["structure"], row["predicted_loop_type"]))
    target_sequence = list(zip(row["reactivity"], row["deg_Mg_pH10"], row["deg_pH10"], row["deg_Mg_50C"], row["deg_50C"]))
    target_sequence_errors = list(zip(row["reactivity_error"], row["deg_error_Mg_pH10"], row["deg_error_pH10"], row["deg_error_Mg_50C"], row["deg_error_50C"]))
    # target_sequence = list(zip(row["reactivity"], row["deg_Mg_pH10"], row["deg_Mg_50C"]))
    # target_sequence_errors = list(zip(row["reactivity_error"], row["deg_error_Mg_pH10"], row["deg_error_Mg_50C"]))
    G, graph_data = build_rna_graph_structure(input_sequence, target=target_sequence, target_errors=target_sequence_errors)
    dataset.append(graph_data)
    graphs_train.append(G)


graphs_test = []
testing_set = []
testing_sequence_id_to_dataset_map = dict()
for row in test_data:
    input_sequence = list(zip(row["sequence"], row["structure"], row["predicted_loop_type"]))
    G, graph_data = build_rna_graph_structure(input_sequence)
    testing_set.append(graph_data)
    graphs_test.append(G)
    testing_sequence_id_to_dataset_map[row["id"]] = graph_data
random.shuffle(dataset)

training = dataset[:int(len(dataset) * 0.8)]
evaluation = dataset[int(len(dataset) * 0.8):]
len(training), len(evaluation), len(testing_set)
batch_size_train = 31
batch_size_test = 318
batch_size_submission = 158
number_of_epochs = 100
train_loader = DataLoader(training, batch_size=batch_size_train)
test_loader = DataLoader(evaluation, batch_size=batch_size_test)
submission_loader = DataLoader(testing_set, batch_size=batch_size_submission)
test_batches = list(test_loader)
test_batch = test_batches[0]
class RNAGenConv(nn.Module):
    
    def __init__(self, node_features_dim=None, node_embedding_dim=None, node_output_features_dim=None):
        super(RNAGenConv, self).__init__()
        
        self.conv_layer = gnn.GENConv(in_channels=node_features_dim, out_channels=node_embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(node_embedding_dim, node_embedding_dim + 5)
        self.linear2 = nn.Linear(node_embedding_dim + 5, node_output_features_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv_layer(x, edge_index)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        
        return x


class DeepRNAGenConv(nn.Module):
    
    def __init__(self, node_features_dim=None, node_embedding_dim=None, num_layers=None, node_output_features_dim=None, convolution_dropout=0.1, dense_dropout=0.0):
        super(DeepRNAGenConv, self).__init__()
        
        self.node_encoder = nn.Linear(node_features_dim, node_embedding_dim)
        
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            convolution = gnn.GENConv(in_channels=node_embedding_dim, out_channels=node_embedding_dim)
            norm = nn.LayerNorm(node_embedding_dim)
            activation = nn.ReLU()
            layer = gnn.DeepGCNLayer(conv=convolution, norm=norm, act=activation, dropout=convolution_dropout)
            self.gcn_layers.append(layer)

        self.dropout = nn.Dropout(p=dense_dropout)
        self.decoder = nn.Linear(node_embedding_dim, node_output_features_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.node_encoder(x)
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        
        x = self.dropout(x)
        x = self.decoder(x)
        
        return x
def weighted_mse_loss(output, target, weights=None):
    if weights is not None:
        weighted_sum_errors = weights * ((output - target)**2)
        mean_weighted_sum_errors = weighted_sum_errors.mean()
        return mean_weighted_sum_errors
    else:
        return F.mse_loss(output, target)

def unweighted_mse_loss(output, target, **kwargs):
    return F.mse_loss(output, target)
def batch_loss(batch_input, batch_output, batch_size, loss_function=None, eval_length=None):
    total_loss = torch.tensor(0.0).to("cuda")
    for i in range(batch_size):
        graph_output = batch_output[batch_input.batch == i]
        target = batch_input.y[(i * eval_length):((i + 1) * eval_length), :].to("cuda")
        weights = batch_input.weights[(i * eval_length):((i + 1) * eval_length), :].to("cuda")
        evaluation_nodes = target.size(0)
        graph_output_evaluation = graph_output[:evaluation_nodes, :]
        total_loss += loss_function(graph_output_evaluation, target, weights=weights)
    
    return total_loss
rna_gcnn = DeepRNAGenConv(
    node_features_dim=total_dim_input,
    node_embedding_dim=90,
    num_layers=10,
    node_output_features_dim=5,
    convolution_dropout=0.2,
    dense_dropout=0.0).cuda()

optimizer = optim.Adam(rna_gcnn.parameters(), lr=0.01)
train_counter = list(range(len(train_loader) * number_of_epochs))
train_losses = []
test_losses = []
test_counter = [i * len(train_loader) for i in range(number_of_epochs)]
for n in range(number_of_epochs):
    for batch_input in train_loader:
        rna_gcnn.zero_grad()
        batch_output = rna_gcnn(batch_input.to("cuda"))
        loss = batch_loss(batch_input, batch_output, batch_size_train, loss_function=unweighted_mse_loss, eval_length=evaluation_length) / batch_size_train
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        rna_gcnn.zero_grad()
        train_losses.append(torch.sqrt(loss).item())
    with torch.no_grad():
        batch_output_test = rna_gcnn(test_batch.to("cuda"))
        test_loss = batch_loss(test_batch, batch_output_test, batch_size_test, loss_function=unweighted_mse_loss, eval_length=evaluation_length) / batch_size_test
        test_losses.append(torch.sqrt(test_loss).item())
fig, ax = plt.subplots()
ax.plot(train_counter, train_losses, color="b", label="Train Loss")
ax.scatter(test_counter, test_losses, color="red", label="Test Loss")
leg = ax.legend()
min(train_losses), min(test_losses)
rna_gcnn.eval()
sample_submission_file = open("/kaggle/input/stanford-covid-vaccine/sample_submission.csv", "r")
sample_submission_data = sample_submission_file.read()
sample_submission_file.close()
sample_submission_lines = list(filter(lambda l: len(l) > 0, map(lambda ll: ll.strip(), sample_submission_data.split("\n"))))
sample_submission_lines[0]
sample_submission_lines[1]
testing_sequence_id_to_dataset_map["id_00073f8be"]
output_header = 'id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C'
output_lines = [output_header]
for sequence_id, graph_dataset in testing_sequence_id_to_dataset_map.items():
    output = rna_gcnn(graph_dataset.to("cuda"))
    output = output.to("cpu")
    seq_length = output.size(0)
    for i in range(seq_length):
        seq_id_pos = f"{sequence_id}_{i}"
        seq_id_pos_entry = list(map(lambda num: str(num), output[i,:].tolist()))
        submission_line_entries = [seq_id_pos] + seq_id_pos_entry
        submission_line = ",".join(submission_line_entries)
        output_lines.append(submission_line)
    
submission_data = "\n".join(output_lines)
output_lines[1]
output_lines[107]
submission_file = open("submission.csv", "w")
submission_file.write(submission_data)
submission_file.close()
