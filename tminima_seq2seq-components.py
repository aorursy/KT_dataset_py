import torch



from IPython.display import Image

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
Image("../input/images/embedding_layer.jpg")
# Size of the dictionary

input_vocab_dim = 5



# Size of embedding

embedding_dim = 20



# Defining the embedding layer

embedding_layer = torch.nn.Embedding(input_vocab_dim, embedding_dim)

embedding_layer
# A single sentence having 5 tokens

inp = torch.tensor([1, 2, 3, 4, 0])

print("input shape\t:", inp.shape)



# getting the vector for each token using the embedding layer

out = embedding_layer(inp)

print("output shape\t:", out.shape)

print(out)
# Batch of 4 sentences having 5 tokens each.

# Batch size is 4. Max sentence len is 5

inp = torch.tensor([1, 2, 3, 4, 0])

inp = inp.unsqueeze(1).repeat(1, 4)

print("input shape\t:", inp.shape)



out = embedding_layer(inp)

print("output shape\t:", out.shape)

print(out[:, 0, :])
dropout_probab = 0.5



dropout = torch.nn.Dropout(dropout_probab)

dropout
# A random tensor with 8 values

inp = torch.randn(8)

print("input shape\t:", inp.shape)

print("input tensor\t:", inp)



# getting the vector for each token using the embedding layer

out = dropout(inp)

print("\noutput shape\t:", out.shape)

print("output tensor\t:", out)
# a random tensor with 10 values

inp = torch.randn(5, 4, 20)

print("input shape\t:", inp.shape)



# getting the vector for each token using the embedding layer

out = dropout(inp)

print("output shape\t:", out.shape)



print("\nTotal cells in input\t:", 5*4*20)

print("Zero values in input\t:", torch.sum(inp==0.00))

print("Zero values in output\t:", torch.sum(out==0.00))
Image("../input/images/lstm.jpg")
Image("../input/images/lstm2.jpg")
input_dim = 5

hidden_dim = 15



lstm = torch.nn.LSTM(input_dim, hidden_dim)

lstm
# Random input with shape - (1, 1, 5)

inp = torch.randn(1, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# Random input with shape - (4, 1, 5)

inp = torch.randn(4, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# Random input with shape - (4, 6, 5)

inp = torch.randn(4, 6, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
Image("../input/images/stacked_lstm1.png")
Image("../input/images/stacked_lstm2.png")
input_dim = 5

hidden_dim = 15

num_layers = 2



lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=0.5)

lstm
# Random input with shape - (1, 1, 5)

inp = torch.randn(1, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# Random input with shape - (4, 1, 5)

inp = torch.randn(4, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# Random input with shape - (4, 6, 5)

inp = torch.randn(4, 6, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
Image("../input/images/bilstm.jpg")
input_dim = 5

hidden_dim = 15



lstm = torch.nn.LSTM(input_dim, hidden_dim, bidirectional=True)

lstm
# Random input with shape - (1, 1, 5)

inp = torch.randn(1, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# Random input with shape - (4, 1, 5)

inp = torch.randn(4, 1, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
# random input is of shape - (4, 6, 5)

inp = torch.randn(4, 6, 5)

print("input shape\t:", inp.shape)



out, (hid, cell) = lstm(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)

print("cell shape\t:", cell.shape)
input_dim = 5

hidden_dim = 15



gru = torch.nn.GRU(input_dim, hidden_dim)

gru
# Random input with shape - (4, 6, 5)

inp = torch.randn(4, 6, 5)

print("input shape\t:", inp.shape)



out, hid = gru(inp)

print("output shape\t:", out.shape)

print("hidden shape\t:", hid.shape)
input_dim = 5

output_dim = 15



linear_layer = torch.nn.Linear(input_dim, output_dim)

linear_layer
# Random input with shape - (5)

inp = torch.randn(5)

print("input shape\t:", inp.shape)



out = linear_layer(inp)

print("output shape\t:", out.shape)
# Random input with shape - (1, 1, 5)

inp = torch.randn(1, 1, 5)

print("input shape\t:", inp.shape)



out = linear_layer(inp)

print("output shape\t:", out.shape)
# Random input with shape - (4, 1, 5)

inp = torch.randn(4, 1, 5)

print("input shape\t:", inp.shape)



out = linear_layer(inp)

print("output shape\t:", out.shape)
# Random input with shape - (4, 6, 5)

inp = torch.randn(4, 6, 5)

print("input shape\t:", inp.shape)



out = linear_layer(inp)

print("output shape\t:", out.shape)