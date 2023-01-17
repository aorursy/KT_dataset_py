import torch

import torch.nn as nn
# Defining input size, hidden layer size, output size and batch size respectively

n_in, n_h, n_out, batch_size = 10, 5, 1, 10
# Create dummy input and target tensors (data)

x = torch.randn(batch_size, n_in)

y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
# Create a model

model = nn.Sequential(nn.Linear(n_in, n_h),

                     nn.ReLU(),

                     nn.Linear(n_h, n_out),

                     nn.Sigmoid())
# Construct the loss function

criterion = torch.nn.MSELoss()



# Construct the optimizer (Stochastic Gradient Descent in this case)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Gradient Descent

for epoch in range(50):

    # Forward pass: Compute predicted y by passing x to the model

    y_pred = model(x)



    # Compute and print loss

    loss = criterion(y_pred, y)

    print('epoch: ', epoch,' loss: ', loss.item())



    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()

    

    # perform a backward pass (backpropagation)

    loss.backward()

    

    # Update the parameters

    optimizer.step()