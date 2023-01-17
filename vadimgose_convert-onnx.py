import torch
model = torch.load('../input/recursive-training/aerialmodel.pth')

model = model.to('cpu')

dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
torch.onnx.export(model, dummy_input, "aerialmodel.onnx")