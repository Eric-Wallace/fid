import torch
from copy import deepcopy
from utils import symmetric_linear_quantization_params, linear_quantize, linear_dequantize

model = torch.load('checkpoint/retriever/single/nq/bert-base-encoder.cp', map_location=torch.device('cpu'))
# model = torch.load('dpr_reader.9.934', map_location=torch.device('cpu'))
print(model['model_dict'][list(model['model_dict'].keys())[0]])
if 'optimizer_dict' in model:
    # del model['optimizer_dict']
    model['optimizer_dict'] = "none"

quantized_state_dict = deepcopy(model['model_dict'])
for key in list(quantized_state_dict.keys()):
    if 'ctx_model' in key:
        del quantized_state_dict[key]
k = 8
for i, key in enumerate(quantized_state_dict):
    weight = quantized_state_dict[key]
    # scale = symmetric_linear_quantization_params(weight, torch.min(weight, dim=1)[0], torch.max(weight, dim=1)[0])
    scale = symmetric_linear_quantization_params(k, torch.min(weight), torch.max(weight))
    n = 2 ** (k - 1) - 1

    weight_integer = linear_quantize(weight, scale, torch.zeros(1), inplace=False)
    weight_integer = torch.clamp(weight_integer, -n-1, n)
    weight_integer = weight_integer.to(torch.int8)

    quantized_state_dict[key] = (weight_integer, scale)

model['model_dict'] = quantized_state_dict
# torch.save(model, 'dpr_reader.9.934.quantized')
torch.save(model, 'checkpoint/retriever/single/nq/bert-base-encoder.cp.quantized')

## Load Model
# model = torch.load('dpr_reader.9.934.quantized')
model = torch.load('checkpoint/retriever/single/nq/bert-base-encoder.cp.quantized')
state_dict = deepcopy(model['model_dict'])
for key in state_dict:
    weight, scale =  state_dict[key]
    weight = linear_dequantize(weight, scale, torch.zeros(1), inplace=False)
    state_dict[key] = weight
model['model_dict'] = state_dict
print(model['model_dict'][list(model['model_dict'].keys())[0]])
