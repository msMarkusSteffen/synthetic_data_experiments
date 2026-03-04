import torch 
import torch.nn as nn
import numpy as np

# NOTE see https://gianluca.ai/table-diffusion/ for reference

def build_noise_layers(shape, steps=1):
    noise_layers = []
    
    for s in range(1,steps+1):
        # beta_s = s/steps
        beta_s = (1.0 - np.cos(np.pi*s/steps))/2.0 
        # NOTE as to mentioned in the upper reference using cos instead of linear noise reduce the forward 
        # process steps by an order of a magnitude (implementing nonlinearity)
        noise_layer = torch.randn(shape) *np.sqrt(beta_s) # NOTE randn for normal distribution
        noise_layers.append(noise_layer)
    return noise_layers 
X= torch.ones(8, requires_grad=True)
steps = 3
print(build_noise_layers(X.shape, steps))
noise_layers = build_noise_layers(X.shape, steps)

for layer in noise_layers:
    X = X+layer

print("noisedX", X)

for layer in noise_layers:
    X = X-layer

print("denoisedX", X)
