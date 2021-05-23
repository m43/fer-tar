from collections import OrderedDict

import torch
import torch.nn as nn


class DeepFC(nn.Module):
    def __init__(self, neurons_per_layer, activation_module=torch.nn.ReLU, softmax_at_end=False):
        super().__init__()
        assert (len(neurons_per_layer) > 1)

        self.neurons_per_layer = neurons_per_layer
        self.activation_fn = activation_module

        layers = OrderedDict()
        for i in range(1, len(neurons_per_layer)):
            n_in, n_out = neurons_per_layer[i - 1], neurons_per_layer[i]
            layers[f"ll_{i}"] = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

            if i != len(self.neurons_per_layer) - 1:
                layers[f"a_{i}"] = self.activation_fn()
            elif softmax_at_end:
                layers[f"a_{i}"] = nn.Softmax()

        self.seq = nn.Sequential(layers)

    def forward(self, x):
        return self.seq(x)
