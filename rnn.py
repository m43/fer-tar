from collections import OrderedDict

import torch.nn
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, **kwargs):
        super(LSTM, self).__init__()
        self.num_layers = kwargs["rnn_layers"]
        self.rnn_hidden = kwargs["rnn_hidden"]
        self.bidirectional = kwargs["bidirectional"]
        self.fc_hidden = kwargs["fc_hidden"]
        self.activation_fn = kwargs["activation_fn"]
        softmax_at_end = kwargs["softmax_at_end"]

        layers = OrderedDict()
        for i in range(1, len(self.rnn_hidden)):
            n_in, n_out = self.rnn_hidden[i - 1], self.rnn_hidden[i]
            layers[f"lstm_{i}"] = nn.LSTM(input_size=n_in, hidden_size=n_out, num_layers=self.num_layers,
                                          bidirectional=self.bidirectional, batch_first=True)

        for i in range(1, len(self.fc_hidden)):
            n_in, n_out = self.fc_hidden[i - 1], self.fc_hidden[i]
            layers[f"fc_{i}"] = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

            if i != len(self.fc_hidden) - 1:
                layers[f"a_{i}"] = self.activation_fn()
            elif softmax_at_end:
                layers[f"a_{i}"] = nn.Softmax()

        self.seq = nn.Sequential(layers)

    def forward(self, input):
        return self.seq(input)


if __name__ == "__main__":
    args = {
        "rnn_layers": 2,
        "rnn_hidden": [300, 150],
        "bidirectional": False,
        "fc_hidden": [150, 150, 1],
        "activation_fn": torch.nn.ReLU,
        "softmax_at_end": False
    }
    lstm = LSTM(**args)
