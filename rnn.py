from collections import OrderedDict

import torch.nn
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, **kwargs):
        super(LSTM, self).__init__()
        self.fc_hidden = kwargs["fc_hidden"]
        self.activation_fn = kwargs["activation_fn"]

        assert len(self.fc_hidden) > 1

        layers = OrderedDict()
        if 'embeddings' in kwargs:
            layers[f"embeddings"] = kwargs['embeddings']

        layers[f"lstm"] = LSTMWrapper(**kwargs)

        for i in range(1, len(self.fc_hidden)):
            n_in, n_out = self.fc_hidden[i - 1], self.fc_hidden[i]
            layers[f"fc_{i}"] = nn.Linear(in_features=n_in, out_features=n_out, bias=True)

            if i != len(self.fc_hidden) - 1:
                layers[f"a_{i}"] = self.activation_fn()

        self.seq = nn.Sequential(layers)
        self.to(device=kwargs['device'])

    def forward(self, x):
        return self.seq(x)

    def reset_params(self):
        for child in self.seq.children():
            if isinstance(child, nn.Linear):
                child.reset_parameters()
            elif isinstance(child, LSTMWrapper):
                child.reset_params()


class LSTMWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(LSTMWrapper, self).__init__()
        d_in, d_out = kwargs['rnn_dims']
        self.num_layers = kwargs["rnn_layers"]
        self.bidirectional = kwargs["bidirectional"]

        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_out, num_layers=self.num_layers,
                            bidirectional=self.bidirectional, batch_first=False)

    def forward(self, x):
        x_sequence_first = torch.transpose(x, dim0=0, dim1=1)  # transposing from BxTxD to TxBxD
        out, _ = self.lstm(x_sequence_first)
        return out[-1, :, :]

    def reset_params(self):
        self.lstm.reset_parameters()


def train(model, train_loader, valid_loader, trainval_loader, test_loader, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"])
    criterion = nn.BCEWithLogitsLoss()

    best_loss = best_epoch = None
    for epoch in range(1, kwargs["epochs"] + 1):
        _train_epoch(model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        if kwargs["debug_print"] and epoch % 5 == 0:
            print(f"[{epoch}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

        if best_loss is None or best_loss > valid_loss + kwargs["es_epsilon"]:
            best_epoch = epoch
            best_loss = valid_loss

        if kwargs["es_patience"] != -1 and (epoch - best_epoch) >= kwargs["es_patience"]:
            if kwargs["debug_print"]:
                print(f"EARLY STOPPING. epoch={epoch}")
            break

    train_loss = _evaluate(model, train_loader, criterion, **kwargs)
    valid_loss = train_loss + 42  # 42 chosen randomly, valid_loss just has to be bigger than train_loss
    e = 0

    while valid_loss > train_loss and e < kwargs["es_maxiter"]:
        _train_epoch(model, trainval_loader, optimizer, criterion, e,  **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        e += 1

    test_loss = _evaluate(model, test_loader, criterion, **kwargs)
    if kwargs["debug_print"]:
        print(f"[FINISHED] TEST LOSS = {test_loss}")


def _train_epoch(model, data, optimizer, criterion, current_epoch, **kwargs):
    model.train()
    losses, batch_sizes = [], []
    for batch_num, (x, y, _) in enumerate(data):
        x, y = x.to(kwargs["device"]), y[:, kwargs['index']].to(kwargs["device"])

        model.zero_grad()
        logits = model(x).reshape(y.shape)
        loss = criterion(logits, y)
        loss.backward()
        if "clip" in kwargs.keys():
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["clip"])
        optimizer.step()

        losses.append(loss)
        batch_sizes.append(len(x))

    if kwargs["debug_print"]:
        avg_loss = sum([loss * size for loss, size in zip(losses, batch_sizes)]) / sum(batch_sizes)
        print(f"[{current_epoch}/{kwargs['epochs']}] TRAIN LOSS = {avg_loss:3.5f}")


def _evaluate(model, data, criterion, **kwargs):
    model.eval()
    with torch.no_grad():
        losses, batch_sizes = [], []
        for batch_num, (x, y, _) in enumerate(data):
            x, y = x.to(kwargs["device"]), y[:, kwargs['index']].to(kwargs["device"])

            logits = model(x).reshape(y.shape)
            loss = criterion(logits, y)

            losses.append(loss)
            batch_sizes.append(len(x))

    loss = sum([loss * size for loss, size in zip(losses, batch_sizes)]) / sum(batch_sizes)
    return loss


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
