import copy
from collections import OrderedDict

import torch.nn
import torch.nn as nn


class MaRNN(nn.Module):
    def __init__(self, device, rnn_class=nn.LSTM, dropout=0.5, **kwargs):
        super().__init__()
        self.embeddings = kwargs['embeddings'] if 'embeddings' in kwargs else None

        d_in, d_out = kwargs['rnn_dims']
        num_layers = kwargs["rnn_layers"]
        bidirectional = kwargs["bidirectional"]

        self.rnn = rnn_class(input_size=d_in, hidden_size=d_out, num_layers=num_layers, dropout=dropout,
                             bidirectional=bidirectional)
        fc_in = d_out * 2 if bidirectional else d_out
        self.fc1 = nn.Linear(fc_in, kwargs["fc_hid_dim"])
        self.fc2 = nn.Linear(kwargs["fc_hid_dim"], 1)
        self.to(device=device)

    def forward(self, x, lengths=None):
        if self.embeddings is not None:
            x = self.embeddings(x)
        x = torch.transpose(x, dim0=0, dim1=1)
        x, _ = self.rnn(x)
        # Better off using torch.nn.utils.rnn.pack_padded_sequence instead of lengths and vstack
        x = torch.vstack([x[lengths[i] - 1, i] for i in range(x.shape[1])])
        # x = torch.vstack([x[-1, i] for i in range(x.shape[1])])
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


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

    valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
    print(f"[{-1}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

    best_model_dict = best_loss = best_epoch = None
    for epoch in range(kwargs["epochs"]):
        epoch += 1

        _train_epoch(epoch, model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        print(f"[{epoch}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

        if best_loss is None or best_loss > valid_loss + kwargs["es_epsilon"]:
            print(f"New best epoch is {epoch} with valid loss {valid_loss}")
            best_epoch = epoch
            best_loss = valid_loss
            best_model_dict = copy.deepcopy(model.state_dict())

        if kwargs["es_patience"] != -1 and epoch - best_epoch >= kwargs["es_patience"]:
            print(f"EARLY STOPPING at epoch {epoch}. Best epoch {best_epoch}. Best valid loss: {best_loss}. Cheers")
            assert best_model_dict is not None
            model.load_state_dict(best_model_dict)
            break

    test_loss = _evaluate(model, test_loader, criterion, **kwargs)
    print(f"[FINITO] TEST LOSS\n{test_loss}")


def train_OLD(model, train_loader, valid_loader, trainval_loader, test_loader, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"])
    criterion = nn.BCEWithLogitsLoss()

    best_loss = best_epoch = None
    for epoch in range(1, kwargs["epochs"] + 1):
        _train_epoch(epoch, model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        if kwargs["debug_print"]:
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
        _train_epoch(e, model, trainval_loader, optimizer, criterion, e, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        e += 1

    test_loss = _evaluate(model, test_loader, criterion, **kwargs)
    if kwargs["debug_print"]:
        print(f"[FINISHED] TEST LOSS = {test_loss}")


# my_batch = None


def _train_epoch(epoch, model, data, optimizer, criterion, current_epoch, **kwargs):
    # global my_batch
    model.train()
    losses, batch_sizes = [], []
    loss_sum = correct = total = 0
    for batch_num, (x, y, l) in enumerate(data):
        x, y, l = x.to(kwargs["device"]), y[:, kwargs['index']].to(kwargs["device"]), l.to(kwargs["device"])
        # if my_batch is None:
        #     my_batch = (x, y, l)
        # x, y, l = my_batch

        model.zero_grad()
        logits = model(x, l).reshape(y.shape)
        loss = criterion(logits, y)
        loss.backward()
        if "clip" in kwargs.keys():
            torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["clip"])
        optimizer.step()

        losses.append(loss)
        batch_sizes.append(len(x))

        y_predicted = (logits > 0).clone().detach().type(torch.int).reshape(y.shape)
        correct += (y_predicted == y).sum()
        total += len(x)

        loss_sum += loss
    print(
        f"[{epoch}/{kwargs['epochs']}] TRAIN --> losses={' '.join([str(x) for x in losses])}\t last iter logits: {' '.join([str(x) for x in logits])}")
    print(f"[{epoch}/{kwargs['epochs']}] TRAIN --> avg_loss={loss_sum / (batch_num + 1)}\tacc={correct / total}")

    if kwargs["debug_print"]:
        avg_loss = sum([loss * size for loss, size in zip(losses, batch_sizes)]) / sum(batch_sizes)
        print(f"[{current_epoch}/{kwargs['epochs']}] TRAIN LOSS = {avg_loss:3.5f}")


def _evaluate(model, data, criterion, **kwargs):
    model.eval()
    with torch.no_grad():
        losses, batch_sizes = [], []
        loss_sum = correct = total = 0
        for batch_num, (x, y, l) in enumerate(data):
            x, y, l = x.to(kwargs["device"]), y[:, kwargs['index']].to(kwargs["device"]), l.to(kwargs["device"])

            logits = model(x, l).reshape(y.shape)
            loss = criterion(logits, y)

            losses.append(loss)
            batch_sizes.append(len(x))

            y_predicted = (logits > 0).clone().detach().type(torch.int).reshape(y.shape)
            correct += (y_predicted == y).sum()
            total += len(x)

            # print(f"EVAL --> {batch_num} --> Loss: {loss:3.5f}")
            loss_sum += loss
    print(f"EVAL --> losses: {[' '.join([str(x) for x in losses])]}")
    print(f"EVAL --> avg_loss={loss_sum / (batch_num + 1)}\tacc={correct / total}")

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
