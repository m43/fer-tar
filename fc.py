from collections import OrderedDict

import torch
import torch.nn as nn


class DeepFC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dims, act_hook = kwargs['neurons_per_layer'], kwargs['activation_module']
        self.dims = dims

        layers = OrderedDict()
        for i in range(1, len(dims)):
            layers[f"ll_{i}"] = nn.Linear(in_features=dims[i - 1], out_features=dims[i], bias=True)

            if i != len(self.dims) - 1:
                layers[f"a_{i}"] = act_hook()

        self.seq = nn.Sequential(layers)
        self.to(device=kwargs['device'])

    def reset_params(self):
        for child in self.seq.children():
            if isinstance(child, nn.Linear):
                child.reset_parameters()

    def forward(self, x):
        return self.seq(x)


def train(model, train_loader, valid_loader, test_loader, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"])
    criterion = nn.BCEWithLogitsLoss()

    best_loss = best_epoch = None
    for epoch in range(1, kwargs["epochs"] + 1):
        _train_epoch(model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        if kwargs["debug_print"] and epoch % 5 == 0:
            print(f"[{epoch}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

        if best_loss is None or best_loss > valid_loss + kwargs["early_stopping_epsilon"]:
            best_epoch = epoch
            best_loss = valid_loss

        if kwargs["early_stopping_iters"] != -1 and (epoch - best_epoch) >= kwargs["early_stopping_iters"]:
            if kwargs["debug_print"]:
                print(f"EARLY STOPPING. epoch={epoch}")
            break

    model.reset_params()
    for epoch in range(1, best_epoch + 1):
        _train_epoch(model, train_loader, optimizer, criterion, **kwargs)
        _train_epoch(model, valid_loader, optimizer, criterion, **kwargs)

    test_loss = _evaluate(model, test_loader, criterion, **kwargs)
    if kwargs["debug_print"]:
        print(f"[FINISHED] TEST LOSS = {test_loss}")


def _train_epoch(model, data, optimizer, criterion, **kwargs):
    model.train()
    losses, batch_sizes = [], []
    for batch_num, (x, y) in enumerate(data):
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

    if kwargs["debug_print"] and "current_epoch" in kwargs.keys():
        avg_loss = sum([loss * size for loss, size in zip(losses, batch_sizes)]) / sum(batch_sizes)
        print(f"[{kwargs['current_epoch']}/{kwargs['epochs']}] TRAIN LOSS = {avg_loss:3.5f}")


def _evaluate(model, data, criterion, **kwargs):
    model.eval()
    with torch.no_grad():
        losses, batch_sizes = [], []
        for batch_num, (x, y) in enumerate(data):
            x, y = x.to(kwargs["device"]), y[:, kwargs['index']].to(kwargs["device"])

            logits = model(x).reshape(y.shape)
            loss = criterion(logits, y)

            losses.append(loss)
            batch_sizes.append(len(x))

    loss = sum([loss * size for loss, size in zip(losses, batch_sizes)]) / sum(batch_sizes)
    return loss
