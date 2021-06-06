import copy
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


def train(model, train_loader, valid_loader, trainval_loader, test_loader, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"])
    criterion = nn.BCEWithLogitsLoss()

    if kwargs["debug_print"]:
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        print(f"[{-1}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

    best_model_dict = best_loss = best_epoch = None
    for epoch in range(1, kwargs["epochs"] + 1):
        _train_epoch(model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        if kwargs["debug_print"]:
            print(f"[{epoch}/{kwargs['epochs']}] VALID LOSS = {valid_loss}")

        if best_loss is None or best_loss > valid_loss + kwargs["es_epsilon"]:
            if kwargs["debug_print"]:
                print(f"New best epoch is {epoch} with valid loss {valid_loss}")
            best_epoch = epoch
            best_loss = valid_loss
            best_model_dict = copy.deepcopy(model.state_dict())

        if kwargs["es_patience"] != -1 and epoch - best_epoch >= kwargs["es_patience"]:
            if kwargs["debug_print"]:
                print(f"EARLY STOPPING at epoch {epoch}. Best epoch {best_epoch}. Best valid loss: {best_loss}. Cheers")
            assert best_model_dict is not None
            model.load_state_dict(best_model_dict)
            break

    test_loss = _evaluate(model, test_loader, criterion, **kwargs)
    if kwargs["debug_print"]:
        print(f"[FINISHED] TEST LOSS = {test_loss}")


def train_OLD(model, train_loader, valid_loader, trainval_loader, test_loader, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["wd"])
    criterion = nn.BCEWithLogitsLoss()

    best_loss = best_epoch = None
    for epoch in range(1, kwargs["epochs"] + 1):
        _train_epoch(model, train_loader, optimizer, criterion, current_epoch=epoch, **kwargs)
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
        _train_epoch(model, trainval_loader, optimizer, criterion, **kwargs)
        valid_loss = _evaluate(model, valid_loader, criterion, **kwargs)
        e += 1

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
