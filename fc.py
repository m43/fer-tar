import torch
import torch.nn as nn


class DeepFC(nn.Module):

    def __init__(self, in_dim, device):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, 400), nn.ReLU(),
            nn.Linear(400, 100), nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()

        if device is not None:
            self.dvc = device
            self.to(device=torch.device(device))

    def forward(self, X):
        return self.model(X)


def train(model, train_x, train_y, device, epochs, lr, batch_size, wd):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for i in range(epochs):
        perm = torch.randperm(train_x.size(0), device=device)
        train_x, train_y = train_x[perm], train_y[perm]

        start, end = 0, min(batch_size, train_x.size(0))
        batch_i = 0

        while True:
            opt.zero_grad()
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]
            loss = model.loss(model.forward(x_batch), y_batch)
            loss.backward()
            opt.step()

            if batch_i % 10 == 0:
                print(f"epoch: {i}, batch_i: {batch_i}, loss: {loss}")

            batch_i += 1
            start = end
            end = min(start+batch_size, train_x.size(0))
            if start == end:
                break
