import copy

import torch
from torch.nn import BCEWithLogitsLoss


class DeepFCTrainer:
    @staticmethod
    def train(model_cls, train_dataloader, valid_dataloader, test_dataloader, params):
        model = model_cls().to(params["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
        criterion = BCEWithLogitsLoss()

        best_model_dict = best_loss = best_epoch = None
        for epoch in range(1, params["epochs"] + 1):
            DeepFCTrainer._train_epoch(model, train_dataloader, optimizer, criterion, params)
            valid_results = DeepFCTrainer._evaluate(model, valid_dataloader, criterion, params)
            if params["debug_print"]:
                print(f"[{epoch}/{params['epochs']}] VALID RESULTS\n{valid_results}")

            if best_loss is None or best_loss > valid_results["loss"] + params["early_stopping_epsilon"]:
                best_epoch = epoch
                best_loss = valid_results["loss"]
                best_model_dict = copy.deepcopy(model.state_dict())

            if params["early_stopping_iters"] != -1 and (epoch - best_epoch) >= params["early_stopping_iters"]:
                if params["debug_print"]:
                    print(f"EARLY STOPPING. epoch={epoch}")
                model.load_state_dict(best_model_dict)
                model.to(params["device"])
                break

        test_results = DeepFCTrainer._evaluate(model, test_dataloader, criterion, params)
        if params["debug_print"]:
            print(f"[FINITO] TEST RESULTS\n{test_results}")
        return test_results

    @staticmethod
    def _train_epoch(model, dataloader, optimizer, criterion, params):
        model.train()
        loss_sum = correct = total = 0
        for batch_num, (x, y) in enumerate(dataloader):
            x, y = x.to(params["device"]), y.to(params["device"])

            model.zero_grad()
            logits = model(x).reshape(y.shape)
            loss = criterion(logits, y)
            loss.backward()
            if params["clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
            optimizer.step()

            y_predicted = (logits > 0.5).clone().detach().type(torch.int).reshape(y.shape)
            correct += (y_predicted == y).sum()
            total += len(x)

            if params["debug_print"]:
                if batch_num % 100 == 0:
                    print(f"{batch_num} --> Loss: {loss:3.5f}")
            loss_sum += loss
        if params["debug_print"]:
            print(f"TRAIN --> avg_loss={loss_sum / (batch_num + 1)}\tacc={correct / total}")

    @staticmethod
    def _evaluate(model, dataloader, criterion, params):
        model.eval()
        with torch.no_grad():
            losses = []
            tp = fp = tn = fn = 0
            for batch_num, (x, y, lengths) in enumerate(dataloader):
                x, y = x.to(params["device"]), y.to(params["device"])

                logits = model(x, lengths).reshape_as(y)
                loss = criterion(logits, y)
                losses.append(loss)

                y_predicted = (logits > 0.5).clone().detach().type(torch.int).reshape(y.shape)
                assert list(y_predicted.shape) == list(y.shape)

                tp += torch.logical_and(y_predicted == y, y == 1).sum()
                fp += torch.logical_and(y_predicted != y, y == 0).sum()
                tn += torch.logical_and(y_predicted == y, y == 0).sum()
                fn += torch.logical_and(y_predicted != y, y == 1).sum()

            assert sum([tp, tn, fp, fn]) == len(dataloader.dataset)

            results = {}
            results["loss"] = sum(losses) / len(
                losses)  # TODO last batch might be smaller but this is not taken into account here.
            results["acc"] = (tp + tn) / (tp + tn + fp + fn)
            results["pre"] = tp / (tp + fp)
            results["rec"] = tp / (tp + fn)
            results["f1"] = 2 * results["pre"] * results["rec"] / (results["pre"] + results["rec"])
            results["confmat"] = [[tp, fp], [fn, tn]]

        return results
