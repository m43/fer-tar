TP=0
FP=1
FN=2
TN=3
NAMES = ['ext', 'neu', 'agr', 'con', 'opn']


def eval(model, dataset):
    preds = []
    for ex in dataset:
        preds.append(model.classify(ex[2]))
    counts = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    for i_ex in range(len(preds)):
        true_lab = dataset[i_ex][2:]
        pred_lab = preds[i_ex]
        for i_tr in range(5):
            if true_lab[i_tr]:
                if pred_lab[i_tr]:
                    counts[i_tr][TP] += 1
                else:
                    counts[i_tr][FN] += 1
            else:
                if pred_lab[i_tr]:
                    counts[i_tr][FP] += 1
                else:
                    counts[i_tr][TN] += 1
    # Acc, Pre, Rec, F1
    scores = {}
    for i, trait_counts in enumerate(counts):
        acc = (trait_counts[TP] + trait_counts[TN]) / len(dataset)
        pre_denom = trait_counts[TP] + trait_counts[FP]
        rec_denom = trait_counts[TP] + trait_counts[FN]
        pre = trait_counts[TP] / pre_denom if pre_denom != 0 else None
        rec = trait_counts[TP] / rec_denom if rec_denom != 0 else None
        f1 = (2*pre*rec)/(pre+rec) if pre is not None and rec is not None else None
        scores[NAMES[i]] = (acc, pre, rec, f1)
    return scores
