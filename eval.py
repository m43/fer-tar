from dataset import TRAITS

TP, FP, FN, TN = 0, 1, 2, 3


def eval(classifier, data):
    """
    Calculates accuracy, precision, recall and the F1 score of the
    given classifier on the given dataset. The metrics are computed
    independently for each of the five traits.

    The metrics are returned in dictionary format, where the keys
    are the codes of each of the traits (ext, neu, agr, con, opn),
    and the values are also dictionaries where the key is the name
    of the metric (acc, pre, rec, f1) and the value is a float.

    :param classifier: the essay classifier; Classifier
    :param x: list of essays; list[string]
    :param y: torch tensor with targets; torch.tensor(n,5)
    :return: a dictionary with the format <trait>:<metric_name>:<metric>; dict{str:dict{str:float}}
    """
    # ~~~~~~~~~~~~~~~ GET PREDICTIONS ~~~~~~~~~~~~~~~ #
    preds, true = classifier.classify(data)
    # ~~~~~~~~~~ COMPUTE CONFUSION MATRIX ~~~~~~~~~~~ #
    counts = [
        # TP FP FN TN
        [0, 0, 0, 0],  # extroversion
        [0, 0, 0, 0],  # neuroticism
        [0, 0, 0, 0],  # agreeableness
        [0, 0, 0, 0],  # conscientiousness
        [0, 0, 0, 0]  # openness
    ]
    total = len(preds)
    for i_ex in range(total):
        true_lab = true[i_ex]
        pred_lab = preds[i_ex]
        for i_tr in range(5):
            if true_lab[i_tr] == 1.:
                if pred_lab[i_tr] == 1.:
                    counts[i_tr][TP] += 1
                else:
                    counts[i_tr][FN] += 1
            else:
                if pred_lab[i_tr] == 1.:
                    counts[i_tr][FP] += 1
                else:
                    counts[i_tr][TN] += 1

    # ~~~~~~~~~ COMPUTE EVALUATION METRICS ~~~~~~~~~~ #
    scores = {trait: {} for trait in TRAITS}
    for i, trait_counts in enumerate(counts):
        acc = (trait_counts[TP] + trait_counts[TN]) / total
        pre_denom = trait_counts[TP] + trait_counts[FP]
        rec_denom = trait_counts[TP] + trait_counts[FN]
        pre = trait_counts[TP] / pre_denom if pre_denom != 0 else None
        rec = trait_counts[TP] / rec_denom if rec_denom != 0 else None
        f1 = (2 * pre * rec) / (pre + rec) if pre and rec else None
        scores[TRAITS[i]]["acc"] = acc
        scores[TRAITS[i]]["pre"] = pre
        scores[TRAITS[i]]["rec"] = rec
        scores[TRAITS[i]]["f1"] = f1
    return scores
