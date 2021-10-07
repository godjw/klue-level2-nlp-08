import numpy as np
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, f1_score


def only_no_rel_micro_f1(preds, labels):
    """
    KLUE-RE micro f1 (without no_relation)
    """

    label_list = [
        'relation', 'no_relation'
    ]
    label_indices = list(range(len(label_list)))
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def only_rel_micro_f1(preds, labels):
    """
    KLUE-RE micro f1 (without no_relation)
    """

    label_list = [
        'org:top_members/employees', 'org:members',
        'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product',
        'org:number_of_employees/members', 'per:children',
        'per:place_of_residence', 'per:alternate_names',
        'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
        'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved',
        'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
        'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
        'per:religion'
    ]
    label_indices = list(range(1, len(label_list) + 1))
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def no_rel_auprc(probs, labels):
    """
    KLUE-RE AUPRC (with no_relation)
    """

    labels = np.eye(2)[labels]

    score = np.zeros((2,))
    for c in range(2):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0


def rel_auprc(probs, labels):
    """
    KLUE-RE AUPRC (with no_relation)
    """

    labels = np.eye(29)[labels]

    score = np.zeros((29,))
    for c in range(29):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = precision_recall_curve(targets_c, preds_c)
        score[c] = auc(recall, precision)
    return np.average(score) * 100.0


def no_rel_compute_metrics(pred):
    """
    Returns metrics for prediction evaluation
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    f1 = only_no_rel_micro_f1(preds, labels)
    auprc = no_rel_auprc(probs, labels)
    acc = accuracy_score(labels, preds)

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def rel_compute_metrics(pred):
    """
    Returns metrics for prediction evaluation
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    f1 = only_rel_micro_f1(preds, labels)
    auprc = rel_auprc(probs, labels)
    acc = accuracy_score(labels, preds)

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }
