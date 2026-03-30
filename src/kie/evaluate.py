import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score

from configs.layoutlmv3_config import ID2LABEL



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_labels = []
        gold_labels = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            pred_labels.append(ID2LABEL[int(pred_id)])
            gold_labels.append(ID2LABEL[int(label_id)])
        true_predictions.append(pred_labels)
        true_labels.append(gold_labels)

    return {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
    }
