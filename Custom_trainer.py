from torch import nn
from transformers import Trainer
from loss_func import *
from sklearn.model_selection import StratifiedKFold
from utils import *


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_type = "focal"
        beta = 0.9999
        gamma = 2.0

        criterion = CB_loss(beta, gamma)
        if torch.cuda.is_available():
            criterion.cuda()
        loss_fct = criterion(logits, labels, loss_type)

        return (loss_fct, outputs) if return_outputs else loss_fct
