from numpy.core.fromnumeric import argmax
import torch
import numpy as np
from transformers import Trainer
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from loss import WeightedFocalLoss


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        loss_fct = WeightedFocalLoss()
        outputs = model(**inputs)
        if labels is not None:
            loss = loss_fct(outputs[0], labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, *args, **kwargs):
        eval_loop_output = super().evaluation_loop(*args, **kwargs)

        pred = eval_loop_output.predictions
        label_ids = eval_loop_output.label_ids

        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100
        cmn = cmn.astype('int')
        fig = plt.figure(figsize=(22, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        cm_plot = sns.heatmap(cm, cmap='Blues', fmt='d', annot=True, ax=ax1)
        cm_plot.set_xlabel('pred')
        cm_plot.set_ylabel('true')
        cm_plot.set_title('confusion matrix')
        cmn_plot = sns.heatmap(
            cmn, cmap='Blues', fmt='d', annot=True, ax=ax2)
        cmn_plot.set_xlabel('pred')
        cmn_plot.set_ylabel('true')
        cmn_plot.set_title('confusion matrix normalize')
        wandb.init(
            project='klue',
            entity='chungye-mountain-sherpa',
            name='klue/roberta-small_plain',
            group='roberta-small'
        )
        wandb.log({'confusion_matrix': wandb.Image(fig)})

        return eval_loop_output
