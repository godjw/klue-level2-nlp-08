import torch

from transformers import Trainer

import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from model.loss import CB_loss, LDAMLoss


class MyTrainer(Trainer):
    def __init__(self, disable_wandb=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_wandb = disable_wandb

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

    def evaluation_loop(self, *args, **kwargs):
        eval_loop_output = super().evaluation_loop(*args, **kwargs)

        pred = eval_loop_output.predictions
        label_ids = eval_loop_output.label_ids

        self.draw_confusion_matrix(pred, label_ids)
        return eval_loop_output

    def draw_confusion_matrix(self, pred, label_ids):
        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
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
        if self.disable_wandb:
            wandb.init(
                project='yohan',
                entity='chungye-mountain-sherpa',
                name='base',
                group='koelectra'
            )
            wandb.log({'confusion_matrix': wandb.Image(fig)})

class LDAMLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_per_labels = self.train_dataset.get_n_per_labels()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')

        betas = [0, 0.99]
        beta_idx = self.state.epoch >= 2
        n_per_labels = self.n_per_labels

        effective_num = 1.0 - np.power(betas[beta_idx], n_per_labels)
        cls_weights = (1.0 - betas[beta_idx]) / np.array(effective_num)
        cls_weights = cls_weights / np.sum(cls_weights) * len(n_per_labels)
        cls_weights = torch.FloatTensor(cls_weights)

        criterion = LDAMLoss(cls_num_list=n_per_labels, max_m=0.5, s=30, weight=cls_weights)
        if torch.cuda.is_available():
            criterion.cuda()

        loss_fct = criterion(logits, labels)
        return (loss_fct, outputs) if return_outputs else loss_fct
