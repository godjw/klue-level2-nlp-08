from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

import torch
from torch import nn
import torch.nn.functional as F

from utils import *
from metric import compute_metrics
import wandb
import optuna

# https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
# https://huggingface.co/blog/ray-tune
#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

helper = DataHelper(data_dir='data/cleaned_target_augmented.csv')
train_idxs, val_idxs = helper.split(ratio=0.1, n_splits=5, mode='plain')[0]
train_data, train_labels = helper.from_idxs(idxs=train_idxs)
val_data, val_labels = helper.from_idxs(idxs=val_idxs)

train_data = helper.tokenize(train_data, tokenizer=tokenizer)
val_data = helper.tokenize(val_data, tokenizer=tokenizer)

train_dataset = RelationExtractionDataset(
    train_data, labels=train_labels)
val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

model_config = AutoConfig.from_pretrained('klue/roberta-large')
model_config.num_labels = 30


def model_init():
    return AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=model_config)


training_args = TrainingArguments(
    output_dir='hp_search',
    evaluation_strategy='steps',
    save_total_limit=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=100,
    gradient_accumulation_steps=32,
)


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


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


# trainer = Trainer(
trainer = MyTrainer(
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 4, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 24, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.1),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8, 16, 32])
    }


trainer.hyperparameter_search(
    direction="maximize",
    hp_space=my_hp_space,
    # pruner=optuna.pruners.MedianPruner(
    #     n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    # ),
)

# Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
# trainer.hyperparameter_search(direction="maximize")
