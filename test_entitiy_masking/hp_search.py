from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback, AutoModelForSequenceClassification

import torch
from torch import nn
import torch.nn.functional as F

from utils import *
from metric import compute_metrics
import wandb
import optuna
from custom_model import RBERT
from torch.utils.data import DataLoader
import os
import pandas as pd
from loss_func import CB_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

helper = DataHelper(data_dir='/opt/ml/dataset/train/train.csv')
#entity_data = pd.read_csv('/opt/ml/dataset/train/entity_train.csv')

#feature = entity_data.columns

# for k, (ti, vi) in enumerate(helper.split(ratio=0.2, n_splits=5, mode='skf')):
#     train_idxs = ti
#     val_idxs = vi
#     break

train_idxs, val_idxs = helper.split(ratio=0.1, n_splits=5, mode='plain')[0]

#train_entity = entity_data[feature].iloc[train_idxs]

train_data, train_labels = helper.from_idxs(idxs=train_idxs)
#val_entity = entity_data[feature].iloc[val_idxs]

val_data, val_labels = helper.from_idxs(idxs=val_idxs)

train_data = helper.tokenize(
    train_data, tokenizer=tokenizer)
val_data = helper.tokenize(
    val_data, tokenizer=tokenizer)

train_dataset = RelationExtractionDataset(
    train_data, labels=train_labels)
val_dataset = RelationExtractionDataset(val_data, labels=val_labels)


def model_init():
    model_config = AutoConfig.from_pretrained(
        'klue/roberta-large')
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        'klue/roberta-large', config=model_config)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.roberta.encoder.layer[-1].parameters():
        param.requires_grad = True
    for param in model.roberta.encoder.layer[-2].parameters():
        param.requires_grad = True

    return model


wandb.login()

wandb.init(
    project='hp_search',
    entity='chungye-mountain-sherpa',
    name='entitiy_fc_roberta_large'
)


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss_type = "focal"
        beta = 0.9999
        gamma = 2.0

        criterion = CB_loss(beta, gamma)
        if torch.cuda.is_available():
            criterion.cuda()
        loss_fct = criterion(logits, labels, loss_type)

        return (loss_fct, outputs) if return_outputs else loss_fct


training_args = TrainingArguments(
    output_dir='hp_search',
    evaluation_strategy='steps',
    save_total_limit=1,
    num_train_epochs=2,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=50,
    gradient_accumulation_steps=1,
    report_to="wandb",
    load_best_model_at_end=True,
    fp16=True,
    fp16_opt_level='O1'
)


trainer = MyTrainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True),
        "seed": trial.suggest_int("seed", 1, 42),
        "num_train_epochs": trial.suggest_int("num_train_epochs",  4, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [128, 256, 512]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.3),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1])
    }


trainer.hyperparameter_search(
    direction="maximize",
    hp_space=my_hp_space,
    # pruner=optuna.pruners.MedianPruner(
    #     n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    # ),
)
