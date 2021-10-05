from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

helper = DataHelper(data_dir='/opt/ml/dataset/train/preprocess_train.csv')
train_idxs, val_idxs = helper.split(ratio=0.1, n_splits=5, mode='plain')[0]
train_data, train_labels = helper.from_idxs(idxs=train_idxs)
val_data, val_labels = helper.from_idxs(idxs=val_idxs)

train_data = helper.entity_tokenize(train_data, tokenizer=tokenizer)
val_data = helper.entity_tokenize(val_data, tokenizer=tokenizer)

train_dataset = RelationExtractionDataset(
    train_data, labels=train_labels)
val_dataset = RelationExtractionDataset(val_data, labels=val_labels)


def model_init():
    model_config = AutoConfig.from_pretrained(
        'klue/roberta-large', num_labels=30)
    model = RBERT(model_name='klue/roberta-large',
                  config=model_config, dropout_rate=0.1)
    return model


wandb.login()

wandb.init(
    project='hp_search',
    entity='chungye-mountain-sherpa',
    name='entitiy_fc_roberta_large'
)
training_args = TrainingArguments(
    output_dir='hp_search',
    evaluation_strategy='steps',
    save_total_limit=1,
    num_train_epochs=2,
    learning_rate=5e-5,
    per_device_train_batch_size=45,
    per_device_eval_batch_size=45,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=20,
    gradient_accumulation_steps=1,
    report_to="wandb",
    load_best_model_at_end=True,
    fp16=True,
    fp16_opt_level='O1'
)


trainer = Trainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True),
        "seed": trial.suggest_int("seed", 1, 42),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 45]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.3),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8])
    }


trainer.hyperparameter_search(
    direction="maximize",
    hp_space=my_hp_space,
    # pruner=optuna.pruners.MedianPruner(
    #     n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    # ),
)
