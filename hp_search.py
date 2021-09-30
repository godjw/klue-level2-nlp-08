from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

import torch

from utils import *
from metric import compute_metrics
import wandb
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

# https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
# https://huggingface.co/blog/ray-tune
#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_helper = DataHelper(data_dir='data/train_10.csv')
train_preprocessed, train_labels = train_helper.preprocess()
train_data = train_helper.tokenize(
    data=train_preprocessed, tokenizer=tokenizer)
train_dataset = RelationExtractionDataset(train_data, train_labels)

valid_helper = DataHelper(data_dir='data/valid_10.csv')
valid_preprocessed, valid_labels = valid_helper.preprocess()
valid_data = valid_helper.tokenize(
    data=valid_preprocessed, tokenizer=tokenizer)
valid_dataset = RelationExtractionDataset(valid_data, valid_labels)

model_config = AutoConfig.from_pretrained('klue/roberta-large')
model_config.num_labels = 30


def model_init():
    return AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=model_config)


# Evaluate during training and a bit more often than the default to be able to prune bad trials early.
# Disabling tqdm is a matter of preference.
training_args = TrainingArguments(
    output_dir='hp_search',
    evaluation_strategy='steps',
    save_total_limit=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=100,
    gradient_accumulation_steps=32,
)
trainer = Trainer(
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
)


def my_hp_space(trial):
    from ray import tune
    return {
        "learning_rate": tune.uniform(1e-6, 1e-4, log=True),
        "per_device_train_batch_size": tune.choice([4, 8, 16, 24, 32]),
        "warmup_steps": tune.choice(range(50, 500)),
        "weight_decay": tune.uniform(0.0, 0.3),
        "gradient_accumulation_steps": tune.suggest_categorical("gradient_accumulation_steps", [2, 4, 8, 16, 32])
        # "learning_rate": trial.suggest_float("learning_rate", 3e-6, 1e-5, log=True),
        # # "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        # # "seed": trial.suggest_int("seed", 1, 42),
        # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 24, 32]),
        # "warmup_steps": trial.suggest_int("warmup_steps", 50, 500),
        # "weight_decay": trial.suggest_float("weight_decay", 0, 0.1),
        # "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8, 16, 32])
    }


wandb.init(
    project='ray',
    name='hp-search',
)


trainer.hyperparameter_search(
    direction="maximize", backend="ray", hp_space=my_hp_space)


# Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
# trainer.hyperparameter_search(direction="maximize")
