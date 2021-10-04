import argparse
from os import path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets.load import load_metric
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger

from tqdm import tqdm
import wandb

from utils import DataHelper, RelationExtractionDataset, seed_everything
from metric import compute_metrics

def search(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30

    helper = DataHelper(data_dir=args.data_dir)
    for train_idxs, val_idxs in helper.split(ratio=args.split_ratio, n_splits=args.n_splits, mode=args.mode):
        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(args.model_name, config=model_config)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            save_total_limit=1,
            evaluation_strategy=args.eval_strategy,
            save_steps=100,
        )
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )

        tune_config = {
            'wandb': {
                'project': 'klue',
                'entity': 'chungye-mountain-sherpa',
                'name': args.model_name,
                'group': args.model_name.split('/')[-1] + '_hyperparameter_search',
                'log_config': True
            },
            'per_device_train_batch_size': tune.choice([4, 16, 32, 48, 96]),
            'per_device_eval_batch_size': 48,
            'gradient_accumulation_steps': tune.choice([1, 4, 16, 32]),
            'learning_rate': tune.uniform(1e-4, 1e-5),
            'weight_decay': tune.uniform(0, 0.01),
            'num_train_epochs': tune.choice([3, 4, 5]),
            'seed': tune.choice([34, 42, 50])
        }
        scheduler = PopulationBasedTraining(
            time_attr='training_iteration',
            metric='eval_micro f1 score',
            mode='max',
            perturbation_interval=1,
            hyperparam_mutations={
                'per_device_train_batch_size': [4, 16, 32, 48, 96],
                'gradient_accumulation_steps': [1, 4, 16, 32],
                'num_train_epochs': [3, 4, 5],
                'learning_rate': tune.uniform(1e-4, 1e-5),
                'weight_decay': tune.uniform(0.0, 0.03),
                'seed': [34, 42, 50]
            }
        )

        trainer.hyperparameter_search(
            direction='maximize',
            hp_space=lambda _: tune_config,
            backend='ray',
            n_trials=16,
            resources_per_trial={
                "cpu": 8,
                "gpu": 1
            },
            scheduler=scheduler,
            loggers=DEFAULT_LOGGERS + (WandbLogger,),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/gen_add_small_train.csv')
    parser.add_argument('--output_dir', type=str, default='./hyperparameter_search')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./best_model')
    parser.add_argument('--test_name', type=str, default='')

    parser.add_argument('--model_name', type=str, default='klue/roberta-small')
    parser.add_argument('--mode', type=str, default='plain', choices=['plain', 'skf'])
    parser.add_argument('--split_ratio', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--eval_strategy', type=str, default='steps', choices=['steps', 'epoch'])

    args = parser.parse_args()

    wandb.login()

    search(args=args)
