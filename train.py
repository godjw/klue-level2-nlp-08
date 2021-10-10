import argparse
from os import path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets.load import load_metric

from tqdm import tqdm
import wandb

from trainer import MyTrainer, init_tarining_arguments
from utils import RelationExtractionDataset, DataHelper, ConfigParser
from model.metric import compute_metrics
import os
import random
import numpy as np


def evaluate(model, val_dataset, batch_size, collate_fn, device, eval_method='f1'):
    metric = load_metric(eval_method)
    dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    model.eval()
    for data in tqdm(dataloader):
        data = {key: value.to(device) for key, value in data.items()}
        with torch.no_grad():
            outputs = model(**data)
        preds = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=preds, references=data['labels'])
    model.train()

    return metric.compute(average='micro')[eval_method]


def train(args):
    evaluation_strategy = args.evaluation_strategy
    assert (evaluation_strategy not in args.config.split('/')[1]) == False,\
        "not matched evaluation strategy and config file"

    # Config parse and init configures
    config = ConfigParser(config=args.config).config
    data_config = config['data']
    training_arguments_config = config['training_arguments']
    hyperparameter_config = config['training_arguments']['hyperparameter']
    model_dir = config['model_dir']
    mode = args.mode
    wandb_config = config['wandb']
    disable_wandb = args.disable_wandb

    if disable_wandb == True:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.login()

    # Fix all seeds
    seed_everything(hyperparameter_config['seed'])

    # init device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # init model config of transformers
    model_config = AutoConfig.from_pretrained(model_dir, num_labels=30)

    val_scores = []
    helper = DataHelper(data_dir=data_config['data_dir'],
                        add_ent_token=data_config['add_ent_token'],
                        aug_data_dir=data_config['aug_data_dir'])

    # train loop
    for k, (train_idxs, val_idxs) in enumerate(helper.split(ratio=data_config['split_ratio'], n_splits=data_config['n_splits'], mode=mode, random_seed=hyperparameter_config['seed'])):
        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(
            train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, config=model_config)
        model.to(device)

        if args.disable_wandb == False:
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                name=wandb_config['name'],
                group=wandb_config['group']
            )

        training_args = init_tarining_arguments(
            evaluation_strategy, training_arguments_config, hyperparameter_config)

        trainer = MyTrainer(
            disable_wandb=disable_wandb,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
        trainer.train()
        model.save_pretrained(
            path.join(training_arguments_config['save_dir'], f'{k}_fold' if mode == 'skf' else mode))

        score = evaluate(
            model=model,
            val_dataset=val_dataset,
            batch_size=hyperparameter_config['batch_size'],
            collate_fn=data_collator,
            device=device
        )
        val_scores.append(score)

        if disable_wandb == False:
            wandb.log({'fold': score})
            wandb.finish()

    if mode == 'skf' and disable_wandb == False:
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=wandb_config['name'],
            group=wandb_config['group']
        )
        wandb.log({'fold_avg_eval': sum(val_scores) / data_config['n_splits']})


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        default='config/eval_epoch_config.json')
    parser.add_argument('--evaluation_strategy', type=str,
                        default='epoch', choices=['steps', 'epoch'])
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--disable_wandb', type=bool, default=True)

    args = parser.parse_args()

    train(args=args)
