import argparse
from os import path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets.load import load_metric

from tqdm import tqdm
import wandb

from utils import RelationExtractionDataset, DataHelper, ConfigParser
from metric import compute_metrics
from GetModel import GetModel

import os
import random
import numpy as np


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
        labels = inputs.pop("labels")
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        # try giving more weight to no_relation
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        loss = ce_loss(outputs.get('logits'), labels)
        return (loss, outputs) if return_outputs else loss.mean()


def evaluate(model, val_dataset, batch_size, collate_fn, device, eval_method='f1'):
    metric = load_metric(eval_method)
    dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    model.eval()
    for data in tqdm(dataloader):
        data = {key: value.to(device) for key, value in data.items()}
        with torch.no_grad():
            outputs = model(data['input_ids'], data['attention_mask'])
        preds = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=preds, references=data['labels'])
    model.train()

    return metric.compute(average='micro')[eval_method]


def train(args):
    hp_config = ConfigParser(config=args.hp_config).config
    seed_everything(hp_config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30

    if args.disable_wandb == True:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.login()

    val_scores = []
    helper = DataHelper(data_dir=args.data_dir,
                        add_ent_token=args.add_ent_token,
                        aug_data_dir=args.aug_data_dir)
    for k, (train_idxs, val_idxs) in enumerate(helper.split(ratio=args.split_ratio, n_splits=args.n_splits, mode=args.mode, random_seed=hp_config['seed'])):
        # 여기 숫자 수정해주세요!
        # 준영님 2
        # 진성님 3
        # 하겸님 4
        if k != 4:
            continue
        ####

        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(
            train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        if args.new_hat == False:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, config=model_config)
        else:
            model = GetModel(config=model_config)

        model.to(device)

        if args.disable_wandb == False:
            wandb.init(
                project='klue',
                entity='chungye-mountain-sherpa',
                name=f'{args.model_name}_large_lstm_dropout_' +
                (f'fold_{k}' if args.mode == 'skf' else f'{args.mode}'),
                group='roberta-large-lstm_'
            )

        if args.eval_strategy == 'epoch':
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=hp_config['batch_size'],
                per_device_eval_batch_size=hp_config['batch_size'],
                gradient_accumulation_steps=hp_config['gradient_accumulation_steps'],
                learning_rate=hp_config['learning_rate'],
                weight_decay=hp_config['weight_decay'],
                num_train_epochs=hp_config['epochs'],
                logging_dir=args.logging_dir,
                logging_steps=200,
                save_total_limit=2,
                evaluation_strategy=args.eval_strategy,
                save_strategy=args.eval_strategy,
                load_best_model_at_end=True,
                metric_for_best_model='micro f1 score',
                fp16=True,
                fp16_opt_level='O1'
            )
        elif args.eval_strategy == 'steps':
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=hp_config['batch_size'],
                per_device_eval_batch_size=hp_config['batch_size'],
                gradient_accumulation_steps=hp_config['gradient_accumulation_steps'],
                learning_rate=hp_config['learning_rate'],
                weight_decay=hp_config['weight_decay'],
                num_train_epochs=hp_config['epochs'],
                logging_dir=args.logging_dir,
                logging_steps=200,
                save_total_limit=2,
                evaluation_strategy=args.eval_strategy,
                eval_steps=200,
                save_steps=200,
                load_best_model_at_end=True,
                metric_for_best_model='micro f1 score',
                fp16=True,
                fp16_opt_level='O1'
            )

        # trainer = Trainer(
        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
        trainer.train()
        model.save_pretrained(
            path.join(args.save_dir, f'{k}_fold' if args.mode == 'skf' else args.mode))

        score = evaluate(
            model=model,
            val_dataset=val_dataset,
            batch_size=hp_config['batch_size'],
            collate_fn=data_collator,
            device=device
        )
        val_scores.append(score)

        if args.disable_wandb == False:
            wandb.log({'fold': score})
            wandb.finish()

    if args.mode == 'skf' and args.disable_wandb == False:
        wandb.init(
            project='klue',
            entity='chungye-mountain-sherpa',
            name=f'{args.model_name}_{args.n_splits}_fold_avg',
            group='roberta-large-lstm'
        )
        wandb.log({'fold_avg_eval': sum(val_scores) / args.n_splits})


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

    parser.add_argument('--hp_config', type=str,
                        default='hp_config/roberta_large_focal_loss.json')

    parser.add_argument('--data_dir', type=str,
                        default='data/cleaned_target_augmented.csv')
    parser.add_argument('--aug_data_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str,
                        default='./results')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str,
                        default='./best_model')

    parser.add_argument('--model_name', type=str, default='klue/roberta-large')
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--eval_strategy', type=str,
                        default='epoch', choices=['steps', 'epoch'])
    parser.add_argument('--add_ent_token', type=bool, default=True)
    parser.add_argument('--disable_wandb', type=bool, default=False)
    parser.add_argument('--new_hat', type=bool,
                        default=False, choices=[True, False])

    args = parser.parse_args()

    train(args=args)
