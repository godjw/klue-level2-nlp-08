import argparse
from os import path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets.load import load_metric

from tqdm import tqdm
import wandb

from utils import DataHelper, RelationExtractionDataset, seed_everything
from metric import compute_metrics
from trainer import LDAMLossTrainer


def evaluate(model, val_dataset, batch_size, collate_fn, device, eval_method='f1'):
    metric = load_metric(eval_method)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30

    val_scores = []
    helper = DataHelper(data_dir=args.data_dir)
    for k, (train_idxs, val_idxs) in enumerate(helper.split(ratio=args.split_ratio, n_splits=args.n_splits, mode=args.mode)):
        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=model_config)
        model.to(device)

        wandb.init(
            project='klue',
            entity='chungye-mountain-sherpa',
            name=f'{args.model_name}_' + (f'fold_{k}' if args.mode == 'skf' else f'{args.mode}'),
            group=args.model_name.split('/')[-1] + args.test_name
        )

        if args.eval_strategy == 'epoch':
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=5e-05,
                weight_decay=0.01,
                num_train_epochs=args.epochs,
                logging_dir=args.logging_dir,
                logging_steps=100,
                save_total_limit=2,
                evaluation_strategy=args.eval_strategy,
                save_strategy=args.eval_strategy,
                load_best_model_at_end=True,
                metric_for_best_model='micro f1 score'
            )
        elif args.eval_strategy == 'steps':
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=5e-05,
                weight_decay=0.01,
                fp16=True,
                fp16_opt_level='O1',
                num_train_epochs=args.epochs,
                warmup_steps=args.warmup_steps,
                logging_dir=args.logging_dir,
                logging_steps=144,
                save_total_limit=1,
                evaluation_strategy=args.eval_strategy,
                save_steps=144,
                load_best_model_at_end=True,
                metric_for_best_model='auprc'
            )

        trainer = LDAMLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator
        )
        trainer.train()
        model.save_pretrained(path.join(args.save_dir, f'{k}_fold' if args.mode == 'skf' else args.mode))

        score = evaluate(
            model=model,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            device=device
        )
        val_scores.append(score)
        wandb.log({'fold': score})
        wandb.finish()

    if args.mode == 'skf':
        wandb.init(
            project='klue',
            entity='chungye-mountain-sherpa',
            name=f'{args.model_name}_{args.n_splits}_fold_avg',
            group=args.model_name.split('/')[-1] + args.test_name
        )
        wandb.log({'fold_avg_eval': sum(val_scores) / args.n_splits})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/train.csv')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./best_model')
    parser.add_argument('--test_name', type=str, default='')

    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--mode', type=str, default='plain', choices=['plain', 'skf'])
    parser.add_argument('--split_ratio', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--eval_strategy', type=str, default='steps', choices=['steps', 'epoch'])

    args = parser.parse_args()

    wandb.login()

    seed_everything(23)
    train(args=args)
