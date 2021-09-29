import argparse

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import wandb

from utils import *
from metric import compute_metrics


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    helper = DataHelper(data_dir=args.data_dir)
    for train_idxs, val_idxs in helper.split(mode=args.mode, ratio=args.split_ratio):
        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,                     # output directory
            evaluation_strategy='steps',                    # evaluation strategy to adopt during training
            per_device_train_batch_size=args.batch_size,    # batch size per device during training
            per_device_eval_batch_size=args.batch_size,     # batch size for evaluation
            learning_rate=5e-5,                             # learning_rate
            weight_decay=0.01,                              # strength of weight decay
            num_train_epochs=args.epochs,                   # total number of training epochs
            warmup_steps=args.warmup_steps,                 # number of warmup steps for learning rate scheduler
            logging_dir=args.logging_dir,                   # directory for storing logs
            logging_steps=100,                              # log saving step
            save_steps=500,                                 # model saving step
            save_total_limit=2,                             # number of total save model
            eval_steps=250,                                 # evaluation step
            load_best_model_at_end=True
        )
        if args.eval_strategy == 'epoch':
            training_args.evaluation_strategy = args.eval_strategy
            training_args.save_strategy = args.eval_strategy

        trainer = Trainer(
            model=model,
            args=training_args,                             # training arguments
            train_dataset=train_dataset,                    # training dataset
            eval_dataset=val_dataset,                       # evaluation dataset
            compute_metrics=compute_metrics,                # define metrics function
            data_collator=data_collator
        )

        trainer.train()
        model.save_pretrained(args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/train.csv')
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--mode', type=str, default='plain', choices=['plain', 'skf'])
    parser.add_argument('--split_ratio', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str, default='./best_model')
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_strategy', type=str, default='steps')

    args = parser.parse_args()

    wandb.login()
    wandb.init(
        project='klue',
        entity='chungye-mountain-sherpa',
        name=args.model_name,
        group=args.model_name.split('/')[-1]
    )

    train(args=args)
