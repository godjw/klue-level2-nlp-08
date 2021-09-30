import argparse

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
import wandb

from utils import *
from metric import compute_metrics
import os
import random
import numpy as np


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_helper = DataHelper(data_dir=args.train_data_dir)
    train_preprocessed, train_labels = train_helper.preprocess()
    train_data = train_helper.tokenize(
        data=train_preprocessed, tokenizer=tokenizer)
    train_dataset = RelationExtractionDataset(train_data, train_labels)

    valid_helper = DataHelper(data_dir=args.valid_data_dir)
    valid_preprocessed, valid_labels = valid_helper.preprocess()
    valid_data = valid_helper.tokenize(
        data=valid_preprocessed, tokenizer=tokenizer)
    valid_dataset = RelationExtractionDataset(valid_data, valid_labels)

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=model_config
    )
    print(model.config)
    model.parameters
    model.to(device)

    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
    if args.eval_strategy == 'epoch':  # evaluation at epochs
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            save_strategy='epoch',
            evaluation_strategy='epoch',
            save_total_limit=2,
            num_train_epochs=args.epochs,
            learning_rate=8.643223664444307e-05,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            warmup_steps=args.warmup_steps,
            weight_decay=0.029856983237295187,
            logging_dir=args.logging_dir,
            logging_steps=100,
            gradient_accumulation_steps=32,
        )
    elif args.eval_strategy == 'steps':  # evaluation at steps
        training_args = TrainingArguments(
            output_dir=args.output_dir,                     # output directory
            save_total_limit=5,  # number of total save model.
            save_steps=500,  # model saving step.
            num_train_epochs=args.epochs,  # total number of training epochs
            learning_rate=5e-5,  # learning_rate
            # batch size per device during training
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            # number of warmup steps for learning rate scheduler
            warmup_steps=args.warmup_steps,
            weight_decay=0.01,  # strength of weight decay
            logging_dir=args.logging_dir,  # directory for storing logs
            logging_steps=250,  # log saving step.
            evaluation_strategy='steps',  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=500,  # evaluation step.
            load_best_model_at_end=True
        )
    trainer = Trainer(
        model=model,
        args=training_args,                             # training arguments, defined above
        train_dataset=train_dataset,                    # training dataset
        eval_dataset=valid_dataset,                     # evaluation dataset
        compute_metrics=compute_metrics,                # define metrics function
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(args.save_dir)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


if __name__ == '__main__':
    seed_everything(30)

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data_dir', type=str,
                        default='data/train_10.csv')
    parser.add_argument('--valid_data_dir', type=str,
                        default='data/valid_10.csv')

    parser.add_argument('--model_name', type=str, default='klue/roberta-large')
    parser.add_argument('--output_dir', type=str, default='./best_hp_results')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str,
                        default='./best_hp_klue_roberta_large_model')
    parser.add_argument('--warmup_steps', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_strategy', type=str, default='epoch')

    args = parser.parse_args()

    # wandb.login()
    wandb.init(
        project='klue',
        entity='chungye-mountain-sherpa',
        name=args.model_name + '/best-hp',
        group=args.model_name.split('/')[-1]
    )
    # NOTE: wandb disable
    # os.environ["WANDB_DISABLED"] = "true"

    train(args=args)
