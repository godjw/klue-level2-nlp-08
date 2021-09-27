import argparse

import torch

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import *
from metric import compute_metrics


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])

    helper = DataHelper(data_dir=args['data_dir'])
    preprocessed, train_labels = helper.preprocess()
    train_data = helper.tokenize(data=preprocessed, tokenizer=tokenizer)

    train_dataset = RelationExtractionDataset(train_data, train_labels)

    model_config = AutoConfig.from_pretrained(args['model_name'])
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(
        args['model_name'], config=model_config
    )
    print(model.config)
    model.parameters
    model.to(device)

    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
    training_args = TrainingArguments(
        output_dir='./results',         # output directory
        save_total_limit=5,             # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=20,            # total number of training epochs
        learning_rate=5e-5,             # learning_rate
        per_device_train_batch_size=96, # batch size per device during training
        per_device_eval_batch_size=96,  # batch size for evaluation
        warmup_steps=100,               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,              # strength of weight decay
        logging_dir='./logs',           # directory for storing logs
        logging_steps=100,              # log saving step.
        evaluation_strategy='steps',    # evaluation strategy to adopt during training
                                        # `no`: No evaluation during training.
                                        # `steps`: Evaluate every `eval_steps`.
                                        # `epoch`: Evaluate every end of epoch.
        eval_steps=250,                 # evaluation step.
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,             # training arguments, defined above
        train_dataset=train_dataset,    # training dataset
        eval_dataset=train_dataset,     # evaluation dataset
        compute_metrics=compute_metrics # define metrics function
    )

    trainer.train()
    model.save_pretrained(args['save_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../dataset/train/train.csv')

    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--save_dir', type=str, default='./best_model')
    
    args = parser.parse_args()

    train(args=args)
