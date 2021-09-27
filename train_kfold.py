import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from load_data import *
from klue_re_metric import *
from datasets import load_metric
from tqdm import tqdm
import wandb


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():

    MODEL_NAME = "klue/bert-base"
    # MODEL_NAME = "klue/roberta-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    stratified_kfold = StratifiedKFold(n_splits=5)
    total_data = load_data("../dataset/train/train.csv")

    features = total_data.columns.values
    # for train_index, test_index in stratified_kfold.split(data_df, data_df['label']):
    #     train_df = data_df[]
    #     print(data_df['label'][test_index].value_counts())
    valids = []

    n_fold = 0
    for train_index, valid_index in stratified_kfold.split(
        total_data, total_data["label"]
    ):
        n_fold += 1
        train_data = total_data[features].iloc[train_index]
        valid_data = total_data[features].iloc[valid_index]
        # train_data = total_data[features].iloc[0:20]
        # valid_data = total_data[features].iloc[0:20]

        train_label = label_to_num(train_data["label"].values)
        valid_label = label_to_num(valid_data["label"].values)

        # print(valid_data['label'][valid_index].value_counts())

        tokenized_train = tokenized_dataset(train_data, tokenizer)
        tokenized_valid = tokenized_dataset(valid_data, tokenizer)

        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')

        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )
        model.parameters
        model.to(device)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

        for param in model.bert.pooler.parameters():
            param.requires_grad = True

        grad_layer = list(range(6, 12))
        for i in grad_layer:
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # wandb.init()
        training_args = TrainingArguments(
            output_dir="./results/" + str(n_fold),  # output directory
            save_total_limit=5,  # number of total save model.
            save_steps=100,  # model saving step.
            num_train_epochs=2,  # total number of training epochs
            learning_rate=5e-5,  # learning_rate
            per_device_train_batch_size=256,  # batch size per device during training
            per_device_eval_batch_size=256,  # batch size for evaluation
            warmup_steps=300,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=100,  # log saving step.
            evaluation_strategy="steps",  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=100,  # evaluation step.
            report_to="wandb",
            run_name=str(n_fold) + " fold",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
        )

        # train model
        trainer.train()
        # model.save_pretrained('./best_model')
        model.save_pretrained("./results/" + str(n_fold))
        metric = load_metric("f1")

        model.eval()

        valid_dataloader = DataLoader(RE_valid_dataset, batch_size=128, drop_last=False)
        for batch in tqdm(valid_dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # batch = {k: v.to("cpu") for k, v in batch.items()}
            # print(batch)
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        f1_score = metric.compute(average="micro")["f1"]
        wandb.log({"f1": f1_score})
        valids.append(f1_score)
        wandb.finish()
    wandb.init(project="huggingface", name="total")
    wandb.log({"average_f1": sum(valids) / 5})


def main():
    train()


if __name__ == "__main__":
    main()
