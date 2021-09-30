import argparse

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import wandb

from utils import *
from metric import compute_metrics

from sklearn.model_selection import StratifiedKFold
from datasets import load_metric
from tqdm import tqdm


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    helper = DataHelper(data_dir=args.data_dir)
    total_data, train_labels = helper.preprocess()

    stratified_kfold = StratifiedKFold(n_splits=5)

    features = total_data.columns.values

    valids = []
    n_fold = 0
    for train_index, valid_index in stratified_kfold.split(
        total_data, total_data["label"]
    ):
        n_fold += 1

        train_data = total_data[features].iloc[train_index]
        valid_data = total_data[features].iloc[valid_index]

        train_label = helper.convert_labels_by_dict(train_data['label'])
        valid_label = helper.convert_labels_by_dict(valid_data['label'])

        train_data = helper.tokenize(train_data, tokenizer)
        valid_data = helper.tokenize(valid_data, tokenizer)

        RE_train_dataset = RelationExtractionDataset(train_data, train_label)
        RE_valid_dataset = RelationExtractionDataset(valid_data, valid_label)

        model_config = AutoConfig.from_pretrained(args.model_name)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=model_config
        )

        model.to(device)

        wandb.init(
            project="klue",
            entity="chungye-mountain-sherpa",
            name=args.model_name + " fold" + str(n_fold),
            group=args.model_name.split("/")[-1],
        )

        batch_size = args.batch_size
        step = args.save_steps
        training_args = TrainingArguments(
            # output directory
            output_dir="./results/" + str(n_fold),
            # number of total save model.
            save_total_limit=2,
            # model saving step.
            save_steps=step,
            # total number of training epochs
            num_train_epochs=args.epochs,
            # learning_rate
            learning_rate=args.learning_rate,

            # batch size per device during training
            per_device_train_batch_size=batch_size,
            # batch size for evaluation
            per_device_eval_batch_size=batch_size,

            # number of warmup steps for learning rate scheduler
            warmup_steps=args.warmup_steps,
            # strength of weight decay
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # directory for storing logs
            logging_dir="./logs",
            # log saving step.
            logging_steps=step,
            # evaluation strategy to adopt during training
            evaluation_strategy="steps",

            # evaluation step.
            eval_steps=step,
            report_to="wandb",
            run_name=str(n_fold) + " fold",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_valid_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            data_collator=data_collator,
        )

        trainer.train()
        model.save_pretrained(args.save_dir + '/' + str(n_fold))

        metric = load_metric("f1")
        model.eval()

        valid_dataloader = DataLoader(
            RE_valid_dataset, batch_size=batch_size, collate_fn=data_collator, drop_last=False)

        for batch in tqdm(valid_dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions,
                             references=batch["labels"])
        f1_score = metric.compute(average="micro")["f1"]

        wandb.log({"fold_best_f1": f1_score})
        valids.append(f1_score)
        wandb.finish()

    wandb.init(
        project="klue",
        entity="chungye-mountain-sherpa",
        name=args.model_name + " k fold average",
        group=args.model_name.split("/")[-1],
    )
    wandb.log({"average_f1": sum(valids) / 5})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        default="data/train.csv")

    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--save_dir", type=str, default="./best_model")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--save_steps", type=int, default=100)  # add
    parser.add_argument("--learning_rate", type=float, default=5e-5)  # add
    parser.add_argument("--weight_decay", type=float, default=0.01)  # add
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1)  # add

    args = parser.parse_args()

    wandb.login()

    train(args=args)
