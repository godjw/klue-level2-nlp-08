import argparse
from os import path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from custom_model import RobertaEmbeddings
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from datasets.load import load_metric

from tqdm import tqdm
import wandb

from utils import RelationExtractionDataset, DataHelper, ConfigParser
from metric import compute_metrics
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
    def __init__(self, beta, gamma, epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, labels, loss_type='softmax'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor(
            [sum(labels == i) for i in range(logits.shape[1])])
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num) + 1e-8)

        weights = weights / torch.sum(weights) * no_of_classes
        labels = labels.reshape(-1, 1)

        weights = torch.tensor(weights.clone().detach()).float()

        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = torch.zeros(
                len(labels), no_of_classes).cuda().scatter_(1, labels, 1).cuda()

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + \
            self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, pos_weight=weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


class MyTrainer(Trainer):
    def __init__(self, disable_wandb=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_wandb = disable_wandb

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_type = "focal"
        beta = 0.9999
        gamma = 2.0

        criterion = CB_loss(beta, gamma)
        if torch.cuda.is_available():
            criterion.cuda()
        loss_fct = criterion(logits, labels, loss_type)

        return (loss_fct, outputs) if return_outputs else loss_fct

    def evaluation_loop(self, *args, **kwargs):
        eval_loop_output = super().evaluation_loop(*args, **kwargs)

        pred = eval_loop_output.predictions
        label_ids = eval_loop_output.label_ids

        cm = confusion_matrix(label_ids, np.argmax(pred, axis=-1))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cmn = cmn.astype('int')
        fig = plt.figure(figsize=(22, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        cm_plot = sns.heatmap(cm, cmap='Blues', fmt='d', annot=True, ax=ax1)
        cm_plot.set_xlabel('pred')
        cm_plot.set_ylabel('true')
        cm_plot.set_title('confusion matrix')
        cmn_plot = sns.heatmap(
            cmn, cmap='Blues', fmt='d', annot=True, ax=ax2)
        cmn_plot.set_xlabel('pred')
        cmn_plot.set_ylabel('true')
        cmn_plot.set_title('confusion matrix normalize')
        if self.disable_wandb:
            wandb.init(
                project='yohan',
                entity='chungye-mountain-sherpa',
                name='base',
                # group='koelectra/' + args.model_name.split('/')[-1]
                group='koelectra'
            )
            wandb.log({'confusion_matrix': wandb.Image(fig)})

        return eval_loop_output


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
        train_data, train_labels = helper.from_idxs(idxs=train_idxs)
        val_data, val_labels = helper.from_idxs(idxs=val_idxs)

        train_data = helper.tokenize(train_data, tokenizer=tokenizer)
        val_data = helper.tokenize(val_data, tokenizer=tokenizer)

        train_dataset = RelationExtractionDataset(
            train_data, labels=train_labels)
        val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, config=model_config)

        if args.entity_embedding:
            custom_embedding = RobertaEmbeddings(model, config=model_config)
            model.roberta.embeddings = custom_embedding
            print(model.roberta.embeddings)

        model.to(device)

        if args.disable_wandb == False:
            wandb.init(
                project='yohan',
                entity='chungye-mountain-sherpa',
                # name=f'{args.model_name}_' +
                # (f'fold_{k}' if args.mode == 'skf' else f'{args.mode}'),
                name='base',
                # group='koelectra/' + args.model_name.split('/')[-1]
                group='koelectra'
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
            disable_wandb=args.disable_wandb,
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
            project='yohan',
            entity='chungye-mountain-sherpa',
            # name=f'{args.model_name}_{args.n_splits}_fold_avg',
            name='base',
            # group='koelectra/' + args.model_name.split('/')[-1]
            group='koelectra'
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
    # default='hp_config/roberta_small.json')

    # parser.add_argument('--data_dir', type=str, default='data/change_label_entities_dataset.csv')
    # parser.add_argument('--data_dir', type=str, default='data/cleaned_target_augmented.csv')
    parser.add_argument('--data_dir', type=str,
                        default='data/prinstine_data.csv')
    parser.add_argument('--aug_data_dir', type=str,
                        default='data/ja_back_translation.csv')
    parser.add_argument('--output_dir', type=str,
                        default='./backtranslation')
    parser.add_argument('--logging_dir', type=str, default='./logs')
    parser.add_argument('--save_dir', type=str,
                        default='./backtranslation')

    parser.add_argument('--model_name', type=str, default='klue/roberta-large')
    # parser.add_argument('--model_name', type=str,
    #                     default='monologg/koelectra-base-v3-discriminator')
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--split_ratio', type=float, default=0.1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--eval_strategy', type=str,
                        default='epoch', choices=['steps', 'epoch'])
    parser.add_argument('--add_ent_token', type=bool, default=True)
    parser.add_argument('--disable_wandb', type=bool, default=True)
    parser.add_argument('--entity_embedding', type=bool, default=False)

    args = parser.parse_args()

    train(args=args)

#
