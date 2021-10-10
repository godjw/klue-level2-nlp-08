from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback

import torch
from torch import nn
import torch.nn.functional as F

from utils import *
from metric import compute_metrics
import wandb
import optuna

# https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
# https://huggingface.co/blog/ray-tune
#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 'monologg/koelectra-base-v3-discriminator'
tokenizer = AutoTokenizer.from_pretrained(
    'klue/roberta-large')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

helper = DataHelper(data_dir='data/train.csv', mode='train',
                    add_ent_token=True, aug_data_dir='')
train_idxs, val_idxs = helper.split(ratio=0.1, n_splits=5, mode='plain')[0]

train_data, train_labels = helper.from_idxs(idxs=train_idxs)
val_data, val_labels = helper.from_idxs(idxs=val_idxs)

train_data = helper.tokenize(train_data, tokenizer=tokenizer)
val_data = helper.tokenize(val_data, tokenizer=tokenizer)

train_dataset = RelationExtractionDataset(
    train_data, labels=train_labels)
val_dataset = RelationExtractionDataset(val_data, labels=val_labels)

model_config = AutoConfig.from_pretrained(
    'klue/roberta-large')
model_config.num_labels = 30


def model_init():
    return AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', config=model_config)


# wandb.login()

training_args = TrainingArguments(
    output_dir='hp_search',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    num_train_epochs=4,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=100,
    gradient_accumulation_steps=32,
    load_best_model_at_end=True,
    report_to='wandb',
    fp16=True,
    fp16_opt_level='O1'
)


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


# trainer = Trainer(
trainer = MyTrainer(
    args=training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    model_init=model_init,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True),
        "seed": trial.suggest_int("seed", 1, 123),
        # "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 6),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 24, 32, 45]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.3),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [8, 16, 32, 64])
    }


trainer.hyperparameter_search(
    direction="maximize",
    hp_space=my_hp_space,
    # pruner=optuna.pruners.MedianPruner(
    #     n_startup_trials=2, n_warmup_steps=5, interval_steps=3
    # ),
)

# Defaut objective is the sum of all metrics when metrics are provided, so we have to maximize it.
# trainer.hyperparameter_search(direction="maximize")
