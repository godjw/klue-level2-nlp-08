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
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import wandb

from utils import RelationExtractionDataset, DataHelper, ConfigParser
from metric import compute_metrics
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


from utils import *


def infer(model, test_dataset, batch_size, collate_fn, device):
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    preds, probs = [], []
    model.eval()
    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            )
        logits = outputs.logits
        result = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)

        preds.append(result)
        probs.append(prob)

    return torch.cat(preds).tolist(), torch.cat(probs, dim=0).tolist()


class SplitModelsTest(RobertaPreTrainedModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=29)
        self.roberta1 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/0_fold", config=c1)
        self.roberta2 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/0_fold", config=c2)
        for p in self.roberta1.parameters():
            p.requires_grad = False
        for p in self.roberta2.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(2, 768)
        self.fc2 = nn.Linear(29, 768)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768 * 2, 768, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, 30, bias=True)
        )

    def forward(self, input_ids, attention_mask):
        logits_a = self.roberta1(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_b = self.roberta2(
            input_ids.clone(), attention_mask=attention_mask).get('logits')

        logits_a = self.fc1(logits_a)
        logits_b = self.fc2(logits_b)

        concatenated_vectors = torch.cat(
            (logits_a, logits_b), dim=-1)
        output = self.classifier(concatenated_vectors)
        outputs = SequenceClassifierOutput(logits=output)
        return outputs


class SplitModels(nn.Module):
    def __init__(self):
        super(SplitModels, self).__init__()

        # c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        # c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=29)
        # c3 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # self.roberta1 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/0_fold", config=c1)
        # self.roberta2 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/0_fold", config=c2)
        # self.roberta3 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/0_fold", config=c3)
        # for p in self.roberta1.parameters():
        #     p.requires_grad = False
        # for p in self.roberta2.parameters():
        #     p.requires_grad = False
        # for p in self.roberta3.parameters():
        #     p.requires_grad = False
        # self.fc1 = nn.Linear(2, 768)
        # self.fc2 = nn.Linear(29, 768)
        # self.fc3 = nn.Linear(30, 768)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768 * 3, 768, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768, 30, bias=True)
        # )

        # NOTE: roberta-large, small, base, bert-base
        # self.roberta3 = AutoModelForSequenceClassification.from_pretrained(
        #     "klue/roberta-large", config=c3)
        # self.roberta4 = AutoModelForSequenceClassification.from_pretrained(
        #     "klue/roberta-small", config=c4)
        # self.roberta5 = AutoModelForSequenceClassification.from_pretrained(
        #     "klue/roberta-base", config=c5)
        # self.bert1 = AutoModelForSequenceClassification.from_pretrained(
        #     "klue/bert-base", config=c6)
        # c3 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # c4 = AutoConfig.from_pretrained('klue/roberta-small', num_labels=30)
        # c5 = AutoConfig.from_pretrained('klue/roberta-base', num_labels=30)
        # c6 = AutoConfig.from_pretrained('klue/bert-base', num_labels=30)
        # for p in self.roberta3.parameters():
        #     p.requires_grad = False
        # for p in self.roberta4.parameters():
        #     p.requires_grad = False
        # for p in self.roberta5.parameters():
        #     p.requires_grad = False
        # for p in self.bert1.parameters():
        #     p.requires_grad = False
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(30 * 4, 768, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768, 30, bias=True)
        # )

        # NOTE: roberta-large kfold models
        # c3 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # c4 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # c5 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # c6 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # c7 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        # self.roberta3 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/0_fold", config=c3)
        # self.roberta4 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/1_fold", config=c4)
        # self.roberta5 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/2_fold", config=c5)
        # self.roberta6 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/3_fold", config=c6)
        # self.roberta7 = AutoModelForSequenceClassification.from_pretrained(
        #     "sota_focal_loss_kfold_model/4_fold", config=c7)
        # for p in self.roberta3.parameters():
        #     p.requires_grad = False
        # for p in self.roberta4.parameters():
        #     p.requires_grad = False
        # for p in self.roberta5.parameters():
        #     p.requires_grad = False
        # for p in self.roberta6.parameters():
        #     p.requires_grad = False
        # for p in self.roberta7.parameters():
        #     p.requires_grad = False
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(30 * 5, 768, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768, 30, bias=True)
        # )

        # NOTE: norel-rel ensemble
        # c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        # c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=29)
        # self.roberta1 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/0_fold", config=c1)
        # self.roberta2 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/0_fold", config=c2)
        # self.roberta3 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/1_fold", config=c1)
        # self.roberta4 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/1_fold", config=c2)
        # self.roberta5 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/2_fold", config=c1)
        # self.roberta6 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/2_fold", config=c2)
        # self.roberta7 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/3_fold", config=c1)
        # self.roberta8 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/3_fold", config=c2)
        # self.roberta9 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_no_rel_large/4_fold", config=c1)
        # self.roberta10 = AutoModelForSequenceClassification.from_pretrained(
        #     "split_model_rel_large/4_fold", config=c2)
        # for p in self.roberta1.parameters():
        #     p.requires_grad = False
        # for p in self.roberta2.parameters():
        #     p.requires_grad = False
        # for p in self.roberta3.parameters():
        #     p.requires_grad = False
        # for p in self.roberta4.parameters():
        #     p.requires_grad = False
        # for p in self.roberta5.parameters():
        #     p.requires_grad = False
        # for p in self.roberta6.parameters():
        #     p.requires_grad = False
        # for p in self.roberta7.parameters():
        #     p.requires_grad = False
        # for p in self.roberta8.parameters():
        #     p.requires_grad = False
        # for p in self.roberta9.parameters():
        #     p.requires_grad = False
        # for p in self.roberta10.parameters():
        #     p.requires_grad = False
        # self.fc1 = nn.Linear(2, 768)
        # self.fc2 = nn.Linear(29, 768)

        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768 * 10, 768, bias=True),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(768, 30, bias=True)
        # )

        # NOTE: rel add sota fold ensemble
        c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=29)
        c3 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)
        self.roberta1 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/0_fold", config=c1)
        self.roberta2 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/0_fold", config=c2)
        self.roberta3 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/1_fold", config=c1)
        self.roberta4 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/1_fold", config=c2)
        self.roberta5 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/2_fold", config=c1)
        self.roberta6 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/2_fold", config=c2)
        self.roberta7 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/3_fold", config=c1)
        self.roberta8 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/3_fold", config=c2)
        self.roberta9 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_no_rel_large/4_fold", config=c1)
        self.roberta10 = AutoModelForSequenceClassification.from_pretrained(
            "split_model_rel_large/4_fold", config=c2)
        self.roberta11 = AutoModelForSequenceClassification.from_pretrained(
            "sota_focal_loss_kfold_model/0_fold", config=c3)
        self.roberta12 = AutoModelForSequenceClassification.from_pretrained(
            "sota_focal_loss_kfold_model/1_fold", config=c3)
        self.roberta13 = AutoModelForSequenceClassification.from_pretrained(
            "sota_focal_loss_kfold_model/2_fold", config=c3)
        self.roberta14 = AutoModelForSequenceClassification.from_pretrained(
            "sota_focal_loss_kfold_model/3_fold", config=c3)
        self.roberta15 = AutoModelForSequenceClassification.from_pretrained(
            "sota_focal_loss_kfold_model/4_fold", config=c3)
        for p in self.roberta1.parameters():
            p.requires_grad = False
        for p in self.roberta2.parameters():
            p.requires_grad = False
        for p in self.roberta3.parameters():
            p.requires_grad = False
        for p in self.roberta4.parameters():
            p.requires_grad = False
        for p in self.roberta5.parameters():
            p.requires_grad = False
        for p in self.roberta6.parameters():
            p.requires_grad = False
        for p in self.roberta7.parameters():
            p.requires_grad = False
        for p in self.roberta8.parameters():
            p.requires_grad = False
        for p in self.roberta9.parameters():
            p.requires_grad = False
        for p in self.roberta10.parameters():
            p.requires_grad = False
        for p in self.roberta11.parameters():
            p.requires_grad = False
        for p in self.roberta12.parameters():
            p.requires_grad = False
        for p in self.roberta13.parameters():
            p.requires_grad = False
        for p in self.roberta14.parameters():
            p.requires_grad = False
        for p in self.roberta15.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(2, 768)
        self.fc2 = nn.Linear(29, 768)
        self.fc3 = nn.Linear(30, 768)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(768 * 15, 768, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(768, 30, bias=True)
        )

    def forward(self, input_ids, attention_mask):
        # logits_a = self.roberta1(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_b = self.roberta2(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_c = self.roberta3(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')

        # logits_a = self.fc1(logits_a)
        # logits_b = self.fc2(logits_b)
        # logits_c = self.fc3(logits_c)

        # concatenated_vectors = torch.cat(
        #     (logits_a, logits_b, logits_c), dim=-1)

        # NOTE: roberta-large, small, base, bert-base
        # logits_c = self.roberta3(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_d = self.roberta4(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_e = self.roberta5(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_f = self.bert1(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # concatenated_vectors = torch.cat(
        #     (logits_c, logits_d, logits_e, logits_f), dim=-1)

        # NOTE: roberta-large kfold models
        # logits_c = self.roberta3(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_d = self.roberta4(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_e = self.roberta5(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_f = self.roberta6(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_g = self.roberta7(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # concatenated_vectors = torch.cat(
        #     (logits_c, logits_d, logits_e, logits_f, logits_g), dim=-1)

        # NOTE: norel-rel ensemble
        # logits_1 = self.roberta1(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_2 = self.roberta2(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_3 = self.roberta3(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_4 = self.roberta4(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_5 = self.roberta5(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_6 = self.roberta6(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_7 = self.roberta7(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_8 = self.roberta8(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_9 = self.roberta9(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')
        # logits_10 = self.roberta10(
        #     input_ids.clone(), attention_mask=attention_mask).get('logits')

        # logits_1 = self.fc1(logits_1)
        # logits_2 = self.fc2(logits_2)
        # logits_3 = self.fc1(logits_3)
        # logits_4 = self.fc2(logits_4)
        # logits_5 = self.fc1(logits_5)
        # logits_6 = self.fc2(logits_6)
        # logits_7 = self.fc1(logits_7)
        # logits_8 = self.fc2(logits_8)
        # logits_9 = self.fc1(logits_9)
        # logits_10 = self.fc2(logits_10)

        # concatenated_vectors = torch.cat(
        #     (logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, logits_9, logits_10), dim=-1)

        # NOTE: rel add sota fold ensemble
        logits_1 = self.roberta1(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_2 = self.roberta2(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_3 = self.roberta3(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_4 = self.roberta4(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_5 = self.roberta5(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_6 = self.roberta6(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_7 = self.roberta7(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_8 = self.roberta8(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_9 = self.roberta9(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_10 = self.roberta10(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_11 = self.roberta11(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_12 = self.roberta12(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_13 = self.roberta13(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_14 = self.roberta14(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_15 = self.roberta15(
            input_ids.clone(), attention_mask=attention_mask).get('logits')

        logits_1 = self.fc1(logits_1)
        logits_2 = self.fc2(logits_2)
        logits_3 = self.fc1(logits_3)
        logits_4 = self.fc2(logits_4)
        logits_5 = self.fc1(logits_5)
        logits_6 = self.fc2(logits_6)
        logits_7 = self.fc1(logits_7)
        logits_8 = self.fc2(logits_8)
        logits_9 = self.fc1(logits_9)
        logits_10 = self.fc2(logits_10)
        logits_11 = self.fc3(logits_11)
        logits_12 = self.fc3(logits_12)
        logits_13 = self.fc3(logits_13)
        logits_14 = self.fc3(logits_14)
        logits_15 = self.fc3(logits_15)

        concatenated_vectors = torch.cat(
            (logits_1, logits_2, logits_3, logits_4, logits_5, logits_6, logits_7, logits_8, logits_9, logits_10, logits_11, logits_12, logits_13, logits_14, logits_15), dim=-1)

        output = self.classifier(concatenated_vectors)
        outputs = SequenceClassifierOutput(logits=output)
        return outputs


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    helper = DataHelper(data_dir=args.data_dir,
                        mode='inference', add_ent_token=args.add_ent_token)
    _test_data = helper.from_idxs()
    test_data = helper.tokenize(data=_test_data, tokenizer=tokenizer)
    test_dataset = RelationExtractionDataset(test_data)

    probs = []
    for k in range(args.n_splits if args.mode == 'skf' else 1):
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     path.join(args.model_dir,
        #               f'{k}_fold' if args.mode == 'skf' else args.mode)
        # )
        model = SplitModels()
        model.load_state_dict(torch.load(
            'rel_add_sota_fold/checkpoint-684/pytorch_model.bin'))
        model.to(device)

        pred_labels, pred_probs = infer(
            model=model,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            device=device
        )
        pred_labels = helper.convert_labels_by_dict(
            labels=pred_labels,
            dictionary=args.dictionary
        )
        probs.append(pred_probs)

        output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': pred_labels,
            'probs': pred_probs
        })
        output.to_csv(
            path.join(args.output_dir, (f'{k}_fold' if args.mode ==
                      'skf' else args.mode) + '_submission.csv'),
            index=False
        )

    if args.mode == 'skf':
        probs = torch.tensor(probs).mean(dim=0)
        preds = torch.argmax(probs, dim=-1).tolist()
        preds = helper.convert_labels_by_dict(
            labels=preds,
            dictionary=args.dictionary
        )
        output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': preds,
            'probs': probs.tolist()
        })
        output.to_csv(path.join(args.output_dir,
                      f'{args.n_splits}_folds_submission.csv'), index=False)

    print('Inference done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/test_data.csv')
    parser.add_argument('--dictionary', type=str,
                        default='data/dict_num_to_label.pkl')
    parser.add_argument('--output_dir', type=str,
                        default='./rel_add_sota_fold')

    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--add_ent_token', type=bool, default=True)

    args = parser.parse_args()
    print(args)

    inference(args=args)
