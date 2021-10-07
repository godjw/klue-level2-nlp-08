import torch
from torch import nn

from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class CombineModels(nn.Module):
    def __init__(self):
        super(CombineModels, self).__init__()

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

        concatenated_vectors = torch.cat((
            logits_1, logits_2, logits_3, logits_4, logits_5,
            logits_6, logits_7, logits_8, logits_9, logits_10,
            logits_11, logits_12, logits_13, logits_14, logits_15), dim=-1)

        output = self.classifier(concatenated_vectors)
        outputs = SequenceClassifierOutput(logits=output)
        return outputs
