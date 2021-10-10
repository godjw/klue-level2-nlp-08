import torch
from torch import nn

from transformers import AutoConfig, AutoModelForSequenceClassification, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from loss import CB_loss


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


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(RobertaPreTrainedModel):
    '''
    orgin code: https://github.com/monologg/R-BERT
    edit by 문하겸_T2076
    '''

    def __init__(self, config, model_name):
        super(RBERT, self).__init__(config)
        self.roberta = RobertaModel.from_pretrained(
            model_name)  # Load pretrained bert

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, 0.1)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(
            dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(),
                               hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, e1_mask=None, e2_mask=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_type = "focal"
                beta = 0.9999
                gamma = 2.0

                loss_fct = CB_loss(beta=beta, gamma=gamma)
                loss = loss_fct(logits.view(-1, self.num_labels),
                                labels.view(-1), loss_type)

            outputs = (loss,) + outputs
        return outputs
