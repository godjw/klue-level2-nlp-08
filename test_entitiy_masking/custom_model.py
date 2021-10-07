import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, RobertaPreTrainedModel
from loss_func import *


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
    def __init__(self, config, model_name):
        super(RBERT, self).__init__(config)
        self.bert = RobertaModel.from_pretrained(
            model_name)  # Load pretrained bert

        # for param in self.bert.parameters():
        #     param.requires_grad = False

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
        # print(input_ids)
        # print(attention_mask)
        # print(e1_mask)
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                            )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # print(outputs)
        # print(outputs.shape())
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

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
        # print(outputs)

        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CB_loss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))

        #     outputs = (loss,) + outputs

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

                # loss_fct = CB_loss()
                # loss = loss_fct(
                #     outputs[0], labels)

            outputs = (loss,) + outputs
        # outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        print(input.size())
        print(target.size())
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
