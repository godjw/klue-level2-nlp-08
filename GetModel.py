from transformers import RobertaModel
import torch.nn as nn
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

#class GetModel(nn.Module): # model save, load(in checkpoint)
class GetModel(RobertaPreTrainedModel):
      def __init__(self, config, *args, **kwargs):
            # super(GetModel, self).__init__()
            super().__init__(config=config)

            self.bert = RobertaModel.from_pretrained("klue/roberta-large")
            ### New layers:
            self.lstm = nn.LSTM(1024, 256, batch_first=True, bidirectional=True)
            self.linear = nn.Linear(256*2, 30)
            self.dropout = nn.Dropout(0.5)
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(30, 1)

      def forward(self, input_ids, attention_mask):
            output = self.bert(input_ids, attention_mask=attention_mask)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
            lstm_output, (h,c) = self.lstm(output[0]) ## extract the 1st token's embeddings
            hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
            linear_output = self.linear(hidden.view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
            x = self.tanh(linear_output)
            x = self.dropout(x)
            outputs = SequenceClassifierOutput(logits=x)

            return outputs