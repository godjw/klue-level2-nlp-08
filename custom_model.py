import torch
from torch import nn
from packaging import version

class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    Copied from https://huggingface.co/transformers/_modules/transformers/models/roberta/modeling_roberta.html#RobertaModel
    nice
    """
    
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, model, config, pre_model_state_dict=None):
        super().__init__()
        self.word_embeddings = model.roberta.embeddings.word_embeddings
        self.position_embeddings = model.roberta.embeddings.position_embeddings
        self.token_type_embeddings = model.roberta.embeddings.token_type_embeddings
        self.entity_embeddings = nn.Embedding(3, config.hidden_size, padding_idx=0)
        # added by jinseong, entity embedding layer
        if pre_model_state_dict:
            pre_weight = pre_model_state_dict['roberta.embeddings.entity_embeddings.weight']
            self.entity_embeddings.weight = torch.nn.parameter.Parameter(pre_weight, requires_grad=True)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = model.roberta.embeddings.LayerNorm
        self.dropout = model.roberta.embeddings.dropout
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

        # End copy
        self.padding_idx = config.pad_token_id
        #self.position_embeddings = nn.Embedding(
        #    config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        #)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # added entity embeddings code
        entity_ids = self.create_entity_ids_from_input_ids(input_ids)
        entity_embeddings = self.entity_embeddings(entity_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # add embeddings
        embeddings += entity_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def create_entity_ids_from_input_ids(self, input_ids):
        s_ids = torch.nonzero((input_ids == 36)) # subject
        o_ids = torch.nonzero((input_ids == 7)) # object

        #type_map = {4410 : 3, 7119 : 4, 3860 : 5, 5867 : 6, 12395 : 7, 9384 : 8}

        entity_ids = torch.zeros_like(input_ids)
        for i in range(len(s_ids)):
            s_id = s_ids[i]
            o_id = o_ids[i]
            if i % 2 == 0:
                continue # when you only embed sbj and obj
                #entity_ids[s_id[0], s_id[1]+2] = type_map[input_ids[s_id[0], s_id[1]+2].item()]
                #entity_ids[o_id[0], o_id[1]+2] = type_map[input_ids[o_id[0], o_id[1]+2].item()]
            else:
                prev_s_id = s_ids[i-1]
                prev_o_id = o_ids[i-1]
                entity_ids[s_id[0], prev_s_id[1]+4:s_id[1]] = 1
                entity_ids[o_id[0], prev_o_id[1]+4:o_id[1]] = 2

        return entity_ids

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx