import torch
from torch import nn
from packaging import version

class RobertaEmbeddingsWithTokenEmbedding(nn.Module):
    """
    Roberta Embedding Module with Entity and Entity Type Embedding layer added
    can load trained weights of the Entity Type Embedding Layer from the state_dict
    """
    
    def __init__(self, model, config, pre_model_state_dict=None):
        super().__init__()
        self.word_embeddings = model.roberta.embeddings.word_embeddings
        self.position_embeddings = model.roberta.embeddings.position_embeddings
        self.token_type_embeddings = model.roberta.embeddings.token_type_embeddings
        
        # add entity embedding Layer
        # 0 for the words that are not neither entity nor entity type
        # 1, 2 for subject and object entity
        # 3 ~ 8 for the entity type which are annotated by NER tagger
        self.entity_embeddings = nn.Embedding(9, config.hidden_size, padding_idx=0)

        # load weights of the layer in order not to use randomly initialized weights
        if pre_model_state_dict:
            pre_weight = pre_model_state_dict['roberta.embeddings.entity_embeddings.weight']
            self.entity_embeddings.weight = torch.nn.parameter.Parameter(pre_weight, requires_grad=True)

        self.LayerNorm = model.roberta.embeddings.LayerNorm
        self.dropout = model.roberta.embeddings.dropout
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )
        self.padding_idx = config.pad_token_id

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

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

        # get the positional index of entity and its type,
        # then embed entity information using the index
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
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def create_entity_ids_from_input_ids(self, input_ids):
        """
        map index 1~8 to the token that is related to sbj, obj entities
        """
        s_ids = torch.nonzero((input_ids == 36)) # subject
        o_ids = torch.nonzero((input_ids == 7)) # object
        # entity type mapped into index 3 ~ 8
        type_map = {4410 : 3, 7119 : 4, 3860 : 5, 5867 : 6, 12395 : 7, 9384 : 8}

        entity_ids = torch.zeros_like(input_ids)
        for i in range(len(s_ids)):
            s_id = s_ids[i]
            o_id = o_ids[i]
            if i % 2 == 0:
                entity_ids[s_id[0], s_id[1]+2] = type_map[input_ids[s_id[0], s_id[1]+2].item()]
                entity_ids[o_id[0], o_id[1]+2] = type_map[input_ids[o_id[0], o_id[1]+2].item()]
            else:
                prev_s_id = s_ids[i-1]
                prev_o_id = o_ids[i-1]
                entity_ids[s_id[0], prev_s_id[1]+4:s_id[1]] = 1
                entity_ids[o_id[0], prev_o_id[1]+4:o_id[1]] = 2

        return entity_ids

    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx