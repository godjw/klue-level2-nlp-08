import ast
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class RelationExtractionDataset(Dataset):
    """
    A dataset class for loading Relation Extraction data
    """

    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.data.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.data['input_ids'])


class DataHelper:
    """
    A helper class for data loading and processing
    """

    def __init__(self, data_dir, dictionary_dir):
        self._raw = pd.read_csv(data_dir)
        self.dictionary_dir = dictionary_dir

    def preprocess(self, data=None, mode='train'):
        if data is None:
            data = self._raw

        def extract(data): return ast.literal_eval(data)['word']

        subjects = list(map(extract, data['subject_entity']))
        objects = list(map(extract, data['object_entity']))

        preprocessed = pd.DataFrame({
            'id': data['id'],
            'sentence': data['sentence'],
            'subject_entity': subjects,
            'object_entity': objects,
            'label': data['label']
        })
        if mode == 'train':
            labels = self.convert_labels_by_dict(labels=data['label'])
        elif mode == 'inference':
            labels = data['label']

        return preprocessed, labels

    def tokenize(self, data, tokenizer):
        concated_entities = [
            sub + obj for sub, obj in zip(data['subject_entity'], data['object_entity'])
        ]
        tokenized = tokenizer(
            # concated_entities,
            data['sentence'].tolist(),
            truncation=True,
            # return_token_type_ids=False,
        )
        return tokenized

    def roberta_tokenize(self, data, tokenizer):
        concated_entities = [
            sub + obj for sub, obj in zip(data['subject_entity'], data['object_entity'])
        ]
        tokenized = tokenizer(
            # concated_entities,
            data['sentence'].tolist(),
            truncation=True,
            return_token_type_ids=False,
        )
        return tokenized

    def convert_labels_by_dict(self, labels):
        with open(self.dictionary_dir, 'rb') as f:
            dictionary = pickle.load(f)
        return [dictionary[label] for label in labels]


class TestDataset(Dataset):
    """
    A dataset class for loading Relation Extraction data
    """

    def __init__(self, pair_dataset, labels):
        self.labels = labels
        self.data_inven = self.get_data(pair_dataset)

    def get_data(self, pair_dataset):
        pd_pair_dataset = pd.DataFrame()

        for key, val in pair_dataset.items():
            pd_pair_dataset[key] = val

        pd_pair_dataset['labels'] = torch.tensor(self.labels)

        temp_df = pd.DataFrame(pd_pair_dataset)
        temp_df.reset_index(inplace=True, drop=True)

        return temp_df

    def __getitem__(self, idx):
        return dict(self.data_inven.iloc[idx])

    def __len__(self):
        return len(self.data_inven)
