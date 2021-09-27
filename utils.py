import pickle
import re

import torch
from torch.utils.data import Dataset

import pandas as pd


class RelationExtractionDataset(Dataset):
    """
    A dataset class for loading Relation Extraction data
    """

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach()
            for key, val in self.pair_dataset.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class DataHelper:
    """
    A helper class for data loading and processing
    """

    def __init__(self, data_dir):
        self._raw = pd.read_csv(data_dir)

    def preprocess(self, data=None, mode='train'):
        if data is None:
            data = self._raw

        pattern = re.compile(pattern='[a-z가-힣0-9]+')
        extract = lambda data: pattern.findall(string=data)[1]

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
            labels = self.convert_by(labels=data['label'])
        elif mode == 'inference':
            labels = data['label']
            
        return preprocessed, labels

    def tokenize(self, data, tokenizer):
        concated_entities = [
            sub + '[SEP]' + obj for sub, obj in zip(data['subject_entity'], data['object_entity'])
        ]
        tokenized = tokenizer(
            concated_entities,
            data['sentence'].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized

    def convert_by(self, labels, dictionary='dict_label_to_num.pkl'):
        with open(dictionary, 'rb') as f:
            dictionary = pickle.load(f)
        return [dictionary[label] for label in labels]
