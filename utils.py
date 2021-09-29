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

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.data.items()}
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

    def preprocess(self, data=None, mode='train', test_size=0.2):
        if data is None:
            data = self._raw

        extract = lambda data: ast.literal_eval(data)['word']

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
            train_idxs, val_idxs = train_test_split(
                np.arange(len(labels)),
                test_size=test_size,
                shuffle=True
            )
            preprocessed = {
                'train_data': preprocessed.iloc[train_idxs],
                'train_labels': labels[train_idxs],
                'val_data': preprocessed.iloc[val_idxs],
                'val_labels':labels[val_idxs]
            }
        elif mode == 'inference':
            labels = np.array(data['label'])
            preprocessed = {
                'test_data': preprocessed,
                'test_labels': labels
            }

        return preprocessed

    def tokenize(self, data, tokenizer):
        concated_entities = [
            sub + '[SEP]' + obj for sub, obj in zip(data['subject_entity'], data['object_entity'])
        ]
        tokenized = tokenizer(
            concated_entities,
            data['sentence'].tolist(),
            truncation=True,
        )
        return tokenized

    def convert_labels_by_dict(self, labels, dictionary='data/dict_label_to_num.pkl'):
        with open(dictionary, 'rb') as f:
            dictionary = pickle.load(f)
        return np.array([dictionary[label] for label in labels])
