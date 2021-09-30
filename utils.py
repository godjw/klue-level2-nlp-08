import ast
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold
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

    def __init__(self, data_dir, mode='train'):
        self._data = pd.read_csv(data_dir)
        self._mode = mode
        self._preprocess()
    
    def _preprocess(self):
        data = self._data
        extract = lambda d: ast.literal_eval(d)['word']

        subjects = list(map(extract, data['subject_entity']))
        objects = list(map(extract, data['object_entity']))

        self._processed = pd.DataFrame({
            'id': data['id'],
            'sentence': data['sentence'],
            'subject_entity': subjects,
            'object_entity': objects,
        })
        if self._mode == 'train':
            self._labels = self.convert_labels_by_dict(labels=data['label'])

    def split(self, ratio=0.2, n_splits=5, mode='plain'):
        if mode == 'plain':
            idxs_list = [train_test_split(
                np.arange(len(self._data)),
                test_size=ratio,
                shuffle=True
            )]
        elif mode == 'skf':
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            idxs_list = skf.split(self._processed, self._labels)
        return idxs_list

    def from_idxs(self, idxs=None):
        return (self._processed.iloc[idxs], self._labels[idxs]) if self._mode == 'train' else self._processed

    def tokenize(self, data, tokenizer):
        concated_entities = [sub + '[SEP]' + obj for sub, obj in zip(data['subject_entity'], data['object_entity'])]
        tokenized = tokenizer(
            concated_entities,
            data['sentence'].tolist(),
            truncation=True,
            return_token_type_ids=False,
        )
        return tokenized

    def convert_labels_by_dict(self, labels, dictionary='data/dict_label_to_num.pkl'):
        with open(dictionary, 'rb') as f:
            dictionary = pickle.load(f)
        return np.array([dictionary[label] for label in labels])
