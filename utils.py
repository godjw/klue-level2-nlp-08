import ast
import pickle
import os
import random

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
        to_dict = lambda d: ast.literal_eval(d)

        subjects = list(map(to_dict, data['subject_entity']))
        objects = list(map(to_dict, data['object_entity']))

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
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            idxs_list = skf.split(self._processed, self._labels)
        return idxs_list

    def from_idxs(self, idxs=None):
        return (self._processed.iloc[idxs], self._labels[idxs]) if self._mode == 'train' else self._processed

    def tokenize(self, data, tokenizer):
        tokenized = tokenizer(
            [
                self._emphasize_entities(sent=sent, sub_info=sub_info, obj_info=obj_info)
                for sent, sub_info, obj_info in zip(data['sentence'], data['subject_entity'], data['object_entity'])
            ],
            truncation=True,
            return_token_type_ids=False
        )
        return tokenized

    def _emphasize_entities(self, sent, sub_info, obj_info):
        entities = {'PER': '인물',  'ORG': '기관', 'LOC': '지명', 'POH': '기타', 'DAT': '날짜', 'NOH': '수량'}
        sub_s, sub_e, sub_type = sub_info['start_idx'], sub_info['end_idx'] + 1, sub_info['type']
        obj_s, obj_e, obj_type = obj_info['start_idx'], obj_info['end_idx'] + 1, obj_info['type']
        if sub_s < obj_s:
            sent = sent[:sub_s] + '#+' + entities[sub_type] + '+' + sent[sub_s:sub_e] + '#' + sent[sub_e:obj_s] + '@^' + entities[obj_type] + '^' + sent[obj_s:obj_e] + '@' + sent[obj_e:]
        else:
            sent = sent[:obj_s] + '@^' + entities[obj_type] + '^' + sent[obj_s:obj_e] + '@' + sent[obj_e:sub_s] + '#+' + entities[sub_type] + '+' + sent[sub_s:sub_e] + '#' + sent[sub_e:]
        return sent

    def convert_labels_by_dict(self, labels, dictionary='data/dict_label_to_num.pkl'):
        with open(dictionary, 'rb') as f:
            dictionary = pickle.load(f)
        return np.array([dictionary[label] for label in labels])


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
