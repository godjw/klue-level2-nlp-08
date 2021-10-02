import ast
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import json


class ConfigParser():
    def __init__(self, config):
        self.config = self.json_to_dict(config)

    def json_to_dict(self, config):
        with open(config) as json_config:
            return json.load(json_config)


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

<<<<<<< HEAD
    def __init__(self, data_dir, dictionary_dir):
        self._raw = pd.read_csv(data_dir)
        self.dictionary_dir = dictionary_dir
=======
    def __init__(self, data_dir, mode='train', add_ent_token=False):
        self._data = pd.read_csv(data_dir)
        self._mode = mode
        self.add_ent_token = add_ent_token
        self._preprocess()
>>>>>>> ff38c6721a5eef9af8a87e3a0ddb85bad4064a1e

    def _preprocess(self):
        data = self._data
        if self.add_ent_token:
            data = self.ent_preprocess(data)

        def extract(d): return ast.literal_eval(d)['word']

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

    def split(self, ratio=0.2, n_splits=5, mode='plain', random_seed=42):
        if mode == 'plain':
            idxs_list = [train_test_split(
                np.arange(len(self._data)),
                test_size=ratio,
                shuffle=True
            )]
        elif mode == 'skf':
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed)
            idxs_list = skf.split(self._processed, self._labels)
        return idxs_list

    def from_idxs(self, idxs=None):
        return (self._processed.iloc[idxs], self._labels[idxs]) if self._mode == 'train' else self._processed

    def tokenize(self, data, tokenizer):
<<<<<<< HEAD
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
=======
        concated_entities = [sub + '[SEP]' + obj for sub,
                             obj in zip(data['subject_entity'], data['object_entity'])]
        if self.add_ent_token:
            tokenized = tokenizer(
                data['sentence'].tolist(),
                truncation=True,
                return_token_type_ids=False,
            )
        else:
            tokenized = tokenizer(
                concated_entities,
                data['sentence'].tolist(),
                truncation=True,
                return_token_type_ids=False,
            )
>>>>>>> ff38c6721a5eef9af8a87e3a0ddb85bad4064a1e
        return tokenized

    def convert_labels_by_dict(self, labels):
        with open(self.dictionary_dir, 'rb') as f:
            dictionary = pickle.load(f)
<<<<<<< HEAD
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
=======
        return np.array([dictionary[label] for label in labels])

    def ent_preprocess(self, data):
        data['sentence'] = data.apply(lambda row: self.add_entity_tokens(
            row['sentence'], row['object_entity'], row['subject_entity']), axis=1)
        return data

    def add_entity_tokens(self, sentence, object_entity, subject_entity):

        def entity_mapper(entity_type):
            e_map = {'PER': '인물', 'ORG': '기관', 'LOC': '지명',
                     'POH': '기타', 'DAT': '날짜', 'NOH': '수량'}
            return e_map[entity_type]

        def extract(entity):
            return int(ast.literal_eval(entity)['start_idx']), int(ast.literal_eval(entity)['end_idx']), entity_mapper(ast.literal_eval(entity)['type'])

        obj_start_idx, obj_end_idx, obj_type = extract(object_entity)
        subj_start_idx, subj_end_idx, sbj_type = extract(subject_entity)

        if obj_start_idx < subj_start_idx:
            new_sentence = sentence[:obj_start_idx] + '#' + '*' + obj_type + '*' + sentence[obj_start_idx:obj_end_idx + 1] + '#' + \
                sentence[obj_end_idx + 1:subj_start_idx] + '@' + '^' + sbj_type + '^' + sentence[subj_start_idx:subj_end_idx + 1] + \
                '@' + sentence[subj_end_idx + 1:]
        else:
            new_sentence = sentence[:subj_start_idx] + '@' + '^' + sbj_type + '^' + sentence[subj_start_idx:subj_end_idx + 1] + '@' + \
                sentence[subj_end_idx + 1:obj_start_idx] + '#' + '*' + obj_type + '*' + sentence[obj_start_idx:obj_end_idx + 1] + \
                '#' + sentence[obj_end_idx + 1:]

        return new_sentence
>>>>>>> ff38c6721a5eef9af8a87e3a0ddb85bad4064a1e
