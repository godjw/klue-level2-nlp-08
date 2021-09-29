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

    def __init__(self, pair_dataset):
        self.pair_dataset = pair_dataset

    def __getitem__(self, idx):
        return dict(self.pair_dataset.iloc[idx])

    def __len__(self):
        return len(self.pair_dataset)


class DataHelper:
    """
    A helper class for data loading and processing
    """

    def __init__(self, data_dir):
        self._raw = pd.read_csv(data_dir)

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
        return [dictionary[label] for label in labels]

    def split(self, pair_data, labels, phase, split_ratio=0.2, small=False):
        # stratified split if inference!
        if phase == 'inference':
            pd_pair_data = pd.DataFrame()
            for key, val in pair_data.items():
                pd_pair_data[key] = val
            return pd_pair_data

        # stratified split during training/validating!
        train_idx, valid_idx = train_test_split(np.arange(len(
            labels)), test_size=split_ratio, random_state=42, shuffle=True, stratify=labels)

        if small == True:
            train_idx = train_idx[:int(len(train_idx)/20)]
            valid_idx = valid_idx[:int(len(valid_idx)/20)]

        pd_pair_data = pd.DataFrame()
        for key, val in pair_data.items():
            pd_pair_data[key] = val

        pd_pair_data['labels'] = torch.tensor(labels)

        if phase == 'train':
            index = train_idx
        elif phase == 'validation':
            index = valid_idx
        
        temp_df = pd.DataFrame(pd_pair_data, index=index)
        temp_df.reset_index(inplace=True, drop=True)

        return temp_df
            
            

