import ast
import pickle

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import json


class ConfigParser:
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
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.data["input_ids"])


class DataHelper:
    """
    A helper class for data loading and processing
    """

    def __init__(self, data_dir, mode="train", add_ent_token=False, aug_data_dir=""):
        self._data = pd.read_csv(data_dir)
        if aug_data_dir:
            self._aug_data = pd.read_csv(aug_data_dir)
            self._data = pd.concat([self._data, self._aug_data])
        self._mode = mode
        self.add_ent_token = add_ent_token
        self._preprocess()

    def _preprocess(self):
        data = self._data
        if self.add_ent_token:
            data = self.ent_preprocess(data)

        def extract(d):
            return ast.literal_eval(d)["word"]

        subjects = list(map(extract, data["subject_entity"]))
        objects = list(map(extract, data["object_entity"]))

        self._processed = pd.DataFrame(
            {
                "id": data["id"],
                "sentence": data["sentence"],
                "subject_entity": subjects,
                "object_entity": objects,
            }
        )
        if self._mode == "train":
            self._labels = self.convert_labels_by_dict(labels=data["label"])

    def split(self, ratio=0.2, n_splits=5, mode="plain", random_seed=42):
        if mode == "plain":
            idxs_list = [
                train_test_split(
                    np.arange(len(self._data)), test_size=ratio, shuffle=True
                )
            ]
        elif mode == "skf":
            skf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_seed
            )
            idxs_list = skf.split(self._processed, self._labels)
        return idxs_list

    def from_idxs(self, idxs=None):
        return (
            (self._processed.iloc[idxs], self._labels[idxs])
            if self._mode == "train"
            else self._processed
        )

    def tokenize(self, data, tokenizer):
        concated_entities = [
            sub + "[SEP]" + obj
            for sub, obj in zip(data["subject_entity"], data["object_entity"])
        ]
        if self.add_ent_token:
            tokenized = tokenizer(
                data["sentence"].tolist(), truncation=True, return_token_type_ids=False,
            )
        else:
            tokenized = tokenizer(
                concated_entities,
                data["sentence"].tolist(),
                truncation=True,
                return_token_type_ids=False,
            )
        return tokenized

    def convert_labels_by_dict(self, labels, dictionary="data/dict_label_to_num.pkl"):
        with open(dictionary, "rb") as f:
            dictionary = pickle.load(f)
        return np.array([dictionary[label] for label in labels])

    def ent_preprocess(self, data):
        data["sentence"] = data.apply(
            lambda row: self.add_entity_tokens(
                row["sentence"], row["object_entity"], row["subject_entity"]
            ),
            axis=1,
        )
        return data

    def add_entity_tokens(self, sentence, object_entity, subject_entity):
        def entity_mapper(entity_type):
            e_map = {
                "PER": "인물",
                "ORG": "기관",
                "LOC": "지명",
                "POH": "기타",
                "DAT": "날짜",
                "NOH": "수량",
            }
            return e_map[entity_type]

        def extract(entity):
            return (
                int(ast.literal_eval(entity)["start_idx"]),
                int(ast.literal_eval(entity)["end_idx"]),
                entity_mapper(ast.literal_eval(entity)["type"]),
            )

        obj_start_idx, obj_end_idx, obj_type = extract(object_entity)
        subj_start_idx, subj_end_idx, sbj_type = extract(subject_entity)

        if obj_start_idx < subj_start_idx:
            new_sentence = (
                sentence[:obj_start_idx]
                + "#"
                + "*"
                + obj_type
                + "*"
                + sentence[obj_start_idx: obj_end_idx + 1]
                + "#"
                + sentence[obj_end_idx + 1: subj_start_idx]
                + "@"
                + "^"
                + sbj_type
                + "^"
                + sentence[subj_start_idx: subj_end_idx + 1]
                + "@"
                + sentence[subj_end_idx + 1:]
            )
        else:
            new_sentence = (
                sentence[:subj_start_idx]
                + "@"
                + "^"
                + sbj_type
                + "^"
                + sentence[subj_start_idx: subj_end_idx + 1]
                + "@"
                + sentence[subj_end_idx + 1: obj_start_idx]
                + "#"
                + "*"
                + obj_type
                + "*"
                + sentence[obj_start_idx: obj_end_idx + 1]
                + "#"
                + sentence[obj_end_idx + 1:]
            )

        return new_sentence

    def tokenize_with_entity_mask(self, entity_data, tokenizer):
        """
        input sentence format : <e1> 이순신 </e1> 은 <e2> 조선 </e2> 중기의 무신이다.
        """
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
        )

        sub_start = tokenizer.vocab["<e1>"]
        sub_end = tokenizer.vocab["</e1>"]
        obj_start = tokenizer.vocab["<e2>"]
        obj_end = tokenizer.vocab["</e2>"]

        tokenized = tokenizer(
            entity_data["sentence"].tolist(),
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        tokenized_to_check_idx = tokenizer(
            entity_data["sentence"].tolist(),
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )
        e1_mask_list = []
        e2_mask_list = []
        for i in range(len(tokenized_to_check_idx["input_ids"])):
            token_len = len(tokenized_to_check_idx["input_ids"][i])
            e1_mask = [0] * token_len
            e2_mask = [0] * token_len
            sub_start_idx = tokenized_to_check_idx["input_ids"][i].index(
                sub_start)
            sub_end_idx = tokenized_to_check_idx["input_ids"][i].index(sub_end)
            obj_start_idx = tokenized_to_check_idx["input_ids"][i].index(
                obj_start)
            obj_end_idx = tokenized_to_check_idx["input_ids"][i].index(obj_end)

            for idx in range(sub_start_idx, sub_end_idx + 1):
                e1_mask[idx] = 1

            for idx in range(obj_start_idx, obj_end_idx + 1):
                e2_mask[idx] = 1

            e1_mask_list.append(e1_mask)
            e2_mask_list.append(e2_mask)

        tokenized["e1_mask"] = torch.LongTensor(e1_mask_list)
        tokenized["e2_mask"] = torch.LongTensor(e2_mask_list)

        return tokenized
