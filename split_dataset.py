from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd


def label_to_num(label):
    num_label = []
    with open('data/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


pd_train_dataset = pd.read_csv('~/dataset/train/train.csv')
train_label = label_to_num(pd_train_dataset['label'].values)

train_idx_10, valid_idx_10 = train_test_split(
    np.arange(len(train_label)), test_size=0.1, random_state=42, shuffle=True, stratify=train_label)

print('train 10%: ' + str(len(train_idx_10)),
      'valid 10%: ' + str(len(valid_idx_10)))
t_split_10 = pd.DataFrame(pd_train_dataset, index=train_idx_10)
v_split_10 = pd.DataFrame(pd_train_dataset, index=valid_idx_10)

t_split_10.to_csv('data/train_10.csv')
v_split_10.to_csv('data/valid_10.csv')

train_idx_15, valid_idx_15 = train_test_split(
    np.arange(len(train_label)), test_size=0.15, random_state=42, shuffle=True, stratify=train_label)

print('train 15%: ' + str(len(train_idx_15)),
      'valid 15%: ' + str(len(valid_idx_15)))
t_split_15 = pd.DataFrame(pd_train_dataset, index=train_idx_15)
v_split_15 = pd.DataFrame(pd_train_dataset, index=valid_idx_15)

t_split_15.to_csv('data/train_15.csv')
v_split_15.to_csv('data/valid_15.csv')

train_idx_20, valid_idx_20 = train_test_split(
    np.arange(len(train_label)), test_size=0.2, random_state=42, shuffle=True, stratify=train_label)

print('train 20%: ' + str(len(train_idx_20)),
      'valid 20%: ' + str(len(valid_idx_20)))
t_split_20 = pd.DataFrame(pd_train_dataset, index=train_idx_20)
v_split_20 = pd.DataFrame(pd_train_dataset, index=valid_idx_20)

t_split_20.to_csv('data/train_20.csv')
v_split_20.to_csv('data/valid_20.csv')


train_label_small = label_to_num(t_split_10['label'].values)
valid_label_small = label_to_num(v_split_10['label'].values)

_, train_idx_small = train_test_split(
    np.arange(len(train_label_small)), test_size=0.05, random_state=42, shuffle=True, stratify=train_label_small)
_, valid_idx_small = train_test_split(
    np.arange(len(valid_label_small)), test_size=0.05, random_state=42, shuffle=True, stratify=valid_label_small)

t_split_small = pd.DataFrame(pd_train_dataset, index=train_idx_small)
t_split_small.to_csv('data/train_small.csv')

print('train small: ' + str(len(train_idx_small)),
      'valid small: ' + str(len(valid_idx_small)))
v_split_small = pd.DataFrame(pd_train_dataset, index=valid_idx_small)
v_split_small.to_csv('data/valid_small.csv')
