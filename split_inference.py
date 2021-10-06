import argparse
from os import path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import pandas as pd
from tqdm import tqdm

from split_utils import *


def infer(model, test_dataset, batch_size, collate_fn, device):
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    preds, probs = [], []
    model.eval()
    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            )
        logits = outputs[0]
        result = torch.argmax(logits, dim=-1)
        prob = F.softmax(logits, dim=-1)

        preds.append(result)
        probs.append(prob)

    return torch.cat(preds).tolist(), torch.cat(probs, dim=0).tolist()


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    helper = DataHelper(data_dir=args.data_dir,
                        mode='inference', add_ent_token=args.add_ent_token)
    _test_data = helper.from_idxs()
    test_data = helper.tokenize(data=_test_data, tokenizer=tokenizer)
    test_dataset = RelationExtractionDataset(test_data)

    no_rel_probs = []
    rel_probs = []
    total_probs = []
    for k in range(args.n_splits if args.mode == 'skf' else 1):
        no_rel_model = AutoModelForSequenceClassification.from_pretrained(
            path.join(args.no_rel_model_dir,
                      f'{k}_fold' if args.mode == 'skf' else args.mode)
        )
        no_rel_model.to(device)
        rel_model = AutoModelForSequenceClassification.from_pretrained(
            path.join(args.rel_model_dir,
                      f'{k}_fold' if args.mode == 'skf' else args.mode)
        )
        rel_model.to(device)

        no_rel_pred_labels, no_rel_pred_probs = infer(
            model=no_rel_model,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            device=device
        )
        rel_pred_labels, rel_pred_probs = infer(
            model=rel_model,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            device=device
        )

        rel_pred_labels = helper.convert_labels_by_dict(
            labels=rel_pred_labels,
            dictionary=args.dictionary,
            is_rel=True
        )
        no_rel_pred_labels = helper.convert_labels_by_dict(
            labels=no_rel_pred_labels,
            is_rel=False
        )

        no_rel_probs.append(no_rel_pred_probs)
        rel_probs.append(rel_pred_probs)

        no_rel_output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': no_rel_pred_labels,
            'probs': no_rel_pred_probs
        })
        no_rel_output.to_csv(
            path.join(args.no_rel_output_dir, (f'{k}_fold' if args.mode ==
                      'skf' else args.mode) + '_submission.csv'),
            index=False
        )

        rel_output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': rel_pred_labels,
            'probs': rel_pred_probs
        })
        rel_output.to_csv(
            path.join(args.rel_output_dir, (f'{k}_fold' if args.mode ==
                      'skf' else args.mode) + '_submission.csv'),
            index=False
        )

        def make_total_output(no_rel_output, rel_output, total_probs, k):
            print(no_rel_output.head())
            print(rel_output.head())
            rel_idx = no_rel_output.loc[no_rel_output['pred_label']
                                        != 'no_relation', :].index
            rel_df = rel_output.loc[rel_idx]
            no_rel_output[(no_rel_output['pred_label']
                           != 'no_relation')] = rel_df
            no_rel_output['probs'] = total_probs[k]
            return no_rel_output

        def calc_total_probs(no_rel, rel, k):
            # total_prob = [] * len(no_rel[k])
            total_prob = [[] for _ in range(len(no_rel[k]))]

            for i in range(len(no_rel[k])):
                if no_rel[k][i][0] >= no_rel[k][i][1]:  # rel
                    total_prob[i].append(no_rel[k][i][0])
                    for j in range(len(rel[k][i])):
                        total_prob[i].append(no_rel[k][i][1] * rel[k][i][j])

                else:  # no_rel
                    total_prob[i].append(no_rel[k][i][1])
                    for _ in range(len(rel[k][i])):
                        total_prob[i].append(no_rel[k][i][0] / 30)
            return total_prob

        total_probs.append(calc_total_probs(no_rel_probs, rel_probs, k))
        total_output = make_total_output(
            no_rel_output, rel_output, total_probs, k)
        total_output.to_csv(
            path.join('split_total_inf', (f'{k}_fold' if args.mode ==
                      'skf' else args.mode) + '_submission.csv'),
            index=False
        )

    if args.mode == 'skf':
        no_rel_probs = torch.tensor(no_rel_probs).mean(dim=0)
        no_rel_preds = torch.argmax(no_rel_probs, dim=-1).tolist()
        no_rel_preds = helper.convert_labels_by_dict(
            labels=no_rel_preds,
            dictionary=args.dictionary,
            is_rel=False
        )
        no_rel_output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': no_rel_preds,
            'probs': no_rel_probs.tolist()
        })
        no_rel_output.to_csv(path.join(args.no_rel_output_dir,
                                       f'{args.n_splits}_folds_submission.csv'), index=False)

        rel_probs = torch.tensor(rel_probs).mean(dim=0)
        rel_preds = torch.argmax(rel_probs, dim=-1).tolist()
        rel_preds = helper.convert_labels_by_dict(
            labels=rel_preds,
            dictionary=args.dictionary
        )
        rel_output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': rel_preds,
            'probs': rel_probs.tolist()
        })
        rel_output.to_csv(path.join(args.rel_output_dir,
                                    f'{args.n_splits}_folds_submission.csv'), index=False)

        total_probs = torch.tensor(total_probs).mean(dim=0)
        total_preds = torch.argmax(total_probs, dim=-1).tolist()

        def convert_labels_by_dict(labels, dictionary='data/dict_label_to_num.pkl'):
            with open(dictionary, 'rb') as f:
                dictionary = pickle.load(f)
            return np.array([dictionary[label] for label in labels])

        total_preds = convert_labels_by_dict(
            labels=total_preds
        )
        total_output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': total_preds,
            'probs': total_probs.tolist()
        })
        total_output.to_csv(path.join('split_total_inf',
                                      f'{args.n_splits}_folds_submission.csv'), index=False)

    print('Inference done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/test_data.csv')
    parser.add_argument('--is_rel', type=bool, default=False)
    parser.add_argument('--dictionary', type=str,
                        default='data/only_rel_num_to_label_start_0.pkl')
    parser.add_argument('--no_rel_output_dir', type=str,
                        default='./split_model_no_rel_large_inf')
    parser.add_argument('--no_rel_model_dir', type=str,
                        default='./split_model_no_rel_large')
    parser.add_argument('--rel_output_dir', type=str,
                        default='./split_model_rel_large_inf')
    parser.add_argument('--rel_model_dir', type=str,
                        default='./split_model_rel_large')

    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--add_ent_token', type=bool, default=True)

    args = parser.parse_args()
    print(args)

    inference(args=args)
