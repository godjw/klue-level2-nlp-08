import argparse
from os import path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import pandas as pd
from tqdm import tqdm

from utils import *


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

    probs = []
    for k in range(args.n_splits if args.mode == 'skf' else 1):
        model = AutoModelForSequenceClassification.from_pretrained(
            path.join(args.model_dir,
                      f'{k}_fold' if args.mode == 'skf' else args.mode)
        )
        model.to(device)

        pred_labels, pred_probs = infer(
            model=model,
            test_dataset=test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            device=device
        )
        pred_labels = helper.convert_labels_by_dict(
            labels=pred_labels,
            dictionary=args.dictionary
        )
        probs.append(pred_probs)

        output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': pred_labels,
            'probs': pred_probs
        })
        output.to_csv(
            path.join(args.output_dir, (f'{k}_fold_nr_augment' if args.mode ==
                      'skf' else args.mode) + '_submission.csv'),
            index=False
        )

    if args.mode == 'skf':
        probs = torch.tensor(probs).mean(dim=0)
        preds = torch.argmax(probs, dim=-1).tolist()
        preds = helper.convert_labels_by_dict(
            labels=preds,
            dictionary=args.dictionary
        )
        output = pd.DataFrame({
            'id': _test_data['id'],
            'pred_label': preds,
            'probs': probs.tolist()
        })
        output.to_csv(path.join(args.output_dir,
                      f'{args.n_splits}_folds_nr_augment_submission.csv'), index=False)

    print('Inference done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/test_data.csv')
    parser.add_argument('--dictionary', type=str,
                        default='data/dict_num_to_label.pkl')
    parser.add_argument('--output_dir', type=str, default='./prediction')
    parser.add_argument('--model_dir', type=str, default='./best_model')

    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--mode', type=str, default='plain',
                        choices=['plain', 'skf'])
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--add_ent_token', type=bool, default=True)

    args = parser.parse_args()
    print(args)

    inference(args=args)
