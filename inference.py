import argparse

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
    model.eval()
    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            )
        logits = outputs[0]
        result = torch.argmax(logits, axis=-1)
        prob = F.softmax(logits, dim=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return torch.cat(output_pred).tolist(), torch.cat(output_prob, dim=0).tolist()


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    helper = DataHelper(data_dir=args.data_dir)
    preprocessed, test_labels = helper.preprocess(mode='inference')
    test_data = helper.tokenize(data=preprocessed, tokenizer=tokenizer)
    test_dataset = RelationExtractionDataset(test_data, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
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

    output = pd.DataFrame({
        'id': preprocessed['id'],
        'pred_label': pred_labels,
        'probs': pred_probs
    })
    output.to_csv(args.output_path, index=False)

    print('Inference done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default='../dataset/test/test_data.csv')
    parser.add_argument('--dictionary', type=str,
                        default='./dict_num_to_label.pkl')

    parser.add_argument('--model_dir', type=str, default='./best_model')
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--output_path', type=str,
                        default='./prediction/submission.csv')
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    print(args)

    inference(args=args)
