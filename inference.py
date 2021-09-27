import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import *


def infer(model, test_dataset, device):
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        # prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        # logits = logits.detach().cpu().numpy()
        result = torch.argmax(logits, axis=-1)
        prob = F.softmax(logits, dim=-1)

        output_pred.append(result)
        output_prob.append(prob)
    
    # return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    return torch.cat(output_pred).tolist(), torch.cat(output_prob, dim=0).tolist()


def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])

    helper = DataHelper(data_dir=args['data_dir'])
    preprocessed, test_labels = helper.preprocess(mode='inference')
    test_data = helper.tokenize(data=preprocessed, tokenizer=tokenizer)

    test_dataset = RelationExtractionDataset(test_data, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(args['model_dir'])
    model.parameters
    model.to(device)

    # predict answer
    pred_labels, pred_probs = infer(
        model=model,
        test_dataset=test_dataset,
        device=device
    )
    pred_labels = helper.convert_by(
        labels=pred_labels,
        dictionary=args['dictionary']
    )

    # make csv file with predicted answer
    output = pd.DataFrame({
        'id': preprocessed['id'],
        'pred_label': pred_labels,
        'probs': pred_probs
    })
    output.to_csv(args['output_path'], index=False)

    print('Inference done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../dataset/test/test_data.csv')
    parser.add_argument('--dictionary', type=str, default='./dict_id2label.pkl')

    parser.add_argument('--model_dir', type=str, default='./best_model')
    parser.add_argument('--model_name', type=str, default='klue/bert-base')
    parser.add_argument('--output_path', type=str, default='./prediction/submission.csv')
    
    args = parser.parse_args()
    print(args)

    inference(args=args)
