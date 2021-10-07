from koeda import AEDA
import pprint
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

SPACE_TOKEN = "\u241F"


def replace_space(text: str) -> str:
    return text.replace(" ", SPACE_TOKEN)


def revert_space(text: list) -> str:
    clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
    return clean


class myAEDA(AEDA):
    def _aeda(self, data: str, p: float) -> str:
        if p is None:
            p = self.ratio

        split_words = self.morpheme_analyzer.morphs(replace_space(data))
        words = self.morpheme_analyzer.morphs(data)

        new_words = []
        q = random.randint(1, int(p * len(words) + 1))
        qs_list = [
            index
            for index in range(len(split_words))
            if split_words[index] != SPACE_TOKEN
        ]
        if len(qs_list) < q:
            q = len(qs_list)
        qs = random.sample(qs_list, q)

        for j, word in enumerate(split_words):
            if j in qs:
                new_words.append(SPACE_TOKEN)
                new_words.append(
                    self.punctuations[random.randint(
                        0, len(self.punctuations) - 1)]
                )
                new_words.append(SPACE_TOKEN)
                new_words.append(word)
            else:
                new_words.append(word)

        augmented_sentences = revert_space(new_words)

        return augmented_sentences


TRAIN_DATA_PATH = "data/train.csv"
pd_train = pd.read_csv(TRAIN_DATA_PATH)

pp = pprint.PrettyPrinter()

aeda = myAEDA(
    morpheme_analyzer="Mecab", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
)

text = list(pd_train['sentence'].values)

result = []
for _, t in enumerate(tqdm(text)):
    result.append(aeda(t, repetition=1))

# result = aeda(text, repetition=1)
print(pp.pprint(result[:5]))

tmp_pd = pd.DataFrame(
    columns=['id', 'sentence', 'subject_entity', 'object_entity', 'label', 'source'])


def init_idx(sentence, word):
    start = sentence.find(word)
    end = start + len(word) - 1
    return start, end


def create_dict(w, s, e, t):
    tmp_dict = dict()
    tmp_dict['word'] = w
    tmp_dict['start_idx'] = s
    tmp_dict['end_idx'] = e
    tmp_dict['type'] = t
    return tmp_dict


ids = []
sentences = []
sub_entities = []
obj_entities = []
labels = []
sources = []

count = 0
for i, sub in enumerate(list(pd_train['subject_entity'].values)):
    sub = eval(sub)
    if sub['word'] in result[i]:
        ids.append(count)
        count += 1

        sentences.append(result[i])

        start, end = init_idx(result[i], sub['word'])
        word = sub['word']
        t = sub['type']
        tmp_sub_d = create_dict(word, start, end, t)
        sub_entities.append(tmp_sub_d)

        obj = eval(pd_train['object_entity'][i])
        start, end = init_idx(result[i], obj['word'])
        word = obj['word']
        t = obj['type']
        tmp_obj_d = create_dict(word, start, end, t)
        obj_entities.append(tmp_obj_d)

        labels.append(pd_train['label'][i])
        sources.append(pd_train['source'][i])

tmp_pd['id'] = ids
tmp_pd['sentence'] = sentences
tmp_pd['subject_entity'] = sub_entities
tmp_pd['object_entity'] = obj_entities
tmp_pd['label'] = labels
tmp_pd['source'] = sources

tmp_pd.to_csv('aeda_1.csv')
