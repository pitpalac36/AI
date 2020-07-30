import csv
from typing import re
import re
import numpy as np


def read_from_csv(fileName):
    data = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append(row)
            line_count += 1
    inputs = [data[i][0] for i in range(len(data))]
    outputs = [data[i][1] for i in range(len(data))]
    return inputs, outputs


def extract_words(sentence):
    useless = ['a', 'is', 'the', 'an']
    words = re.sub(r"[.,:()!?~=\'@#$%^&*_+-/\\`<>\"\n123456789]", " ", sentence).split()
    return [w.lower() for w in words if w not in useless]


def tokenize_sentences(text):
    words = []
    for each in text:
        w = extract_words(each)
        words.extend(w)
    words = sorted(list(set(words)))
    return words


def bag_of_words(sentence, words):
    sentence_words = extract_words(sentence)
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1
    return list(np.array(bag))
