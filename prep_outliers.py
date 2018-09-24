#!/usr/bin/env python

##########################################################################
# Copyright 2018 Kata.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import argparse
import json
import math

import numpy as np


def truncate_paragraphs(obj, length):
    for key in 'paragraphs gold_labels pred_labels'.split():
        if key in obj:
            obj[key] = obj[key][:length]


def create_outlier_detector(data):
    q1, q3 = np.percentile(data, (25, 75))
    iqr = q3 - q1

    def is_outlier(x):
        return x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr
    return is_outlier


def read_jsonl(path, encoding='utf-8'):
    with open(args.path, encoding=args.encoding) as f:
        return [json.loads(line.strip()) for line in f]


def train_for_paras_length(train_objs):
    paras_lengths = [len(obj['paragraphs']) for obj in train_objs]
    return np.mean(paras_lengths), np.std(paras_lengths)


def train_for_summ_length(train_objs):
    summ_lengths = [len(obj['summary']) for obj in train_objs]
    return create_outlier_detector(summ_lengths)


def main(args):
    train_objs = read_jsonl(args.train_path, encoding=args.encoding)
    objs = read_jsonl(args.path, encoding=args.encoding)

    # Truncate paragraphs length
    mean, std = train_for_paras_length(train_objs)
    for obj in objs:
        truncate_paragraphs(obj, math.floor(mean + 2 * std))

    # Remove articles whose summary length is an outlier
    is_outlier = train_for_summ_length(train_objs)
    objs = [obj for obj in objs if not is_outlier(len(obj['summary']))]

    for obj in objs:
        print(json.dumps(obj, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess outliers in a given JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_path', help='path to the train JSONL file')
    parser.add_argument('path', help='path to the JSONL file to preprocess')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    args = parser.parse_args()
    main(args)
