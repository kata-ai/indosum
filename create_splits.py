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
import collections
import json
import math
import os
import random
import sys


def read_jsonl(path, encoding='utf-8'):
    with open(path, encoding=encoding) as f:
        for line in f:
            yield json.loads(line.strip())


def get_filename(name, fold=None):
    fmt = '{0}.jsonl' if fold is None else '{0}.{1:02}.jsonl'
    return fmt.format(name, fold)


def write_splits(splits, outdir, fold=None, encoding='utf-8'):
    for split, name in zip(splits, ('train', 'dev', 'test')):
        filename = get_filename(name, fold=fold)
        with open(os.path.join(outdir, filename), 'w', encoding=encoding) as f:
            print('\n'.join(json.dumps(obj, sort_keys=True) for obj in split), file=f)


def report_stats(splits, fold=None):
    for split, name in zip(splits, ('train', 'dev', 'test')):
        filename = get_filename(name, fold)
        print(filename, file=sys.stderr)
        for group_by in ('category', 'source'):
            print('*', group_by, file=sys.stderr)
            c = collections.Counter(obj[group_by] for obj in split)
            for label, count in c.most_common():
                ratio = count / len(split)
                print(f'  - {label}: {count} ({ratio:.1%})', file=sys.stderr)
        print(file=sys.stderr)


def main(args):
    random.seed(args.seed)
    all_data = list(read_jsonl(args.path, encoding=args.encoding))
    random.shuffle(all_data)
    os.makedirs(args.outdir, exist_ok=True)

    if args.num_folds == 1:
        n_test = math.floor(len(all_data) * args.test)
        n_dev = math.floor(len(all_data) * args.dev)
        test = all_data[:n_test]
        dev = all_data[n_test:n_test + n_dev]
        train = all_data[n_test + n_dev:]
        write_splits((train, dev, test), args.outdir, encoding=args.encoding)
        report_stats((train, dev, test))
    else:
        for fold in range(args.num_folds):
            rest, test = [], []
            for i, obj in enumerate(all_data):
                if i % args.num_folds == fold:
                    test.append(obj)
                else:
                    rest.append(obj)
            random.shuffle(rest)
            n_dev = math.floor(len(rest) * args.dev)
            dev, train = rest[:n_dev], rest[n_dev:]
            write_splits(
                (train, dev, test), args.outdir, fold=fold + 1, encoding=args.encoding)
            report_stats((train, dev, test), fold=fold + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create dataset splits from a JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path to the JSONL file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument(
        '-o', '--output-dir', metavar='DIR', dest='outdir', default='splits',
        help='output directory to save the splits in')
    parser.add_argument(
        '--dev', type=float, default=0.05,
        help='proportion of dev set (from the non-test set for k-fold CV)')
    parser.add_argument(
        '--test', type=float, default=0.05,
        help='proportion of test set (only used if number of folds is 1)')
    parser.add_argument('-k', '--num-folds', type=int, default=1, help='number of folds')
    parser.add_argument('--seed', type=int, default=12345, help='seed for random generator')
    args = parser.parse_args()
    if args.dev < 0 or args.test < 0:
        parser.error('proportion of dev and test set must be positive')
    if args.dev + args.test >= 1:
        parser.error('sum of proportions of dev and test set must be less than 1')
    if args.num_folds <= 0:
        parser.error('number of folds must be positive')
    main(args)
