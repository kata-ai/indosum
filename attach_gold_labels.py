#!/usr/bin/env python

import argparse
import json


def read_jsonl(path, encoding='utf-8'):
    with open(path, encoding=encoding) as f:
        for line in f:
            yield json.loads(line.strip())


def main(args):
    oracle_dict = {obj['id']: obj['gold_labels']
                   for obj in read_jsonl(args.oracle, encoding=args.encoding)}
    for obj in read_jsonl(args.path, encoding=args.encoding):
        if 'gold_labels' in obj:
            raise RuntimeError(f"object with id {obj['id']} already had 'gold_labels' as key")
        obj['gold_labels'] = oracle_dict[obj['id']]
        print(json.dumps(obj, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Attach gold labels from a JSONL oracle file to a non-oracle JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('oracle', help='path to the oracle JSONL file')
    parser.add_argument('path', help='path to the JSONL file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    args = parser.parse_args()
    main(args)
