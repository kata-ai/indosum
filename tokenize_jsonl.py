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

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse
import copy
import json

import nltk
import spacy

from data import Document


def tokenize_text(nlp, text):
    return [[token.text for token in nlp(sent)]
            for sent in nltk.sent_tokenize(text)]


def tokenize_obj(nlp, obj):
    obj = copy.deepcopy(obj)
    tok_summary = tokenize_text(nlp, obj['summary'])
    tok_paragraphs = [tokenize_text(nlp, para) for para in obj['paragraphs']]
    obj['summary'] = tok_summary
    obj['paragraphs'] = tok_paragraphs
    return obj


def has_long_summary(doc: Document) -> bool:
    return len(doc.summary) >= len(doc.sentences)


def main(args):
    objs = []
    with open(args.path, encoding=args.encoding) as f:
        for linum, line in enumerate(f):
            try:
                objs.append(json.loads(line.strip()))
            except Exception as e:
                message = f'line {linum+1}: {e}'
                raise RuntimeError(message)

    nlp = spacy.blank('id')
    with ProcessPoolExecutor(max_workers=args.max_workers) as exc:
        tok_objs = exc.map(partial(tokenize_obj, nlp), objs, chunksize=args.chunk_size)
        docs = [Document.from_mapping(obj) for obj in tok_objs]
        if args.discard_long_summary:
            docs = [doc for doc in docs if not has_long_summary(doc)]
        print('\n'.join(json.dumps(doc.to_dict(), sort_keys=True) for doc in docs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tokenize a JSONL file from stdin to stdout.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path to the JSONL file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument(
        '-w', '--max-workers', type=int, help='max number of worker processes')
    parser.add_argument(
        '-c', '--chunk-size', type=int, default=1, help='chunk size for Executor.map')
    parser.add_argument(
        '--no-discard-long-summary', action='store_false', dest='discard_long_summary',
        help='do not discard articles having summary longer than the original text')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        parser.error(str(e))
