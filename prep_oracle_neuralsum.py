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
import os
import re

from data import Document


NEWLINES = re.compile(r'\n+')


def remove_whitespace_only_words(doc: Document):
    for para in doc:
        for sent in para:
            words = [w for w in sent.words if w.strip()]
            sent.words = words


def write_neuralsum_oracle(doc: Document, output_dir: str, encoding='utf-8'):
    remove_whitespace_only_words(doc)
    filename = f'{doc.id_}.summary'
    with open(os.path.join(output_dir, filename), 'w', encoding=encoding) as f:
        print(doc.source_url, file=f)
        print(file=f)
        print(NEWLINES.sub('\n', str(doc)), file=f, end='')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.path, encoding=args.encoding) as f:
        for line in f:
            doc = Document.from_mapping(
                json.loads(line.strip()), lower=args.lower)
            write_neuralsum_oracle(doc, args.output_dir, encoding=args.encoding)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare oracles for NeuralSum from an oracle JSONL file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', help='path to the JSONL file')
    parser.add_argument('--encoding', default='utf-8', help='file encoding')
    parser.add_argument(
        '--no-lower', action='store_false', dest='lower', help='do not lowercase words')
    parser.add_argument('-o', '--output-dir', default='output', help='output directory')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        parser.error(str(e))
