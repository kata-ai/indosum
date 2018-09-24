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

import json

from sacred import Ingredient

from data import Document
from utils import SAVE_FILES


ing = Ingredient('corpus')


@ing.config
def cfg():
    # file encoding
    encoding = 'utf-8'
    # path to train oracle JSONL file
    train = 'train.jsonl'
    # path to dev oracle JSONL file
    dev = None
    # path to test oracle JSONL file
    test = 'test.jsonl'
    # whether to lowercase words
    lower = True
    # whether to remove punctuations
    remove_puncts = True
    # whether to replace digits
    replace_digits = True
    # path to stopwords file, one per each line
    stopwords_path = None


@ing.capture
def read_train_jsonl(train):
    yield from read_jsonl(train, name='train')


@ing.capture
def read_dev_jsonl(dev):
    if dev is None:
        return None
    yield from read_jsonl(dev, name='dev')


@ing.capture
def read_test_jsonl(test):
    yield from read_jsonl(test, name='test')


@ing.capture
def read_jsonl(path, _log, _run, name='test', encoding='utf-8', lower=True, remove_puncts=True,
               replace_digits=True, stopwords_path=None):
    _log.info('Reading %s JSONL file from %s', name, path)
    if SAVE_FILES:
        _run.add_resource(path)
    stopwords = None if stopwords_path is None else read_stopwords(stopwords_path)

    with open(path, encoding=encoding) as f:
        for line in f:
            yield Document.from_mapping(
                json.loads(line.strip()), lower=lower, remove_puncts=remove_puncts,
                replace_digits=replace_digits, stopwords=stopwords)


@ing.capture
def read_stopwords(path, _log, _run, encoding='utf-8'):
    _log.info('Reading stopwords from %s', path)
    if SAVE_FILES:
        _run.add_resource(path)
    with open(path, encoding=encoding) as f:
        return set(f.read().strip().splitlines())
