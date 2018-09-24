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

import json
import os
import sys

from sacred import Experiment

from data import Document
from utils import SAVE_FILES, eval_summaries, setup_mongo_observer


here = os.path.dirname(__file__)
neuralsum_dir = os.path.join(here, 'neuralsum')
ex = Experiment(name='summarization-neuralsum-testrun')
setup_mongo_observer(ex)


@ex.config
def default_conf():
    # random seed
    seed = 3435
    # file encoding
    encoding = 'utf-8'
    # model choice [lstm, bilstm]
    model_choice = 'lstm'
    # path to pretrained embedding file
    embedding_path = None
    # size of LSTM hidden layers
    rnn_size = 650
    # number of highway layers
    highway_layers = 2
    # size of word embedding
    word_embed_size = 50
    # CNN kernel widths
    kernels = '[1,2,3,4,5,6,7]'
    # number of kernels for each width
    kernel_features = '[50,100,150,200,200,200,200]'
    # number of LSTM layers
    rnn_layers = 2
    # dropout rate
    dropout = 0.5
    # where to load the model from
    load_model = None
    # data directory, should have train, valid, and test directories
    data_dir = 'oracle'
    # training directory, checkpoints will be saved here
    train_dir = 'train-dir'
    # learning rate decay
    learning_rate_decay = 0.5
    # initial learning rate
    learning_rate = 1.0
    # decay if dev perplexity does not improve by more than this much
    decay_when = 1.0
    # initialize parameters uniformly with this value as limit
    param_init = 0.05
    # batch size
    batch_size = 16
    # maximum number of epochs to train
    max_epochs = 25
    # normalize gradients at
    max_grad_norm = 5.
    # maximum number of sentences a document may have
    max_doc_length = 15
    # maximum number of words a sentence may have
    max_sen_length = 50
    # how often to print loss
    print_every = 5
    # which dataset to evaluate on
    on = 'test'
    # path to the JSONL file containing the dataset to evaluate on
    jsonl_path = 'test.jsonl'
    # test batch size (must evenly divides the number of articles)
    test_batch_size = 1
    # maximum number of sentences in the summary
    size = 3
    # whether to lowercase words (only evaluate)
    lower = True
    # whether to delete temporary directories and files
    delete_temps = True


@ex.named_config
def tuned_on_fold1():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 50
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.3
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 3
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def tuned_on_fold2():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 50
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.2
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 3
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def tuned_on_fold3():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 50
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.2
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 5
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def tuned_on_fold4():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 50
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.7
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def tuned_on_fold5():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 50
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.2
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def emb300_on_fold1():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 4
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def emb300_on_fold2():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 4
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def emb300_on_fold3():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def emb300_on_fold4():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def emb300_on_fold5():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = None
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def fasttext_on_fold1():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = 'wiki.id.vec'
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 5
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def fasttext_on_fold2():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = 'wiki.id.vec'
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 3
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def fasttext_on_fold3():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = 'wiki.id.vec'
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 1
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def fasttext_on_fold4():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = 'wiki.id.vec'
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 3
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.named_config
def fasttext_on_fold5():
    seed = 3435
    encoding = 'utf-8'
    model_choice = 'lstm'
    embedding_path = 'wiki.id.vec'
    rnn_size = 650
    highway_layers = 2
    word_embed_size = 300
    kernels = '[1,2,3,4,5,6,7]'
    kernel_features = '[50,100,150,200,200,200,200]'
    rnn_layers = 2
    dropout = 0.5
    learning_rate_decay = 0.5
    learning_rate = 1.0
    decay_when = 1.0
    param_init = 0.05
    batch_size = 16
    max_epochs = 2
    max_grad_norm = 5.
    max_doc_length = 15
    max_sen_length = 50
    print_every = 5


@ex.capture
def get_model_conf(
        model_choice='lstm', embedding_path=None, rnn_size=650, highway_layers=2,
        word_embed_size=50, kernels='[1,2,3,4,5,6,7]',
        kernel_features='[50,100,150,200,200,200,200]', rnn_layers=2, dropout=0.5,
        load_model=None):
    conf = {
        'model_choice': model_choice,
        'rnn_size': rnn_size,
        'highway_layers': highway_layers,
        'word_embed_size': word_embed_size,
        'kernels': kernels,
        'kernel_features': kernel_features,
        'rnn_layers': rnn_layers,
        'dropout': dropout,
    }

    if embedding_path is not None:
        conf['embedding_path'] = os.path.realpath(embedding_path)
    if load_model is not None:
        conf['load_model'] = os.path.realpath(load_model)

    return conf


@ex.capture
def read_jsonl(path, _log, _run, name='test', encoding='utf-8', lower=True):
    _log.info('Reading %s JSONL file from %s', name, path)
    with open(path, encoding=encoding) as f:
        for line in f:
            yield Document.from_mapping(json.loads(line.strip()), lower=lower)
    if SAVE_FILES:
        _run.add_resource(path)


@ex.command
def train(
        seed, _log, data_dir='oracle', train_dir='train-dir', learning_rate_decay=0.5,
        learning_rate=1.0, decay_when=1.0, param_init=0.05, batch_size=16, max_epochs=25,
        max_grad_norm=5., max_doc_length=15, max_sen_length=50, print_every=5):
    """Train a NeuralSum model."""
    conf = {
        'seed': seed,
        'data_dir': os.path.realpath(data_dir),
        'train_dir': os.path.realpath(train_dir),
        'learning_rate_decay': learning_rate_decay,
        'learning_rate': learning_rate,
        'decay_when': decay_when,
        'param_init': param_init,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'max_grad_norm': max_grad_norm,
        'max_doc_length': max_doc_length,
        'max_sen_length': max_sen_length,
        'print_every': print_every,
    }
    conf.update(get_model_conf())

    args = ' '.join([f'--{key} {value}' for key, value in conf.items()])
    script = os.path.join(neuralsum_dir, 'train.py')
    cmd = f'python {script} {args}'
    _log.info('Command is %s', cmd)
    retval = os.system(cmd)
    if retval:
        sys.exit(retval)


@ex.automain
def evaluate(
        seed, _log, _run, encoding='utf-8', data_dir='oracle', train_dir='train-dir',
        on='test', jsonl_path='test.jsonl', test_batch_size=1, max_doc_length=15,
        max_sen_length=50, size=3, lower=True, delete_temps=True):
    """Evaluate a trained NeuralSum model on a dataset."""
    conf = {
        'seed': seed,
        'data_dir': os.path.realpath(data_dir),
        'train_dir': os.path.realpath(train_dir),
        'which': on,
        'jsonl_path': os.path.realpath(jsonl_path),
        'batch_size': test_batch_size,
        'max_doc_length': max_doc_length,
        'max_sen_length': max_sen_length,
    }
    conf.update(get_model_conf())

    args = ' '.join([f'--{key} {value}' for key, value in conf.items()])
    script = os.path.join(neuralsum_dir, 'summarize.py')
    cmd = f'python {script} {args}'
    _log.info('Command: %s', cmd)
    retval = os.system(cmd)
    if retval:
        sys.exit(retval)

    summary_labels = {}
    with open(os.path.join(train_dir, 'summary.txt'), encoding='utf-8') as f:
        for line in f:
            id_, labels = line.strip().split('\t\t\t')
            summary_labels[id_] = [int(l) for l in labels.split()]

    docs = list(read_jsonl(jsonl_path, name=on, lower=lower))

    summaries = []
    for doc in docs:
        labels = summary_labels[doc.id_][:len(doc.sentences)]
        summary = [k for k, label in enumerate(labels) if label == 1]
        summaries.append(summary[:size])

    score = eval_summaries(
        summaries, docs, logger=_log, encoding=encoding, delete_temps=delete_temps)
    for name, value in score.items():
        _run.log_scalar(name, value)

    return score['ROUGE-1-F']
