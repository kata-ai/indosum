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

import pickle

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_jsonl
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.supervised import HMMSummarizer
from serialization import dump, load
from utils import SAVE_FILES, setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-hmm-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # where to load or save the trained model
    model_path = 'model'
    # smoothing for word probability
    gamma_word = 0.1
    # smoothing for initial transition probability
    gamma_init = 0.1
    # smoothing for transition probability
    gamma_trans = 0.1
    # path to a pickle file containing the TF dictionary
    tf_path = None


@ex.named_config
def tuned_on_fold1():
    seed = 786714831
    tf_path = None
    gamma_word = 0.1
    gamma_init = 0.1
    gamma_trans = 0.1


@ex.named_config
def tuned_on_fold2():
    seed = 466015822
    tf_path = 'tf_table.pkl'
    gamma_word = 0.1
    gamma_init = 0.1
    gamma_trans = 0.1


@ex.named_config
def tuned_on_fold3():
    seed = 203340366
    tf_path = None
    gamma_word = 0.1
    gamma_init = 0.1
    gamma_trans = 0.1


@ex.named_config
def tuned_on_fold4():
    seed = 120179441
    tf_path = 'tf_table.pkl'
    gamma_word = 0.1
    gamma_init = 0.1
    gamma_trans = 0.1


@ex.named_config
def tuned_on_fold5():
    seed = 569503380
    tf_path = 'tf_table.pkl'
    gamma_word = 0.1
    gamma_init = 0.1
    gamma_trans = 0.1


@ex.capture
def read_tf(tf_path, _log, _run):
    _log.info('Reading TF table from %s', tf_path)
    with open(tf_path, 'rb') as f:
        tf_table = pickle.load(f)
    if SAVE_FILES:
        _run.add_resource(tf_path)
    return tf_table


@ex.capture
def load_model(model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path) as f:
        model = load(f.read())
    assert isinstance(model, HMMSummarizer), 'model is not an HMM summarizer'
    if SAVE_FILES:
        _run.add_resource(model_path)
    return model


@ex.command
def train(model_path, _log, _run, gamma_word=0.1, gamma_init=0.1, gamma_trans=0.1,
          tf_path=None):
    """Train an HMM summarizer."""
    train_docs = list(read_train_jsonl())
    tf_table = None if tf_path is None else read_tf()
    model = HMMSummarizer.train(
        train_docs, gamma_word=gamma_word, gamma_init=gamma_init, gamma_trans=gamma_trans,
        tf_table=tf_table)
    _log.info('Saving model to %s', model_path)
    with open(model_path, 'w') as f:
        print(dump(model), file=f)
    if SAVE_FILES:
        _run.add_artifact(model_path)


@ex.command(unobserved=True)
def summarize():
    """Summarize the given file."""
    model = load_model()
    run_summarization(model)


@ex.automain
def evaluate():
    """Evaluate on a corpus."""
    model = load_model()
    return run_evaluation(model)
