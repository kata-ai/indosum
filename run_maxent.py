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

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_jsonl
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.supervised import MaxentSummarizer
from serialization import dump, load
from utils import SAVE_FILES, setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-maxent-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # where to load or save the trained model
    model_path = 'model'
    # path to file containing stopwords, one per line
    stopwords_path = None
    # training algorithm [gis, iis, megam]
    train_algo = 'iis'
    # count cutoff for rare features
    cutoff = 4
    # standard deviation for Gaussian prior on the weights (default: no prior)
    sigma = 0.
    # trim words to this length
    trim_length = 10


@ex.named_config
def tuned_on_fold1():
    cutoff = 5
    seed = 795707921
    sigma = 2.30731
    stopwords_path = None
    trim_length = 10


@ex.named_config
def tuned_on_fold2():
    cutoff = 3
    seed = 955017972
    sigma = 2.11229
    stopwords_path = None
    trim_length = 10


@ex.named_config
def tuned_on_fold3():
    cutoff = 2
    seed = 161045250
    sigma = 227.81
    stopwords_path = None
    trim_length = 10


@ex.named_config
def tuned_on_fold4():
    cutoff = 9
    seed = 608320006
    sigma = 1.7715
    stopwords_path = None
    trim_length = 10


@ex.named_config
def tuned_on_fold5():
    cutoff = 2
    seed = 648134882
    sigma = 0.351031
    stopwords_path = 'stopwords.txt'
    trim_length = 10


@ex.capture
def read_stopwords(stopwords_path, corpus, _log, _run):
    _log.info('Reading stopwords from %s', stopwords_path)
    with open(stopwords_path, encoding=corpus['encoding']) as f:
        stopwords = set(f.read().strip().splitlines())
    if SAVE_FILES:
        _run.add_resource(stopwords_path)
    return stopwords


@ex.capture
def load_model(model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path) as f:
        model = load(f.read())
    assert isinstance(model, MaxentSummarizer), 'model is not a maxent summarizer'
    if SAVE_FILES:
        _run.add_resource(model_path)
    return model


@ex.command
def train(model_path, _log, _run, stopwords_path=None, train_algo='iis', cutoff=4, sigma=0.,
          trim_length=10):
    """Train a maximum entropy summarizer."""
    train_docs = list(read_train_jsonl())
    stopwords = None if stopwords_path is None else read_stopwords()
    model = MaxentSummarizer.train(
        train_docs, stopwords=stopwords, algorithm=train_algo, cutoff=cutoff, sigma=sigma,
        trim_length=trim_length)
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
