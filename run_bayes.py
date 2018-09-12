#!/usr/bin/env python

import pickle

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_jsonl
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.supervised import NaiveBayesSummarizer
from serialization import dump, load
from utils import SAVE_FILES, setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-bayes-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # where to load or save the trained model
    model_path = 'model'
    # proportion of words with highest TF-IDF score to be considered important words
    cutoff = 0.1
    # path to a pickle file containing the IDF dictionary
    idf_path = None


@ex.named_config
def tuned_on_fold1():
    cutoff = 0.249972
    seed = 880655900
    idf_path = 'idf_table.pkl'


@ex.named_config
def tuned_on_fold2():
    cutoff = 0.0671735
    seed = 467408239
    idf_path = 'idf_table.pkl'


@ex.named_config
def tuned_on_fold3():
    cutoff = 0.00148999
    seed = 262472316
    idf_path = None


@ex.named_config
def tuned_on_fold4():
    cutoff = 0.000565754
    seed = 163917137
    idf_path = None


@ex.named_config
def tuned_on_fold5():
    cutoff = 0.109262
    seed = 67740894
    idf_path = None


@ex.capture
def read_idf(idf_path, _log, _run):
    _log.info('Reading IDF table from %s', idf_path)
    with open(idf_path, 'rb') as f:
        idf_table = pickle.load(f)
    if SAVE_FILES:
        _run.add_resource(idf_path)
    return idf_table


@ex.capture
def load_model(model_path, _log, _run):
    _log.info('Loading model from %s', model_path)
    with open(model_path) as f:
        model = load(f.read())
    assert isinstance(model, NaiveBayesSummarizer), 'model is not a naive Bayes summarizer'
    if SAVE_FILES:
        _run.add_resource(model_path)
    return model


@ex.command
def train(model_path, _log, _run, cutoff=0.1, idf_path=None):
    """Train a naive Bayes summarizer."""
    train_docs = list(read_train_jsonl())
    idf_table = None if idf_path is None else read_idf()
    model = NaiveBayesSummarizer.train(train_docs, cutoff=cutoff, idf_table=idf_table)
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
