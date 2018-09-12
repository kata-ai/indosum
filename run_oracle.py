#!/usr/bin/env python

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.oracle import OracleSummarizer
from utils import setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-oracle-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.command(unobserved=True)
def summarize():
    """Summarize the given file."""
    model = OracleSummarizer()
    run_summarization(model)


@ex.automain
def evaluate():
    """Evaluate on a corpus."""
    model = OracleSummarizer()
    return run_evaluation(model)
