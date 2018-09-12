#!/usr/bin/env python

from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.unsupervised import TextRank
from utils import setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-textrank-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # damping factor for PageRank
    damping = 0.85
    # tolerance for convergence check
    tol = 1e-6
    # max number of iterations for the power method
    max_iter = 100


@ex.capture
def create_model(damping=0.85, tol=1e-6, max_iter=100):
    return TextRank(damping_factor=damping, tol=tol, max_iter=max_iter)


@ex.command(unobserved=True)
def summarize():
    """Summarize the given file."""
    model = create_model()
    run_summarization(model)


@ex.automain
def evaluate():
    """Evaluate on a corpus."""
    model = create_model()
    return run_evaluation(model)
