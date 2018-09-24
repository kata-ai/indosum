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
