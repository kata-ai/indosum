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
import pickle

from ingredients.corpus import ing as corpus_ingredient
from ingredients.evaluation import ing as eval_ingredient, run_evaluation
from ingredients.summarization import ing as summ_ingredient, run_summarization
from models.unsupervised import LexRank
from utils import SAVE_FILES, setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-lexrank-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # whether to perform continuous LexRank instead
    continuous = False
    # threshold for the cosine similarity
    sim_threshold = 0.1
    # weight for the LexRank feature
    weight = 0.5
    # threshold for the cross-sentence word overlap as an approximation to CSIS
    cswo_threshold = 0.5
    # path to a pickle file containing the IDF dictionary
    idf_path = None
    # damping factor for PageRank
    damping = 0.85
    # tolerance for convergence check
    tol = 1e-6
    # max number of iterations for the power method
    max_iter = 100


@ex.named_config
def tuned_on_fold1():
    continuous = True
    cswo_threshold = 0.994822
    seed = 621134730
    weight = 0.901003
    idf_path = None
    damping = 0.85
    tol = 1e-6
    max_iter = 100


@ex.named_config
def tuned_on_fold2():
    continuous = True
    cswo_threshold = 0.748531
    seed = 949249175
    weight = 0.838337
    idf_path = None
    damping = 0.85
    tol = 1e-6
    max_iter = 100


@ex.named_config
def tuned_on_fold3():
    continuous = True
    cswo_threshold = 0.636671
    seed = 235132204
    weight = 0.868785
    idf_path = None
    damping = 0.85
    tol = 1e-6
    max_iter = 100


@ex.named_config
def tuned_on_fold4():
    continuous = True
    cswo_threshold = 0.825453
    seed = 229015714
    weight = 0.517698
    idf_path = None
    damping = 0.85
    tol = 1e-6
    max_iter = 100


@ex.named_config
def tuned_on_fold5():
    continuous = True
    cswo_threshold = 0.849182
    seed = 104566206
    weight = 0.925247
    idf_path = None
    damping = 0.85
    tol = 1e-6
    max_iter = 100


@ex.capture
def create_model(continuous=False, sim_threshold=0.1, weight=0.5, cswo_threshold=0.5,
                 idf_path=None, damping=0.85, tol=1e-6, max_iter=100):
    idf_table = None if idf_path is None else read_idf()
    return LexRank(
        idf_table=idf_table, continuous=continuous, similarity_threshold=sim_threshold,
        weight=weight, cswo_threshold=cswo_threshold, damping_factor=damping, tol=tol,
        max_iter=max_iter)


@ex.capture
def read_idf(idf_path, _log, _run):
    _log.info('Reading IDF table from %s', idf_path)
    if SAVE_FILES:
        _run.add_resource(idf_path)
    with open(idf_path, 'rb') as f:
        return pickle.load(f)


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
