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
from models.unsupervised import LSA
from utils import setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-lsa-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.config
def default():
    # LSA summarization algorithm to use [gong, steinberger]
    algo = 'gong'


@ex.named_config
def tuned_on_fold1():
    algo = 'steinberger'
    seed = 826645469


@ex.named_config
def tuned_on_fold2():
    algo = 'steinberger'
    seed = 189985995


@ex.named_config
def tuned_on_fold3():
    algo = 'steinberger'
    seed = 383906628


@ex.named_config
def tuned_on_fold4():
    algo = 'steinberger'
    seed = 68632331


@ex.named_config
def tuned_on_fold5():
    algo = 'steinberger'
    seed = 376780


@ex.capture
def create_model(algo='gong'):
    return LSA(algorithm=algo)


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
