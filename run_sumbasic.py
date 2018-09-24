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
from models.unsupervised import SumBasic
from utils import setup_mongo_observer


ingredients = [corpus_ingredient, eval_ingredient, summ_ingredient]
ex = Experiment(name='summarization-sumbasic-testrun', ingredients=ingredients)
setup_mongo_observer(ex)


@ex.command(unobserved=True)
def summarize():
    """Summarize the given file."""
    model = SumBasic()
    run_summarization(model)


@ex.automain
def evaluate():
    """Evaluate on a corpus."""
    model = SumBasic()
    return run_evaluation(model)
