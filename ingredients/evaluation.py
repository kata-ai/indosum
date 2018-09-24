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

from sacred import Ingredient

from ingredients.corpus import ing as corpus_ingredient, read_dev_jsonl, read_test_jsonl, \
    read_train_jsonl
from models import AbstractSummarizer
from utils import eval_summaries


ing = Ingredient('eval', ingredients=[corpus_ingredient])


@ing.config
def cfg():
    # which corpus set the evaluation should be run on
    on = 'test'
    # extract at most this number of sentences as summary
    size = 3
    # whether to delete temp files after finishes
    delete_temps = True


@ing.capture
def run_evaluation(model: AbstractSummarizer, corpus, _log, _run, on='test', size=3,
                   delete_temps=True):
    try:
        read_fn = {
            'train': read_train_jsonl,
            'dev': read_dev_jsonl,
            'test': read_test_jsonl,
        }[on]
    except KeyError:
        msg = f'{on} is not a valid corpus set, possible choices are: train, dev, test'
        raise RuntimeError(msg)

    docs = list(read_fn())
    summaries = [model.summarize(doc, size=size) for doc in docs]

    score = eval_summaries(
        summaries, docs, logger=_log, encoding=corpus['encoding'], delete_temps=delete_temps)
    for name, value in score.items():
        _run.log_scalar(name, value)
    return score['ROUGE-1-F']
