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

import logging
import os
import shutil
import tempfile

from pythonrouge.pythonrouge import Pythonrouge
from sacred.observers import MongoObserver


SAVE_FILES = os.getenv('SACRED_SAVE_FILES', 'false').lower() == 'true'


def setup_mongo_observer(ex):
    mongo_url = os.getenv('SACRED_MONGO_URL')
    db_name = os.getenv('SACRED_DB_NAME')
    if mongo_url is not None and db_name is not None:
        ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


def eval_summaries(summaries, docs, logger=None, encoding='utf-8', delete_temps=True):
    if logger is None:
        logger = logging.getLogger(__name__)

    references = []
    hypotheses = []
    for summary, doc in zip(summaries, docs):
        refs = [[' '.join(sent) for sent in doc.summary]]
        hyp = [' '.join(doc.sentences[idx].words) for idx in summary]
        references.append(refs)
        hypotheses.append(hyp)

    assert len(references) == len(hypotheses), 'Number of references and hypotheses mismatch'

    ref_dirname = tempfile.mkdtemp()
    logger.info('References directory: %s', ref_dirname)
    hyp_dirname = tempfile.mkdtemp()
    logger.info('Hypotheses directory: %s', hyp_dirname)
    for doc_id, (refs, hyp) in enumerate(zip(references, hypotheses)):
        # Write references
        for rid, ref in enumerate(refs):
            ref_filename = os.path.join(ref_dirname, f'{doc_id}.{rid}.txt')
            with open(ref_filename, 'w', encoding=encoding) as f:
                print('\n'.join(ref), file=f)
        # Write hypothesis
        hyp_filename = os.path.join(hyp_dirname, f'{doc_id}.txt')
        with open(hyp_filename, 'w', encoding=encoding) as f:
            print('\n'.join(hyp), file=f)

    rouge = Pythonrouge(
        peer_path=hyp_dirname, model_path=ref_dirname, stemming=False, ROUGE_L=True,
        ROUGE_SU4=False)
    score = rouge.calc_score()
    logger.info('ROUGE scores: %s', score)

    if delete_temps:
        logger.info('Deleting temporary files and directories')
        shutil.rmtree(ref_dirname)
        shutil.rmtree(hyp_dirname)

    return score
