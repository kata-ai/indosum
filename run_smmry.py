#!/usr/bin/env python

import json
import os
import shutil
import tempfile

from pythonrouge.pythonrouge import Pythonrouge
from sacred import Experiment
from sacred.observers import MongoObserver


ex = Experiment(name='summarization-smmry-testrun')

# Setup Mongo observer
mongo_url = os.getenv('SACRED_MONGO_URL')
db_name = os.getenv('SACRED_DB_NAME')
if mongo_url is not None and db_name is not None:
    ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))


SAVE_FILES = os.getenv('SACRED_SAVE_FILES', 'false').lower() == 'true'


@ex.config
def default_conf():
    # file encoding
    encoding = 'utf-8'
    # path to the JSONL file with precomputed SMMRY summaries
    precomputed_path = 'smmry.jsonl'
    # path to test JSONL file to evaluate
    test_path = 'test.jsonl'
    # whether to delete temporary directories and files
    delete_temps = True


@ex.capture
def read_jsonl(path, _log, _run, encoding='utf-8'):
    _log.info('Reading test JSONL file from %s', path)
    if SAVE_FILES:
        _run.add_resource(path)
    with open(path, encoding=encoding) as f:
        for line in f:
            yield json.loads(line.strip())


def get_summary_table(objs):
    return {obj['id']: obj['smmry_summary'] for obj in objs}


@ex.automain
def evaluate(precomputed_path, test_path, _log, _run, encoding='utf-8', delete_temps=True):
    """Evaluate SMMRY."""
    summary_table = get_summary_table(read_jsonl(precomputed_path))

    ref_dirname = tempfile.mkdtemp()
    _log.info('References directory: %s', ref_dirname)
    hyp_dirname = tempfile.mkdtemp()
    _log.info('Hypotheses directory: %s', hyp_dirname)
    try:
        for obj in read_jsonl(test_path, encoding=encoding):
            id_ = obj['id']
            # Write reference
            ref_path = os.path.join(ref_dirname, f'{id_}.1.txt')
            with open(ref_path, 'w', encoding=encoding) as f:
                print(obj['summary'], file=f)
            # Write hypothesis
            hyp_path = os.path.join(hyp_dirname, f'{id_}.txt')
            with open(hyp_path, 'w', encoding=encoding) as f:
                print(summary_table[id_], file=f)

        rouge = Pythonrouge(
            peer_path=hyp_dirname, model_path=ref_dirname, stemming=False, ROUGE_L=True,
            ROUGE_SU4=False)
        score = rouge.calc_score()
        _log.info('ROUGE scores: %s', score)
        for name, value in score.items():
            _run.log_scalar(name, value)

        return score['ROUGE-1-F']

    finally:
        if delete_temps:
            _log.info('Deleting temporary files and directories')
            shutil.rmtree(ref_dirname)
            shutil.rmtree(hyp_dirname)
