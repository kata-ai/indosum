import json

from sacred import Ingredient

from ingredients.corpus import ing as corpus_ingredient, read_jsonl
from models import AbstractSummarizer


# TODO Putting corpus ingredient here does not feel right. When summarizing, we do not need
# the corpus. Any jsonl file will do. What we need here is the `read_jsonl` function and its
# preprocessing. That might be best put in a separate ingredient.
ing = Ingredient('summ', ingredients=[corpus_ingredient])


@ing.config
def cfg():
    # path to the JSONL file to summarize
    path = 'test.jsonl'
    # extract at most this number of sentences as summary
    size = 3


@ing.capture
def run_summarization(model: AbstractSummarizer, path, size=3):
    for doc in read_jsonl(path):
        summary = set(model.summarize(doc, size=size))
        sent_id = 0
        for para in doc.paragraphs:
            for sent in para:
                sent.pred_label = sent_id in summary
                sent_id += 1
        print(json.dumps(doc.to_dict(), sort_keys=True))
