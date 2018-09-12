import numpy as np
from contextlib import contextmanager
import operator

def softmax(logits):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

def argmax(prob):
    """Return argmax id"""
    prob = prob.ravel()
    return np.argmax(prob)

def topk(prob, k):
    """Return top k id"""
    prob = prob.ravel()
    return (-prob).argsort()[:k]

def random(prob):
    """Return sampled id"""
    prob = prob.ravel()
    return np.random.choice(range(len(prob)), p=prob)

def constrained(prob, restrictions, last_p):
    """Return argmax under restricted vocab"""
    prob = prob.ravel()
    restrictions = restrictions.ravel()
    sorted_prob = sorted(enumerate(prob), key=operator.itemgetter(1), reverse=True)
    for p in sorted_prob:
        # simple way to remove repetition
        if p[0] != last_p and p[0] in restrictions:
            return p[0]

@contextmanager
def open_files(names, mode='r'):
    """ Safely open a list of files in a context manager.
    Example:
    >>> with open_files(['foo.txt', 'bar.csv']) as (f1, f2):
    ...   pass
    """

    files = []
    try:
        for name_ in names:
            files.append(open(name_, mode=mode))
        yield files
    finally:
        for file_ in files:
            file_.close()


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def read_ngrams(lm_path, vocab):
    """
    Read a language model from a file in the ARPA format,
    and return it as a list of dicts.
    :param lm_path: full path to language model file
    :param vocab: vocabulary used to map words from the LM to token ids
    :return: one dict for each ngram order, containing mappings from
      ngram (as a sequence of token ids) to (log probability, backoff weight)
    """
    ngram_list = []
    with open(lm_path) as f:
        for line in f:
            line = line.strip()
            if re.match(r'\\\d-grams:', line):
                ngram_list.append({})
            elif not line or line == '\\end\\':
                continue
            elif ngram_list:
                arr = list(map(str.rstrip, line.split('\t')))
                ngram = arr.pop(1)
                ngram_list[-1][ngram] = list(map(float, arr))

    debug('loaded n-grams, order={}'.format(len(ngram_list)))

    ngrams = []
    mappings = {'<s>': _BOS, '</s>': _EOS, '<unk>': _UNK}

    for kgrams in ngram_list:
        d = {}
        for seq, probas in kgrams.items():
            ids = tuple(vocab.get(mappings.get(w, w)) for w in seq.split())
            if any(id_ is None for id_ in ids):
                continue
            d[ids] = probas
        ngrams.append(d)
    return ngrams

def estimate_lm_score(sequence, ngrams):
    """
    Compute the log score of a sequence according to given language model.
    :param sequence: list of token ids
    :param ngrams: list of dicts, as returned by `read_ngrams`
    :return: log probability of `sequence`
    P(w_3 | w_1, w_2) =
        log_prob(w_1 w_2 w_3)             } if (w_1 w_2 w_3) in language model
        P(w_3 | w_2) + backoff(w_1 w_2)   } otherwise
    in case (w_1 w_2) has no backoff weight, a weight of 0.0 is used
    """
    sequence = tuple(sequence)
    order = len(sequence)
    assert 0 < order <= len(ngrams)
    ngrams_ = ngrams[order - 1]

    if sequence in ngrams_:
        return ngrams_[sequence][0]
    else:
        weights = ngrams[order - 2].get(sequence[:-1])
        backoff_weight = weights[1] if weights is not None and len(weights) > 1 else 0.0
        return estimate_lm_score(sequence[1:], ngrams) + backoff_weight

