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

from collections import defaultdict
from typing import Collection, Dict, List, Mapping, Optional, Set, Sequence, Tuple
import math

from nltk.classify import BinaryMaxentFeatureEncoding, MaxentClassifier, NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ConditionalProbDistI, \
    FreqDist, LidstoneProbDist, ProbDistI
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import numpy as np

from data import Document, Sentence, Word
from models import AbstractSummarizer


class HMMSummarizer(AbstractSummarizer):
    """Summarizer using hidden Markov model (Conroy and O'Leary, 2001).

    In this formulation of HMM, the initial and transition probability are multinomial whereas
    the emission probability is Gaussian. The Gaussian mean is estimated for every possible tag
    while the covariance matrix is estimated using the whole samples. In other words, the
    covariance matrix is shared. There is a difference from the original paper: we do not use
    QR decomposition for sentence selection.

    Args:
        init_pdist (nltk.probability.ProbDistI): Initial state probability.
        trans_pdist (nltk.probability.ConditionalProbDistI): Transition probability.
        emit_pdist (nltk.probability.ConditionalProbDistI): Emission probability.
        states (Sequence[int]): A sequence of possible states.
        gamma (float): Smoothing value for the "word probability in a document" feature.
        tf_table (Mapping[Word, float]): A precomputed term-frequency table that is already
            normalized.
    """
    def __init__(self,
                 init_pdist: ProbDistI,
                 trans_pdist: ConditionalProbDistI,
                 emit_pdist: ConditionalProbDistI,
                 states: Sequence[int],
                 gamma: float = 0.1,
                 tf_table: Optional[Mapping[Word, float]] = None,
                 ) -> None:
        self.init_pdist = init_pdist
        self.trans_pdist = trans_pdist
        self.emit_pdist = emit_pdist
        self.states = states
        self.gamma = gamma
        self.tf_table = tf_table

        self._start_transitions = None
        self._transitions = None

    @classmethod
    def train(cls,
              docs: Collection[Document],
              gamma_word: float = 0.1,
              gamma_init: float = 0.1,
              gamma_trans: float = 0.1,
              tf_table: Optional[Mapping[Word, float]] = None,
              ) -> 'HMMSummarizer':
        """Train the model on a collection of documents.

        Args:
            docs (Collection[Document]): The collection of documents to train on.
            gamma_word (float): Smoothing value for the "word probability in a document"
                feature.
            gamma_init (float): Smoothing value for the initial probability.
            gamma_trans (float): Smoothing value for the transition probability.
            tf_table (Mapping[Word, float]): A precomputed term-frequency table that is already
                normalized.

        Returns:
            HMM: The trained model.
        """
        init_fdist = FreqDist()
        trans_fdist = ConditionalFreqDist()
        tagged_vecs: list = []
        states = set()

        for doc in docs:
            tags = cls._get_tags(doc.sentences)
            if not tags:
                continue

            init_fdist[tags[0]] += 1
            for prev, tag in zip(tags, tags[1:]):
                trans_fdist[prev][tag] += 1
            vecs = cls._get_feature_vectors(doc, gamma_word, tf=tf_table)
            tagged_vecs.extend(zip(vecs, tags))
            states.update(tags)

        # Initial probability
        init_pdist = LidstoneProbDist(init_fdist, gamma_init, bins=len(states))
        # Transition probability
        trans_pdist = ConditionalProbDist(
            trans_fdist, LidstoneProbDist, gamma_trans, bins=len(states))
        # Emission probability
        emit_pdist = _GaussianEmission.train(tagged_vecs)
        return cls(
            init_pdist, trans_pdist, emit_pdist, list(states), gamma=gamma_word,
            tf_table=tf_table)

    def summarize(self, doc: Document, size: int) -> List[int]:
        """Summarize a given document.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        vecs = self._get_feature_vectors(doc, self.gamma, tf=self.tf_table)
        gamma = self._compute_gamma(vecs)
        summ_states = [i for i, st in enumerate(self.states) if st % 2 == 0]
        scores = gamma[:, summ_states].sum(axis=1)
        summary = sorted(
            range(len(doc.sentences)), key=lambda k: scores[k], reverse=True)[:size]
        return summary

    @classmethod
    def _get_tags(cls, sents: Sequence[Sentence]) -> List[int]:
        if not sents:
            return []

        tags = [2 if sents[0].label else 1]
        for sent in sents[1:]:
            if tags[-1] % 2:
                next_tag = tags[-1] + (1 if sent.label else 0)
            else:
                next_tag = tags[-1] + (2 if sent.label else 1)
            tags.append(next_tag)
        return tags

    @classmethod
    def _get_feature_vectors(cls,
                             doc: Document,
                             gamma: float,
                             tf: Optional[Mapping[Word, float]] = None,
                             ) -> List[np.ndarray]:
        word_fdist = FreqDist(doc.words)
        word_pdist = LidstoneProbDist(word_fdist, gamma)

        vecs = []
        for para in doc:
            for i, sent in enumerate(para):
                vec = []
                # Sentence position in paragraph
                if i == 0:
                    vec.append(1.)
                elif i == len(para) - 1:
                    vec.append(2. if len(para) == 2 else 3.)
                else:
                    vec.append(2.)
                # Number of terms
                vec.append(math.log(len(sent) + 1))
                # Probability of terms in document
                vec.append(sum(math.log(word_pdist.prob(w)) for w in sent))
                # Probability of terms in a baseline document
                if tf is not None:
                    vec.append(sum(math.log(tf[w]) for w in sent if w in tf))
                vecs.append(np.array(vec))
        return vecs

    def _build_transitions(self):
        if self._start_transitions is None:
            self._start_transitions = np.log(
                np.array([self.init_pdist.prob(st) for st in self.states]))

        if self._transitions is None:
            n = len(self.states)
            self._transitions = np.zeros((n, n))
            for i, st_i in enumerate(self.states):
                for j, st_j in enumerate(self.states):
                    try:
                        pdist = self.trans_pdist[st_i]
                    except ValueError:  # state st_i never occurred in training data
                        self._transitions[i, j] = -np.inf
                    else:
                        self._transitions[i, j] = np.log(pdist.prob(st_j))

    def _forward(self, obs: Sequence[np.ndarray]) -> np.ndarray:
        assert all(ob.ndim == 1 and ob.shape[0] == self.emit_pdist.ndim for ob in obs)

        self._build_transitions()

        if not obs:
            assert self._start_transitions is not None
            return self._start_transitions.reshape(1, -1)

        m, n = len(obs), len(self.states)
        # Build emission matrix
        emissions = np.zeros((m, n))
        for i, ob in enumerate(obs):
            for j, st in enumerate(self.states):
                emissions[i, j] = np.log(self.emit_pdist[st].prob(ob))

        alpha = np.zeros((m, n))
        # shape: (n,)
        alpha[0] = self._start_transitions + emissions[0]
        for t in range(1, m):
            # shape: (n, 1)
            a = alpha[t - 1].reshape(-1, 1)
            # shape: (1, n)
            e = emissions[t].reshape(1, -1)
            # shape: (n, n)
            s = a + self._transitions + e
            # shape: (n,)
            alpha[t] = logsumexp(s, axis=0)
        return alpha

    def _backward(self, obs: Sequence[np.ndarray]) -> np.ndarray:
        assert all(ob.ndim == 1 and ob.shape[0] == self.emit_pdist.ndim for ob in obs)

        if not obs:
            return np.zeros((1, len(self.states)))

        self._build_transitions()

        m, n = len(obs), len(self.states)
        # Build emission matrix
        emissions = np.zeros((m, n))
        for i, ob in enumerate(obs):
            for j, st in enumerate(self.states):
                emissions[i, j] = np.log(self.emit_pdist[st].prob(ob))

        beta = np.zeros((m, n))
        for t in range(m - 2, -1, -1):
            # shape: (1, n)
            b = beta[t + 1].reshape(1, -1)
            # shape: (1, n)
            e = emissions[t + 1].reshape(1, -1)
            # shape: (n, n)
            s = self._transitions + e + b
            # shape: (n,)
            beta[t] = logsumexp(s, axis=0)
        return beta

    def _compute_gamma(self, obs: Sequence[np.ndarray]) -> np.ndarray:
        alpha = self._forward(obs)
        beta = self._backward(obs)
        omega = logsumexp(alpha[-1])
        return alpha + beta - omega


class MaxentSummarizer(AbstractSummarizer):
    """Summarizer using maximum entropy classifier (Osborne, 2002).

    There is a difference from the original paper: we put Gaussian prior on the classifier
    weights while the original paper puts the prior on the class labels distribution. This
    difference is fine because our classifier has a bias feature which is able to capture
    the prior class labels distribution from the training data.

    Args:
        classifier (nltk.classify.MaxentClassifier): The underlying classifier object used.
        stopwords (Collection[Word]): Collection of stopwords.
        word_pairs (Collection[Tuple[Word, Word]]): Collection of word pairs, where a word pair
            is defined as two consecutive words found in a sentence.
        trim_length (int): Trim words to this length (measured by the number of characters).

    Attributes:
        STOPWORD_TOKEN (str): Special token for stopwords. Stopwords will be converted into
            this token beforehand.
    """
    STOPWORD_TOKEN = '<stopword>'

    def __init__(self,
                 classifier: MaxentClassifier,
                 stopwords: Optional[Collection[Word]] = None,
                 word_pairs: Optional[Collection[Tuple[Word, Word]]] = None,
                 trim_length: int = 10,
                 ) -> None:
        if stopwords is None:
            stopwords = set()
        if word_pairs is None:
            word_pairs = set()

        self.classifier = classifier
        self.stopwords = stopwords
        self.word_pairs = word_pairs
        self.trim_length = trim_length

    @classmethod
    def train(cls,
              docs: Collection[Document],
              stopwords: Optional[Collection[Word]] = None,
              algorithm: str = 'iis',
              cutoff: int = 4,
              sigma: float = 0.,
              trim_length: int = 10,
              ) -> 'MaxentSummarizer':
        """Train the model on a collection of documents.

        Args:
            docs (Collection[Document]): The collection of documents to train on.
            stopwords (Collection[Word]): Collection of stopwords.
            algorithm (str): Optimization algorithm for training. Possible values are 'iis',
                'gis', or 'megam' (requires `megam`_ to be installed).
            cutoff (int): Features that occur fewer than this value in the training data will
                be discarded.
            sigma (float): Standard deviation for the Gaussian prior. Default is no prior.
            trim_length (int): Trim words to this length.

        Returns:
            MaxEntropy: The trained model.

        .. _megam: https://www.umiacs.umd.edu/~hal/megam/
        """
        if stopwords is None:
            stopwords = set()

        word_pairs = {pair for doc in docs for sent in doc.sentences
                      for pair in cls._get_word_pairs(sent, stopwords, trim_len=trim_length)}

        train_data: list = []
        for doc in docs:
            featuresets = cls._extract_featuresets(doc, stopwords, word_pairs, trim_length)
            labels = [sent.label for sent in doc.sentences]
            train_data.extend(zip(featuresets, labels))

        encoding = BinaryMaxentFeatureEncoding.train(
            train_data, count_cutoff=cutoff, alwayson_features=True)
        classifier = MaxentClassifier.train(
            train_data, algorithm=algorithm, encoding=encoding, gaussian_prior_sigma=sigma)
        return cls(classifier, stopwords=stopwords, word_pairs=word_pairs)

    def summarize(self, doc: Document, size: int = 4) -> List[int]:
        """Summarize a given document.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        featuresets = self._extract_featuresets(
            doc, self.stopwords, self.word_pairs, self.trim_length)
        summary = [k for k, fs in enumerate(featuresets)
                   if self.classifier.classify(fs)][:size]
        return summary

    @classmethod
    def _extract_featuresets(cls,
                             doc: Document,
                             stopwords: Collection[Word],
                             word_pairs: Collection[Tuple[Word, Word]],
                             trim_length: int,
                             ) -> List[dict]:
        featuresets = []
        for i, para in enumerate(doc):
            for j, sent in enumerate(para):
                fs: dict = {}
                # Word pair
                for pair in cls._get_word_pairs(sent, stopwords, trim_length):
                    if pair in word_pairs:
                        fs[f'has-pair({pair[0]},{pair[1]})'] = True
                # Sentence length
                if len(sent) < 6:
                    fs['length'] = 'short'
                elif len(sent) > 20:
                    fs['length'] = 'long'
                # Previous sentence length
                if j > 0 and len(para[j - 1]) < 5:
                    fs['prev-length<5'] = True
                # Sentence position
                if i < 8:
                    fs['pos-para'] = 'first'
                elif i >= len(doc) - 3:
                    fs['pos-para'] = 'last'
                # Limited discourse feat
                if i == 0:
                    fs['para-start'] = True
                featuresets.append(fs)
        return featuresets

    @classmethod
    def _get_word_pairs(cls,
                        sent: Sentence,
                        stopwords: Collection[Word],
                        trim_len: int,
                        ) -> Set[Tuple[Word, Word]]:
        words = [cls.STOPWORD_TOKEN if word in stopwords else word[:trim_len] for word in sent]
        return {(w1, w2) for w1, w2 in zip(words, words[1:])
                if w1 != cls.STOPWORD_TOKEN and w2 != cls.STOPWORD_TOKEN}


class NaiveBayesSummarizer(AbstractSummarizer):
    """Summarizer using naive Bayes method (Aone et al., 1998).

    There is a difference from the original paper: when computing TF-IDF, we operate on word
    token level, while the original paper operates on multi-word tokens that are discovered
    using mutual information.

    Args:
        classifier (nltk.classify.NaiveBayesClassifier): The underlying classifier object used.
        signature_words (Collection[Word]): Collection of words that are deemed important.
    """
    def __init__(self,
                 classifier: NaiveBayesClassifier,
                 signature_words: Optional[Collection[Word]] = None,
                 ) -> None:
        if signature_words is None:
            signature_words = set()

        self.classifier = classifier
        self.signature_words = signature_words

    @classmethod
    def train(cls,
              docs: Collection[Document],
              cutoff: float = 0.1,
              idf_table: Optional[Mapping[Word, float]] = None,
              ) -> 'NaiveBayesSummarizer':
        """Train the model on a collection of documents.

        Args:
            docs (Collection[Document]): The collection of documents to train on.
            cutoff (float): Cutoff for signature words.
            idf_table (Mapping[Word, float]): Precomputed IDF table. If not given, the IDF
                will be computed from ``docs``.

        Returns:
            NaiveBayes: The trained model.
        """
        # Find signature words
        idf = cls._compute_idf(docs) if idf_table is None else idf_table
        n_cutoff = int(cutoff * len(idf))
        signature_words = set(sorted(
            idf.keys(), key=lambda w: idf[w], reverse=True)[:n_cutoff])

        train_data = []  # type: list
        for doc in docs:
            featuresets = cls._extract_featuresets(doc, signature_words)
            labels = [sent.label for sent in doc.sentences]
            train_data.extend(zip(featuresets, labels))
        return cls(
            NaiveBayesClassifier.train(train_data), signature_words=signature_words)

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        featuresets = self._extract_featuresets(doc, self.signature_words)
        summary = [k for k, fs in enumerate(featuresets)
                   if self.classifier.classify(fs)][:size]
        return summary

    @classmethod
    def _extract_featuresets(cls,
                             doc: Document,
                             signature_words: Collection[Word],
                             ) -> List[dict]:
        n_sents = len(doc.sentences)
        featuresets = []
        sent_pos = 0
        for para in doc:
            for i, sent in enumerate(para):
                fs: dict = {
                    # A short sentence
                    'short': len(sent) < 5,
                    # Has signature word
                    'has-signature-word': any(w in signature_words for w in sent),
                }
                # Position in document
                fs['pos-doc'] = math.floor(4 * sent_pos / n_sents)
                # Position in paragraph
                fs['pos-para'] = math.floor(3 * i / len(para))
                featuresets.append(fs)
                sent_pos += 1
        return featuresets

    @staticmethod
    def _compute_idf(docs: Collection[Document]) -> Dict[Word, float]:
        freq_table = defaultdict(int)  # type: ignore
        for doc in docs:
            for word in set(doc.words):
                freq_table[word] += 1
        return {word: math.log(len(docs) / freq_table[word]) for word in freq_table}


class _Gaussian(ProbDistI):
    def __init__(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.mean = mean
        self.cov = cov
        self._rv = multivariate_normal(mean=mean, cov=cov)

    def prob(self, sample: np.ndarray) -> float:
        return self._rv.pdf(sample)

    def max(self) -> np.ndarray:
        return self.mean

    def samples(self) -> list:
        raise NotImplementedError(
            'all samples have non-zero probability in Gaussian distribution')


class _GaussianEmission(ConditionalProbDistI):
    def __init__(self, mean_dict: Dict[int, np.ndarray], cov: np.ndarray) -> None:
        self.mean_dict = mean_dict
        self.cov = cov
        self.update({tag: _Gaussian(mean, cov) for tag, mean in mean_dict.items()})

    @property
    def ndim(self) -> int:
        return len(self.cov)

    @classmethod
    def train(cls, tagged_vecs: Collection[Tuple[np.ndarray, int]]) -> '_GaussianEmission':
        by_tag: dict = defaultdict(list)
        for vec, tag in tagged_vecs:
            by_tag[tag].append(vec)

        mean_dict = {}
        matrices = []
        for tag, vecs in by_tag.items():
            mean = mean_dict[tag] = np.mean(vecs, axis=0)
            for vec in vecs:
                v = (vec - mean).reshape(-1, 1)
                matrices.append(v.dot(v.T))
        cov = np.mean(matrices, axis=0)
        return cls(mean_dict, cov)
