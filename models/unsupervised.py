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

from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional
import math

import networkx as nx
import numpy as np

from data import Document, Sentence, Word
from models import AbstractSummarizer


class Lead(AbstractSummarizer):
    """Summarizer implementing LEAD-N algorithm.

    LEAD-N simply selects leading N sentences as the summary.
    """

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document using LEAD-N algorithm.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        return list(range(min(len(doc.sentences), size)))


class LexRank(AbstractSummarizer):
    """Summarizer implementing LexRank algorithm (Erkan and Radev, 2004).

    The original LexRank removes sentences with length less than a certain threshold. This
    implementation does NOT perform this removal.

    LexRank algorithm uses cross-sentence informational subsumption (CSIS) in sentence
    selection stage. The referred paper for CSIS (Radev, 2000) does not explain clearly
    how to do CSIS. We found another paper by Radev et al. (2000) discussing CSIS and
    propose an approximation to it, called cross-sentence word overlap (CSWO). This
    approximation is what we use here.

    Args:
        idf_table: (Mapping[Word, float]): Precomputed inverse document frequency table. If not
            given, this table will be computed from the input sentences.
        continuous (bool): Whether to perform continuous LexRank instead. In continous LexRank,
            there is a weighted edge between each node/sentence. The weight is set to the
            cosine similarity of the two sentences.
        similarity_threshold (float): Threshold for the cosine similarity. An edge between
            two sentences will be drawn if their similarity exceeds this value. This argument
            is ignored if ``continuous=True``.
        weight (float): The weight for the LexRank feature.
        cswo_threshold (float): Threshold for the cross-sentence word overlap, an approximation
            for the cross-sentence informational subsumption. A sentence will be included in
            the summary if its CSWO scores with all the sentences already in the summary are
            less than this value.
        damping_factor (float): The damping factor for the PageRank algorithm.
        tol (float): Tolerance to test for convergence in running the PageRank algorithm.
        max_iter (int): Maximum number of iterations for the PageRank algorithm.
    """
    def __init__(self,
                 idf_table: Optional[Mapping[Word, float]] = None,
                 continuous: bool = False,
                 similarity_threshold: float = 0.1,
                 weight: float = 0.5,
                 cswo_threshold: float = 0.5,
                 damping_factor: float = 0.85,
                 tol: float = 1e-6,
                 max_iter: int = 100,
                 ) -> None:
        self.idf_table = idf_table
        self.continuous = continuous
        self.similarity_threshold = similarity_threshold
        self.weight = weight
        self.cswo_threshold = cswo_threshold
        self.damping_factor = damping_factor
        self.tol = tol
        self.max_iter = max_iter

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document using LexRank algorithm.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        positions = [self._get_position(k, len(doc.sentences))
                     for k in range(len(doc.sentences))]
        G = self._build_graph(doc.sentences)
        ranks = nx.pagerank(G, alpha=self.damping_factor, tol=self.tol, max_iter=self.max_iter)
        candidates = sorted(
            ranks.keys(), key=lambda k: self._combine_features(positions[k], ranks[k]),
            reverse=True)
        return self._csis(doc.sentences, candidates, size)

    @staticmethod
    def _get_position(sent_idx: int, num_sents: int) -> float:
        return 1. - (sent_idx + 1) / num_sents

    def _combine_features(self, position: float, rank: float) -> float:
        return position + self.weight * rank

    def _build_graph(self, sents: List[Sentence]) -> nx.DiGraph:
        if self.idf_table is None:
            # Fallback to IDF computed from sentences
            self.idf_table = self._compute_idf(sents)

        G = nx.DiGraph()
        for x, sent_x in enumerate(sents):
            for y, sent_y in enumerate(sents):
                if x == y:
                    continue
                similarity = self._cosine_similarity(sent_x, sent_y)
                if self.continuous:
                    G.add_edge(x, y, weight=similarity)
                elif similarity > self.similarity_threshold:
                    G.add_edge(x, y)
        return G

    @staticmethod
    def _compute_idf(sents: List[Sentence]) -> Dict[Word, float]:
        freq_table = defaultdict(int)  # type: ignore
        for sent in sents:
            for word in set(sent):
                freq_table[word] += 1
        return {word: math.log(len(sents) / freq_table[word]) for word in freq_table}

    def _cosine_similarity(self,
                           sent_x: Sentence,
                           sent_y: Sentence,
                           ) -> float:
        assert self.idf_table is not None
        words_x = [w for w in sent_x if w in self.idf_table]
        words_y = [w for w in sent_y if w in self.idf_table]
        tf_x, tf_y = Counter(words_x), Counter(words_y)
        set_x, set_y = set(words_x), set(words_y)
        common = set_x.intersection(set_y)

        numerator = sum(tf_x[w] * tf_y[w] * self.idf_table[w]**2 for w in common)
        len_x = math.sqrt(sum(tf_x[w]**2 * self.idf_table[w]**2 for w in set_x))
        len_y = math.sqrt(sum(tf_y[w]**2 * self.idf_table[w]**2 for w in set_y))
        denominator = len_x * len_y
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return 0.

    def _csis(self, sents: List[Sentence], candidates: List[int], size: int) -> List[int]:
        candidates = candidates[::-1]
        summary: List[int] = []

        while candidates and len(summary) < size:
            candidate = candidates.pop()
            assert candidate not in summary
            if all(self._cswo(sents, candidate, s) < self.cswo_threshold for s in summary):
                summary.append(candidate)
        summary.sort()
        return summary

    @staticmethod
    def _cswo(sents: List[Sentence], s_idx: int, t_idx: int) -> float:
        sent_s, sent_t = sents[s_idx], sents[t_idx]
        tf_s, tf_t = Counter(sent_s), Counter(sent_t)
        common = set(sent_s).intersection(set(sent_t))
        numerator = 2 * sum(min(tf_s[w], tf_t[w])for w in common)
        denominator = len(sent_s) + len(sent_t)
        return numerator / denominator


class LSA(AbstractSummarizer):
    """Summarizer using LSA algorithm (Gong and Liu, 2001; Steinberger and Jezek, 2004).

    Args:
        algorithm (str): The LSA summarization algorithm to use. Set to 'gong' for
            (Gong and Liu, 2001), or 'steinberger' for (Steinberger and Jezek, 2004).
    """
    ALGORITHMS = ('gong', 'steinberger')

    def __init__(self, algorithm: str = 'gong') -> None:
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f'invalid algorithm for LSA: {algorithm}')

        self.algorithm = algorithm

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document using LSA algorithm.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        matrix = self._create_matrix(doc)
        _, s, vh = np.linalg.svd(matrix)

        if self.algorithm == 'gong':
            summary = self._build_summary_gong(vh, size)
        else:
            assert self.algorithm == 'steinberger'
            summary = self._build_summary_steinberger(s, vh, size)
        summary.sort()
        return summary

    @staticmethod
    def _create_matrix(doc: Document) -> np.ndarray:
        word2id = {word: i for i, word in enumerate(set(doc.words))}
        n_words, n_sents = len(word2id), len(doc.sentences)

        matrix = np.zeros((n_words, n_sents))
        for j, sent in enumerate(doc.sentences):
            for word in set(sent):
                matrix[word2id[word], j] = 1.
        return matrix

    @staticmethod
    def _build_summary_gong(vh: np.ndarray, size: int) -> List[int]:
        summary: List[int] = []
        while len(summary) < size:
            row = vh[len(summary)]
            best_idx = int(np.argmax(row))
            summary.append(best_idx)
            vh[:, best_idx] = -float('inf')  # prevent this sentence from being reselected
        return summary

    @staticmethod
    def _build_summary_steinberger(s: np.ndarray, vh: np.ndarray, size: int) -> List[int]:
        _, n_sents = vh.shape

        # Find the best number of dimensions
        dim = 1
        while dim < len(s) and s[dim] >= 0.5 * s[0]:
            dim += 1

        s, vh = s[:dim], vh[:dim, :]

        ranks = np.linalg.norm(s[:, np.newaxis] * vh, ord=2, axis=0)
        summary = sorted(range(n_sents), key=lambda i: ranks[i], reverse=True)[:size]
        return summary


class SumBasic(AbstractSummarizer):
    """Summarizer implementing SumBasic algorithm (Nenkova and Vanderwende, 2005)."""

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document using SumBasic algorithm.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        probs = self._compute_word_probs(doc)

        summary: List[int] = []
        while len(summary) < size:
            highest_prob_word = max(probs.keys(), key=lambda w: probs[w])
            best_idx = max(
                [idx for idx, sent in enumerate(doc.sentences) if highest_prob_word in sent],
                key=lambda idx: self._score_sentence(doc.sentences[idx], probs))
            summary.append(best_idx)
            for word in doc.sentences[best_idx]:
                probs[word] = probs[word]**2
        summary.sort()
        return summary

    def _compute_word_probs(self, doc: Document) -> Dict[Word, float]:
        counter = Counter(doc.words)
        n_words = sum(counter.values())
        return {word: freq / n_words for word, freq in counter.items()}

    @staticmethod
    def _score_sentence(sentence: Sentence, probs: Dict[Word, float]) -> float:
        return sum(probs[word] for word in sentence) / len(sentence)


class TextRank(AbstractSummarizer):
    """Summarizer implementing TextRank algorithm (Mihalcea and Tarau, 2004).

    Args:
        damping_factor (float): The damping factor for the PageRank algorithm.
        tol (float): Tolerance to test for convergence in running the PageRank algorithm.
        max_iter (int): Maximum number of iterations for the PageRank algorithm.
    """
    def __init__(self,
                 damping_factor: float = 0.85,
                 tol: float = 1e-6,
                 max_iter: int = 100,
                 ) -> None:
        self.damping_factor = damping_factor
        self.tol = tol
        self.max_iter = max_iter

    def summarize(self, doc: Document, size: int = 3) -> List[int]:
        """Summarize a given document using TextRank algorithm.

        Args:
            doc (Document): The document to summarize.
            size (int): Maximum number of sentences that the summary should have.

        Returns:
            list: The indices of the extracted sentences that form the summary, sorted
                ascending.
        """
        size = min(size, len(doc.sentences))
        G = self._build_graph(doc.sentences)
        ranks = nx.pagerank(G, alpha=self.damping_factor, tol=self.tol, max_iter=self.max_iter)

        summary = sorted(ranks.keys(), key=lambda k: ranks[k], reverse=True)[:size]
        summary.sort()
        return summary

    def _build_graph(self, sents: List[Sentence]) -> nx.DiGraph:
        G = nx.DiGraph()
        for x, sent_x in enumerate(sents):
            for y, sent_y in enumerate(sents):
                if x == y:
                    continue
                G.add_edge(x, y, weight=self._compute_weight(sent_x, sent_y))
        return G

    @staticmethod
    def _compute_weight(sent_x: Sentence, sent_y: Sentence) -> float:
        common = set(sent_x).intersection(set(sent_y))
        eps = 1e-6
        return len(common) / (math.log(len(sent_x)) + math.log(len(sent_y)) + eps)
