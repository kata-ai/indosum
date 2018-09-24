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

from typing import Any

from camel import PYTHON_TYPES, Camel, CamelRegistry
from nltk.classify import BinaryMaxentFeatureEncoding, MaxentClassifier, NaiveBayesClassifier
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ELEProbDist, FreqDist, \
    LidstoneProbDist
import numpy as np

from models.supervised import HMMSummarizer, MaxentSummarizer, NaiveBayesSummarizer, \
    _GaussianEmission


registry = CamelRegistry()


@registry.dumper(HMMSummarizer, 'hmm', version=1)
def _dump_hmm(model: HMMSummarizer) -> dict:
    return {
        'init_pdist': model.init_pdist,
        'trans_pdist': model.trans_pdist,
        'emit_pdist': model.emit_pdist,
        'states': model.states,
        'gamma': model.gamma,
        'tf_table': model.tf_table,
    }


@registry.loader('hmm', version=1)
def _load_hmm(data: dict, version: int) -> HMMSummarizer:
    return HMMSummarizer(
        data['init_pdist'], data['trans_pdist'], data['emit_pdist'], data['states'],
        gamma=data['gamma'], tf_table=data['tf_table'])


@registry.dumper(MaxentSummarizer, 'maxent', version=1)
def _dump_maxent(model: MaxentSummarizer) -> dict:
    return {
        'classifier': model.classifier,
        'stopwords': model.stopwords,
        'word_pairs': model.word_pairs,
        'trim_length': model.trim_length,
    }


@registry.loader('maxent', version=1)
def _load_maxent(data: dict, version: int) -> MaxentSummarizer:
    return MaxentSummarizer(
        data['classifier'], stopwords=data['stopwords'], word_pairs=data['word_pairs'],
        trim_length=data['trim_length'])


@registry.dumper(NaiveBayesSummarizer, 'bayes', version=1)
def _dump_bayes(model: NaiveBayesSummarizer) -> dict:
    return {
        'classifier': model.classifier,
        'signature_words': model.signature_words,
    }


@registry.loader('bayes', version=1)
def _load_bayes(data: dict, version: int) -> NaiveBayesSummarizer:
    return NaiveBayesSummarizer(data['classifier'], signature_words=data['signature_words'])


@registry.dumper(LidstoneProbDist, 'lidstone_pdist', version=1)
def _dump_lidstone_pdist(pdist: LidstoneProbDist) -> dict:
    return {
        'freqdist': pdist.freqdist(),
        'gamma': pdist._gamma,
        'bins': pdist._bins,
    }


@registry.loader('lidstone_pdist', version=1)
def _load_lidstone_pdist(data: dict, version: int) -> LidstoneProbDist:
    return LidstoneProbDist(data['freqdist'], data['gamma'], bins=data['bins'])


@registry.dumper(FreqDist, 'fdist', version=1)
def _dump_fdist(fdist: FreqDist) -> dict:
    return dict(fdist.items())


@registry.loader('fdist', version=1)
def _load_fdist(data: dict, version: int) -> FreqDist:
    fdist = FreqDist()
    for k, v in data.items():
        fdist[k] += v
    return fdist


@registry.dumper(ConditionalProbDist, 'cpdist', version=1)
def _dump_cpdist(cpdist: ConditionalProbDist) -> dict:
    cfdist = ConditionalFreqDist()
    for cond in cpdist.conditions():
        for k, v in cpdist[cond].freqdist().items():
            cfdist[cond][k] += v

    return {
        'cfdist': cfdist,
        'factory_args': cpdist._factory_args,
        'factory_kw_args': cpdist._factory_kw_args,
    }


@registry.loader('cpdist', version=1)
def _load_cpdist(data: dict, version: int) -> ConditionalProbDist:
    return ConditionalProbDist(
        data['cfdist'], LidstoneProbDist, *data['factory_args'], **data['factory_kw_args'])


@registry.dumper(ConditionalFreqDist, 'cfdist', version=1)
def _dump_cfdist(cfdist: ConditionalFreqDist) -> dict:
    data: dict = {}
    for cond in cfdist.conditions():
        for k, v in cfdist[cond].items():
            if cond not in data:
                data[cond] = {}
            if k not in data[cond]:
                data[cond][k] = 0
            data[cond][k] += v
    return data


@registry.loader('cfdist', version=1)
def _load_cfdist(data: dict, version: int) -> ConditionalFreqDist:
    cfdist = ConditionalFreqDist()
    for cond in data:
        for k, v in data[cond].items():
            cfdist[cond][k] += v
    return cfdist


@registry.dumper(_GaussianEmission, 'gaussian_emission', version=1)
def _dump_gaussian_emission(em: _GaussianEmission) -> dict:
    return {
        'mean_dict': {k: v.tolist() for k, v in em.mean_dict.items()},
        'cov': em.cov.tolist(),
    }


@registry.loader('gaussian_emission', version=1)
def _load_gaussian_emission(data: dict, version: int) -> _GaussianEmission:
    mean_dict = {k: np.array(v) for k, v in data['mean_dict'].items()}
    return _GaussianEmission(mean_dict, np.array(data['cov']))


@registry.dumper(MaxentClassifier, 'maxent_clf', version=1)
def _dump_maxent_clf(clf: MaxentClassifier) -> dict:
    return {
        'encoding': clf._encoding,
        'weights': clf._weights.tolist(),
        'logarithmic': clf._logarithmic,
    }


@registry.loader('maxent_clf', version=1)
def _load_maxent_clf(data: dict, version: int) -> MaxentClassifier:
    return MaxentClassifier(
        data['encoding'], np.array(data['weights']), logarithmic=data['logarithmic'])


@registry.dumper(BinaryMaxentFeatureEncoding, 'binary_enc', version=1)
def _dump_binary_enc(enc: BinaryMaxentFeatureEncoding) -> dict:
    return {
        'labels': enc._labels,
        'mapping': enc._mapping,
        'unseen_features': enc._unseen is not None,
        'alwayson_features': enc._alwayson is not None,
    }


@registry.loader('binary_enc', version=1)
def _load_binary_enc(data: dict, version: int) -> BinaryMaxentFeatureEncoding:
    return BinaryMaxentFeatureEncoding(
        data['labels'], data['mapping'], unseen_features=data['unseen_features'],
        alwayson_features=data['alwayson_features'])


@registry.dumper(NaiveBayesClassifier, 'bayes_clf', version=1)
def _dump_bayes_clf(clf: NaiveBayesClassifier) -> dict:
    return {
        'label_pdist': clf._label_probdist,
        'feature_pdist': clf._feature_probdist,
    }


@registry.loader('bayes_clf', version=1)
def _load_bayes_clf(data: dict, version: int) -> NaiveBayesClassifier:
    return NaiveBayesClassifier(data['label_pdist'], data['feature_pdist'])


@registry.dumper(ELEProbDist, 'ele_pdist', version=1)
def _dump_ele_pdist(pdist: ELEProbDist) -> dict:
    return {
        'freqdist': pdist.freqdist(),
        'bins': pdist._bins,
    }


@registry.loader('ele_pdist', version=1)
def _load_ele_pdist(data: dict, version: int) -> ELEProbDist:
    return ELEProbDist(data['freqdist'], bins=data['bins'])


def dump(obj: Any) -> str:
    return Camel([registry, PYTHON_TYPES]).dump(obj)


def load(data: str) -> Any:
    return Camel([registry, PYTHON_TYPES]).load(data)
