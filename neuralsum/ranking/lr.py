import os
import tensorflow as tf
import numpy as np
from sklearn.linear_model import LogisticRegression as lr
from scipy.spatial.distance import cosine
import json

flags = tf.flags

flags.DEFINE_string ('data_dir',      'data/demo',  'data directory, to compute vocab')
flags.DEFINE_string ('output_dir',    'output',     'output directory, to store summaries')
flags.DEFINE_string ('nn_score_path', 'cv/score',   'a json file storing sentence scores computed with neural model')
flags.DEFINE_boolean('symbolic',       True,        'use symbolic features, e.g., sentence position, length')
flags.DEFINE_boolean('distributional', True,        'use distributional features, e.g., sentence position, length')
flags.DEFINE_string ('embedding_path', 'data',      'emebdding path, which must be specified if distributional=True')
flags.DEFINE_integer('embedding_dim',  50,          'emebdding size')

FLAGS = flags.FLAGS

def load_wordvec(embedding_path):
    '''load word vectors'''

    print ('loading word vectors')
    word_vec = {}
    with open(embedding_path, "r") as f:
        for line in f:
            line = line.rstrip().split(' ')
            word_vec[line[0]] = np.asarray([float(x) for x in line[1:]])
    print ('loading completed')

    return word_vec


def load_nn_score(nn_score_path):
    '''load the output scores predicted by an NN model
       this is a json file, which maps file name to a list of sentence scores'''
    scores = {}
    with open(nn_score_dir, 'r') as f:
        for line in f:
            line = json.loads(line)
            for key, val in line.iteritems():
                scores[key] = val

    return scores


def normalize(lx):
  '''normalize feature vectors in a small subset'''
  nsamples, nfeatures = len(lx), len(lx[0])
  for i in range(nfeatures):
    column = []
    for j in range(nsamples):
      column.append(lx[j][i])
    total = sum(column)
    for j in range(nsamples):
      if total!=0: lx[j][i] = lx[j][i] / total
  return lx


class Sybolic_Extractor(object):
    '''extract symbolic features: sentence length, position, entity counts
       We normalize all features.'''

    def __init__(self, etype='symbolic'):
        self.etype = etype
     
    @staticmethod 
    def length(sen):
        return len(sen)

    @staticmethod 
    def ent_count(sen):
        return sen.count('entity') 

    def extract_feature(self, sen_list):
        features = []
        for sid, sen in enumerate(sen_list):
            sen_feature = [sid, self.length(sen), self.ent_count(sen)]
            features.append(sen_feature) 

        return features


class Distributional_Extractor(object):
    '''extract distributional features: 
           sentence similary with respect to document
           sentence similary with respect to other sentences
       We normalize all features.'''

    def __init__(self, etype='distributional'):
        self.etype = etype

    @staticmethod 
    def compute_sen_vec(sen, word_vec):
        sen_vec = np.zeros(FLAGS.embedding_dim)
        count = 0
        for word in sen.split(' '):
            if word_vec.has_key(word):
                sen_vec += word_vec[word]
                count += 1
        if count > 0:
            sen_vec = sen_vec / count
       
        return sen_vec

    @staticmethod 
    def reduncy(sen_vec, doc_vec):
        return 1 - cosine(sen_vec, (doc_vec - sen_vec))

    @staticmethod 
    def relavence(sen_vec, doc_vec): 
        return 1 - cosine(sen_vec, doc_vec)

    def extract_feature(self, sen_list, word_vec):
        features = []
        sen_vec_list = []
        for sen in sen_list:
            sen_vec_list.append(self.compute_sen_vec(sen, word_vec))

        doc_vec = sum(sen_vec_list)       

        for sen_vec in sen_vec_list:
            sen_feature = [self.reduncy(sen_vec, doc_vec), self.relavence(sen_vec, doc_vec)]
            features.append(sen_feature)

        return features


def train_and_test():
    '''train and test a logistic regression classifier, which uses other features'''

    sExtractor = Sybolic_Extractor()
    dExtractor = Distributional_Extractor()

    word_vec = load_wordvec(FLAGS.embedding_path)

    nn_scores = load_nn_score(FLAGS.nn_score_path)

    train_x, train_y = [], []

    train_dir = os.path.join(FLAGS.data_dir, 'train')
    train_files = os.listdir(train_dir)

    for input_file in train_files:
        input_dir = os.path.join(train_dir, input_file)
        fp = open(input_dir, 'r')
        lines = fp.read().split('\n\n')
        sentences = lines[1].split('\n')
        sens = [sen.split('\t\t\t')[0] for sen in sentences]
        y = [int(sen.split('\t\t\t')[1]) for sen in sentences] 

        x_n = nn_scores[input_file]
        x_s = sExtractor.extract_feature(sens)
        x_d = dExtractor.extract_feature(sens, word_vec)
        x = [[f1] + f2 + f3 for f1, f2, f3 in zip(x_n, x_s, x_d)] 
        x = normalize(x)

        train_x.extend(x)
        train_y.extend(y)

        fp.close()

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)

    my_lr = lr()
    my_lr.fit(train_x, train_y)

    print ('testing...')

    test_dir = os.path.join(FLAGS.data_dir, 'test')
    test_files = os.listdir(test_dir)

    for input_file in test_files:
        input_dir = os.path.join(test_dir, input_file)
        fp = open(input_dir, 'r')
        lines = fp.read().split('\n\n')
        sentences = lines[1].split('\n')
        sens = [sen.split('\t\t\t')[0] for sen in sentences]

        x_n = nn_scores[input_file]
        x_s = sExtractor.extract_feature(sens)
        x_d = dExtractor.extract_feature(sens, word_vec)
        test_x = [[f1] + f2 + f3 for f1, f2, f3 in zip(x_n, x_s, x_d)] 
        test_x = normalize(test_x)

        fp.close()

        score = my_lr.predict_proba(np.asarray(test_x))
        # we need score for the postive classes
        sen_score = {}
        for sid, sentence in enumerate(sens):
            sen_score[sentence] = score[sid][1] + 0.5 * score[sid][2]

        sorted_sen = sorted(sen_score.items(), key=lambda d: d[1], reverse=True)  
        selected = [s[0] for s in sorted_sen[:3]]

        # store selected sentences to output file, following the original order
        file_name = '.'.join(input_file.split('.')[:-1]) + '.output'

        output_fp = open(os.path.join(FLAGS.output_dir, file_name), 'w')
        for sen in sens:
            if sen in selected:
                output_fp.write(sen + '\n')
        output_fp.close()


if __name__ == "__main__":
    train_and_test()

