from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from utils import softmax
import model
from data_reader import load_data, DataReader


flags = tf.flags

# data
flags.DEFINE_string('data_dir',    'data/demo',   'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',   'cv',          'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,         '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_string('model_choice',     'lstm',                         'model choice')
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('word_embed_size', 50,                             'dimensionality of word embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')

# optimization
flags.DEFINE_integer('batch_size',        20,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_doc_length',    15,   'max_doc_length')
flags.DEFINE_integer('max_sen_length',    50,   'maximum sentence length')
flags.DEFINE_float  ('weight_2',          0.5,  'how much do we count about label 2')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')

FLAGS = flags.FLAGS


def run_test(session, m, data, batch_size, num_steps):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)

    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })

        costs += cost
        iters += 1

    return costs / iters

def build_model(word_vocab):
    if FLAGS.model_choice == 'bilstm':
        valid_model = model.cnn_sen_enc(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_sen_length=FLAGS.max_sen_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length)

        valid_model.update(model.bilstm_doc_enc(valid_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=0.0))

        valid_model.update(model.label_prediction(valid_model.enc_outputs))
        valid_model.update(model.loss_extraction(valid_model.logits, FLAGS.batch_size, FLAGS.max_doc_length)) 
    elif FLAGS.model_choice == 'lstm':
        valid_model = model.cnn_sen_enc(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    max_sen_length=FLAGS.max_sen_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length)

        valid_model.update(model.lstm_doc_enc(valid_model.input_cnn,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=0.0))

        valid_model.update(model.lstm_doc_dec(valid_model.input_cnn, valid_model.final_enc_state,
                                           batch_size=FLAGS.batch_size,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=FLAGS.max_doc_length,
                                           dropout=0.0))

        valid_model.update(model.label_prediction_att(valid_model.enc_outputs, valid_model.dec_outputs))
        valid_model.update(model.loss_extraction(valid_model.logits, FLAGS.batch_size, FLAGS.max_doc_length))

    return valid_model


def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + ".index"):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, word_tensors, max_doc_length, label_tensors = \
        load_data(FLAGS.data_dir, FLAGS.max_doc_length, FLAGS.max_sen_length)

    test_reader = DataReader(word_tensors['test'], label_tensors['test'],
                              FLAGS.batch_size)

    print('initialized test dataset reader')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = build_model(word_vocab)
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())

        ''' training starts here '''
        count = 0
        start_time = time.time()
        result_scores = None
        for x, y in test_reader.iter():
            count += 1
            logits = session.run(
                m.logits,
            {
                m.input  : x,
                m.targets: y
            })

            total_scores = []
            for tid, tlogits in enumerate(logits):
                scores = softmax(tlogits)
                weights = np.array([0, 1, 0.5])
                scores = np.dot(scores, weights)
                total_scores.append(scores)
 
            total_scores = np.transpose(np.asarray(total_scores))
            if result_scores is None:
                result_scores = total_scores
            else:
                result_scores = np.vstack((result_scores, total_scores)) 
        
        save_as = '%s/scores' % (FLAGS.train_dir)
        np.savetxt(save_as, result_scores, delimiter=' ')
        time_elapsed = time.time() - start_time
 
        print("test samples:", count*FLAGS.batch_size, "time elapsed:", time_elapsed, "time per one batch:", time_elapsed/count)

        
if __name__ == "__main__":
    tf.app.run()
