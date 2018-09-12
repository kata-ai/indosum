from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from utils import argmax, random, topk, constrained

import model_abs as model
from data_reader import load_data_abs, DataReader_abs


flags = tf.flags

# data
flags.DEFINE_string ('load_model',    None,     'filename of the model to load')
flags.DEFINE_string ('data_dir',      'data/demo',   'data directory, to compute vocab')
flags.DEFINE_string ('train_dir',     'cv',          'training directory (models and summaries are saved there periodically)')
flags.DEFINE_boolean('use_abs',       False,     'do we use human summaries or the selected sentences as the target')

# model params
flags.DEFINE_string ('model_choice',    'bilstm',                       'model choice')
flags.DEFINE_integer('rnn_size',        650,                            'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('word_embed_size', 50,                             'dimensionality of word embeddings')
flags.DEFINE_string ('kernels',         '[1,2,3,4,5,6,7]',              'CNN kernel widths')
flags.DEFINE_string ('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')

# generation choice
flags.DEFINE_integer('batch_size',          20,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_doc_length',      15,       'maximum document length')
flags.DEFINE_integer('max_sen_length',      50,       'maximum sentence length')
flags.DEFINE_integer('max_output_length',   100,      'maximum word allowed in the summary')
flags.DEFINE_float  ('temperature',         1.0,      'sampling temperature')
flags.DEFINE_string ('decode_choice',       'constrained',   'decode choice (argmax, beam or constrained)')

# bookkeeping
flags.DEFINE_integer('seed',           3435, 'random number generator seed')
flags.DEFINE_integer('print_every',    5,    'how often to print current loss')
flags.DEFINE_string ('EOS',            '+',  '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

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


def build_model(word_vocab, target_vocab, max_doc_length, max_output_length):
    my_model = None
    if FLAGS.model_choice == 'bilstm':
            my_model = model.cnn_sen_enc(
                    word_vocab_size=word_vocab.size,
                    word_embed_size=FLAGS.word_embed_size,
                    batch_size=1,
                    num_highway_layers=FLAGS.highway_layers,
                    max_sen_length=FLAGS.max_sen_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    max_doc_length=FLAGS.max_doc_length)

            my_model.update(model.bilstm_doc_enc(my_model.input_cnn,
                                           batch_size=1,
                                           num_rnn_layers=FLAGS.rnn_layers,
                                           rnn_size=FLAGS.rnn_size,
                                           max_doc_length=max_doc_length,
                                           dropout=0.0))

            my_model.update(model.vanilla_attention_decoder(my_model.enc_outputs,
                                       batch_size=1,
                                       num_rnn_layers=FLAGS.rnn_layers,
                                       rnn_size=FLAGS.rnn_size,
                                       enc_state_size=FLAGS.rnn_size * 2,
                                       max_output_length=max_output_length,
                                       dropout=0.0,
                                       word_vocab_size=target_vocab.size,
                                       word_embed_size=FLAGS.word_embed_size,
                                       mode='decode'))


    return my_model


def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, word_tensors, max_doc_length, target_vocab, target_tensors, max_output_length = \
      load_data_abs(FLAGS.data_dir, FLAGS.max_doc_length, FLAGS.max_sen_length, FLAGS.max_output_length, FLAGS.use_abs)

    print('initialized test dataset reader')

    test_reader = DataReader_abs(word_tensors['test'], target_tensors['test'],
                              FLAGS.batch_size)

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = build_model(word_vocab, target_vocab, max_doc_length, 1)
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model)

        save_as = '%s/abstractions' % (FLAGS.train_dir)
        save_file = open(save_as, 'w')

        for x, y in test_reader.iter():
            for i in range(FLAGS.batch_size):
                xi, yi = x[[i], :, :], y[[i], :]

                predicted_yi = yi[:, [0]]
                last_ix = -1 # used in constrained          

                rnn_state = session.run(m.initial_dec_state) 
                for i in range(max_output_length):
                
                    # this is slow, fix it  
                    logits, rnn_state = session.run([m.logits, m.final_dec_state],
                                                  {m.input_dec: predicted_yi,
                                                   m.input  : xi,
                                                   m.initial_dec_state: rnn_state})

                    logits = np.array(logits)
                    if FLAGS.decode_choice == 'argmax':
                        ix = argmax(logits)
                    elif FLAGS.decode_choice == 'constrained':
                        ix = constrained(logits, xi, last_ix)

                    predicted_yi = np.zeros((1, 1))
                    predicted_yi[0, 0] = ix
                    predicted_word = word_vocab.token(ix)
                    last_ix = ix
                    save_file.write(predicted_word + ' ')

                save_file.write('\n' + ' ')
    
        save_file.close()

if __name__ == "__main__":
    tf.app.run()
