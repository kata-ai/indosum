from __future__ import print_function
from __future__ import division

import tensorflow as tf
from utils import adict


def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID') + b


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, scope='TDNN'):
    '''
    :input:           input float tensor of shape [(batch_size*max_doc_length) x max_sen_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    max_sen_length = input_.get_shape()[1]
    embed_size = input_.get_shape()[-1]

    # input_: [batch_size*max_doc_length, 1, max_sen_length, embed_size]
    input_ = tf.expand_dims(input_, 1)

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_sen_length - kernel_size + 1

            # [batch_size x max_sen_length x embed_size x kernel_feature_size]
            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name="kernel_%d" % kernel_size)

            # [batch_size x 1 x 1 x kernel_feature_size]
            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1], [1, 1, 1, 1], 'VALID')

            layers.append(tf.squeeze(pool, [1, 2]))

        if len(kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]

    return output


def cnn_sen_enc(word_vocab_size,
                    word_embed_size=50,
                    batch_size=20,
                    num_highway_layers=2,
                    max_sen_length=65,
                    kernels         = [ 1,   2,   3,   4,   5,   6,   7],
                    kernel_features = [50, 100, 150, 200, 200, 200, 200],
                    max_doc_length=35,
                    pretrained=None):

    # cnn sentence encoder
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    input_ = tf.placeholder(tf.int32, shape=[batch_size, max_doc_length, max_sen_length], name="input")

    ''' First, embed words to sentence '''
    with tf.variable_scope('Embedding'):
        if pretrained is not None:
            word_embedding = tf.get_variable(name='word_embedding', shape=[word_vocab_size, word_embed_size],
                                       initializer=tf.constant_initializer(pretrained))
        else:
            word_embedding = tf.get_variable(name='word_embedding', shape=[word_vocab_size, word_embed_size])

        ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
        of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
        zero embedding vector and ignores gradient updates. For that do the following in TF:
        1. after parameter initialization, apply this op to zero out padding embedding vector
        2. after each gradient update, apply this op to keep padding at zero'''
        clear_word_embedding_padding = tf.scatter_update(word_embedding, [0], tf.constant(0.0, shape=[1, word_embed_size]))

        # [batch_size, max_doc_length, max_sen_length, word_embed_size]
        input_embedded = tf.nn.embedding_lookup(word_embedding, input_)

        input_embedded = tf.reshape(input_embedded, [-1, max_sen_length, word_embed_size])

    ''' Second, apply convolutions '''
    # [batch_size x max_doc_length, cnn_size]  # where cnn_size=sum(kernel_features)
    input_cnn = tdnn(input_embedded, kernels, kernel_features)

    ''' Maybe apply Highway '''
    if num_highway_layers > 0:
        input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers=num_highway_layers)

    return adict(
        input = input_,
        clear_word_embedding_padding=clear_word_embedding_padding,
        input_embedded=input_embedded,
        input_cnn=input_cnn
    )


def bilstm_doc_enc(input_cnn,
                       batch_size=20,
                       num_rnn_layers=2,
                       rnn_size=650,
                       max_doc_length=35,
                       dropout=0.0):

    # bilstm document encoder
    with tf.variable_scope('BILSTMenc'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell

        if num_rnn_layers > 1:
            cell_fw = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
            cell_bw = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell_fw = create_rnn_cell()
            cell_bw = create_rnn_cell()

        initial_rnn_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
        initial_rnn_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)

        input_cnn = tf.reshape(input_cnn, [batch_size, max_doc_length, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, max_doc_length, 1)]

        outputs, final_rnn_state_fw, final_rnn_state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, input_cnn2,
                                         initial_state_fw=initial_rnn_state_fw, initial_state_bw=initial_rnn_state_bw, dtype=tf.float32)

    return adict(
        initial_enc_state_fw=initial_rnn_state_fw,
        initial_enc_state_bw=initial_rnn_state_bw,
        final_enc_state_fw=final_rnn_state_fw,
        final_enc_state_bw=final_rnn_state_bw,
        enc_outputs=outputs
    )


def lstm_doc_enc(input_cnn,
                   batch_size=20,
                   num_rnn_layers=2,
                   rnn_size=650,
                   max_doc_length=35,
                   dropout=0.0):

    # lstm document encoder
    with tf.variable_scope('LSTMenc'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell

        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()

        initial_rnn_state = cell.zero_state(batch_size, dtype=tf.float32)

        input_cnn = tf.reshape(input_cnn, [batch_size, max_doc_length, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, max_doc_length, 1)]

        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, input_cnn2,
                                         initial_state=initial_rnn_state, dtype=tf.float32)

    return adict(
        initial_enc_state=initial_rnn_state,
        final_enc_state=final_rnn_state,
        enc_outputs=outputs
    )


def lstm_doc_dec(input_cnn, final_enc_state,
                   batch_size=20,
                   num_rnn_layers=2,
                   rnn_size=650,
                   max_doc_length=35,
                   dropout=0.0):

    # scoring each sentence with another LSTM that reads the doc again
    with tf.variable_scope('LSTMdec'):
        def create_rnn_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=0.0)
            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-dropout)
            return cell

        if num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell() for _ in range(num_rnn_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell()

        initial_rnn_state = final_enc_state

        input_cnn = tf.reshape(input_cnn, [batch_size, max_doc_length, -1])
        input_cnn2 = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, max_doc_length, 1)]

        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell, input_cnn2,
                                         initial_state=initial_rnn_state, dtype=tf.float32)

    return adict(
        initial_dec_state=initial_rnn_state,
        final_dec_state=final_rnn_state,
        dec_outputs=outputs
    )



def self_prediction(outputs, word_vocab_size):

    # predicting the words in therein, like a paragraph vector
    logits_pretrain = []
    with tf.variable_scope('SelfPrediction') as scope:
        for idx, output in enumerate(outputs):
            if idx > 0:
                scope.reuse_variables()
            logits_pretrain.append(linear(output, word_vocab_size))

    return adict(
        plogits = logits_pretrain
    )


def label_prediction(outputs):

    # scoring labels
    logits = []
    with tf.variable_scope('Prediction') as scope:
        for idx, output in enumerate(outputs):
            if idx > 0:
                scope.reuse_variables()
            logits.append(linear(output, 2))

    return adict(
        logits = logits
    )


def label_prediction_att(outputs_enc, outputs_dec):

    # scoring labels with att
    logits = []
    with tf.variable_scope('Prediction') as scope:
        for idx, output in enumerate(zip(outputs_enc, outputs_dec)):
            if idx > 0:
                scope.reuse_variables()
            output_enc, output_dec = output
            logits.append(linear(tf.concat([output_enc, output_dec], 1), 2))
    return adict(
        logits = logits
    )


def loss_extraction(logits, batch_size, max_doc_length):

    # extraction loss
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, max_doc_length], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, max_doc_length, 1)]

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')

    return adict(
        targets=targets,
        loss=loss
    )


def loss_pretrain(logits, batch_size, max_doc_length, word_vocab_size):

    # reconstruction loss
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.float32, [batch_size, max_doc_length, word_vocab_size], name='targets')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, max_doc_length, 1)]

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = target_list), name='loss')

    return adict(
        targets=targets,
        loss=loss
    )


def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)


def model_size():

    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


if __name__ == '__main__':

    with tf.Session() as sess:
        # training with bidirectional LSTM encoder-scoring
        with tf.variable_scope('Model'):
            graph = cnn_sen_enc(word_vocab_size=100)
            graph.update(bilstm_doc_enc(graph.input_cnn, dropout=0.5))
            graph.update(label_prediction(graph.enc_outputs))
            graph.update(loss_extraction(graph.logits, batch_size=20, max_doc_length=35))
            graph.update(training_graph(graph.loss, learning_rate=1.0, max_grad_norm=5.0))

        # test with bidirectional LSTM encoder-scoring
        with tf.variable_scope('Model', reuse=True):
            tgraph = cnn_sen_enc(word_vocab_size=100)
            tgraph.update(bilstm_doc_enc(tgraph.input_cnn, dropout=0.5))
            tgraph.update(label_prediction(tgraph.enc_outputs))

        # training with LSTM encoder-decoder-scoring
        with tf.variable_scope('Model2'):
            graph2 = cnn_sen_enc(word_vocab_size=100)
            graph2.update(lstm_doc_enc(graph2.input_cnn, dropout=0.5))
            graph2.update(lstm_doc_dec(graph2.input_cnn, graph2.final_enc_state, dropout=0.5))
            graph2.update(label_prediction_att(graph2.enc_outputs, graph2.dec_outputs))
            graph2.update(loss_extraction(graph2.logits, batch_size=20, max_doc_length=35))
            graph2.update(training_graph(graph2.loss, learning_rate=1.0, max_grad_norm=5.0))

        # pre-training
        with tf.variable_scope('Model3'):
            pgraph = cnn_sen_enc(word_vocab_size=100)
            pgraph.update(bilstm_doc_enc(pgraph.input_cnn, dropout=0.5))
            pgraph.update(self_prediction(pgraph.enc_outputs, word_vocab_size=100))
            pgraph.update(loss_pretrain(pgraph.plogits, batch_size=20, max_doc_length=35, word_vocab_size=100))

        print('Model size is:', model_size())

        # need a fake variable to write scalar summary
        tf.summary.scalar('fake', 0)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
