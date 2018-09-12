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
    with tf.variable_scope('embedding'):
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
    '''
    Bilstm document encoder
    It constructs a list of sentence vectors in the document
    '''
    with tf.variable_scope('bilstm_enc'):
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
    with tf.variable_scope('lstm_enc'):
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


def vanilla_attention_decoder(enc_outputs,
                       batch_size=20,
                       num_rnn_layers=1,
                       rnn_size=80,
                       enc_state_size=650,
                       max_output_length=5,
                       dropout=0.0,
                       word_vocab_size=100,
                       word_embed_size=50,
                       mode='train'):

    ''' an attention decoder which does not support customized output layers'''
    input_dec = tf.placeholder(tf.int32, shape=[batch_size, max_output_length], name="input_dec")

    with tf.variable_scope('output_projection'):
          w = tf.get_variable(
              'w', [rnn_size, word_vocab_size], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
          w_t = tf.transpose(w)
          v = tf.get_variable(
              'v', [word_vocab_size], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))


    with tf.variable_scope('lstm_dec'):
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

        enc_outputs = [tf.reshape(x, [batch_size, 1, enc_state_size])
                           for x in enc_outputs]
        enc_states = tf.concat(axis=1, values=enc_outputs)

        dec_input = [tf.squeeze(s, [1]) for s in tf.split(input_dec, max_output_length, 1)] 

        is_feed_previous = (mode == 'decode')

        decoder_outputs, dec_out_state = tf.contrib.legacy_seq2seq.embedding_attention_decoder(
            dec_input, initial_rnn_state, enc_states,
            cell, output_projection=(w, v), feed_previous=is_feed_previous,  num_symbols=word_vocab_size, embedding_size=word_embed_size, num_heads=1)


    with tf.variable_scope('output'):
        model_outputs = []
        for i in range(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v))

    outputs = None
    topk_log_probs = None
    topk_ids = None
    if mode == 'decode':
        with tf.variable_scope('decode_output'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          outputs = tf.concat(
              axis=1, values=[tf.reshape(x, [batch_size, 1]) for x in best_outputs])

          topk_log_probs, topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), batch_size*2)

    return adict(
        input_dec = input_dec,
        logits = model_outputs,
        outputs = outputs,
        topk_log_probs = topk_log_probs,
        topk_ids = topk_ids
    )


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.
    Returns:
      A loop function.
    """
    def loop_function(prev, _):
        """function that feed previous model output rather than ground truth."""
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function


def flexible_attention_decoder(enc_outputs,
                       batch_size=20,
                       num_rnn_layers=1,
                       rnn_size=80,
                       enc_state_size=650,
                       max_output_length=5,
                       dropout=0.0,
                       word_vocab_size=100,
                       word_embed_size=50,
                       mode='train'):
    '''
    FIX THIS
    an attention decoder which supports customized output layers
    '''
    input_dec = tf.placeholder(tf.int32, shape=[batch_size, max_output_length], name="input_dec")
    dec_input = [tf.squeeze(s, [1]) for s in tf.split(input_dec, max_output_length, 1)]

    with tf.variable_scope('target_embedding'):
            target_embedding = tf.get_variable(
                'target_embedding', [word_vocab_size, word_embed_size], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
            clear_target_embedding_padding = tf.scatter_update(target_embedding, [0], tf.constant(0.0, shape=[1, word_embed_size]))

            embed_dec_input = [tf.nn.embedding_lookup(target_embedding, x) for x in dec_input]


    with tf.variable_scope('output_projection'):
          w = tf.get_variable(
              'w', [rnn_size, word_vocab_size], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))
          w_t = tf.transpose(w)
          v = tf.get_variable(
              'v', [word_vocab_size], dtype=tf.float32,
              initializer=tf.truncated_normal_initializer(stddev=1e-4))

    with tf.variable_scope('lstm_dec'):
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

        enc_outputs = [tf.reshape(x, [batch_size, 1, enc_state_size])
                           for x in enc_outputs]
        enc_states = tf.concat(axis=1, values=enc_outputs)

        initial_state_attention = (mode == 'decode')

        loop_function = None
        if mode == 'decode':
            loop_function = _extract_argmax_and_embed(
                target_embedding, (w, v), update_embedding=False)

        decoder_outputs, dec_out_state = tf.contrib.legacy_seq2seq.attention_decoder(
            embed_dec_input, initial_rnn_state, enc_states,
            cell, num_heads=1, loop_function=loop_function, 
            initial_state_attention=initial_state_attention)

    with tf.variable_scope('output'):
        model_outputs = []
        for i in range(len(decoder_outputs)):
          if i > 0:
            tf.get_variable_scope().reuse_variables()
          model_outputs.append(
              tf.nn.xw_plus_b(decoder_outputs[i], w, v))

    outputs = None
    topk_log_probs = None
    topk_ids = None
    if mode == 'decode':
        with tf.variable_scope('decode_output'):
          best_outputs = [tf.argmax(x, 1) for x in model_outputs]
          tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
          outputs = tf.concat(
              axis=1, values=[tf.reshape(x, [batch_size, 1]) for x in best_outputs])

          topk_log_probs, topk_ids = tf.nn.top_k(
              tf.log(tf.nn.softmax(model_outputs[-1])), batch_size*2)

    return adict(
        input_dec = input_dec,
        clear_target_embedding_padding = clear_target_embedding_padding,
        logits = model_outputs,
        outputs = outputs,
        topk_log_probs = topk_log_probs,
        topk_ids = topk_ids
    )


def loss_generation(logits, batch_size, max_output_length):
    '''compute sequence generation loss'''
    
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(tf.int64, [batch_size, max_output_length], name='targets')
        mask = tf.placeholder(tf.float32, [batch_size, max_output_length], name='mask')
        target_list = [tf.squeeze(x, [1]) for x in tf.split(targets, max_output_length, 1)]
        mask_list = [tf.squeeze(x, [1]) for x in tf.split(mask, max_output_length, 1)]

        loss = tf.contrib.legacy_seq2seq.sequence_loss(logits, target_list, mask_list)

    return adict(
        targets=targets,
        mask=mask,
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
        with tf.variable_scope('Model'):
            graph = cnn_sen_enc(word_vocab_size=100)
            graph.update(bilstm_doc_enc(graph.input_cnn, dropout=0.5))
            graph.update(flexible_attention_decoder(graph.enc_outputs, enc_state_size=1300))
            graph.update(loss_generation(graph.logits, batch_size=20, max_output_length=35))
            graph.update(training_graph(graph.loss, learning_rate=1.0, max_grad_norm=5.0))

        with tf.variable_scope('Model', reuse=True):
            tgraph = cnn_sen_enc(word_vocab_size=100)
            tgraph.update(bilstm_doc_enc(tgraph.input_cnn, dropout=0.5))
            tgraph.update(flexible_attention_decoder(tgraph.enc_outputs, enc_state_size=1300, mode='decode'))


        print('Model size is:', model_size())

        # need a fake variable to write scalar summary
        tf.summary.scalar('fake', 0)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
