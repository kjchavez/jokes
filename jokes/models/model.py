import tensorflow as tf

import sys
from jokes.data import char_vocab

DEFAULT_MAX_SAMPLE_LENGTH = 1000
DEFAULT_MAX_GRAD_NORM = 5.0

def clipped_train_op(loss, var_list, optimizer, max_grad_norm=DEFAULT_MAX_GRAD_NORM, add_summaries=True):
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads, tvars = zip(*grads_and_vars)
    grads, _ = tf.clip_by_global_norm(grads,
                                      max_grad_norm)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())
    if add_summaries:
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradient', grad)

    return train_op


def assign_lstm_state(ref, value):
    ops = []
    for i, state_tuple in enumerate(value):
        ops.append(tf.assign(ref[i].c, state_tuple.c))
        ops.append(tf.assign(ref[i].h, state_tuple.h))

    return tf.group(*ops)


def model_fn(features, labels, mode, params):
    """
    NOTE(kjchavez): |mode| can take on an extra value 'GENERATE' which is not a typical estimator
    modekey.
    """
    batch_size = params['batch_size']
    unroll_length = params['unroll_length']
    embedding_dim = params['embedding_dim']
    vocab_size = params['vocab_size']
    keep_prob = params['keep_prob']
    num_layers = params['num_layers']
    learning_rate = params['learning_rate']
    # Used to determine when to reset hidden state!
    iters_per_epoch = params['iters_per_epoch']
    dtype = tf.float32
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    tokens = features['tokens']
    targets = labels

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          embedding_dim, state_is_tuple=True,
          reuse=tf.get_variable_scope().reuse)

    attn_cell = lstm_cell
    if is_training and keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, embedding_dim], dtype=dtype)
      inputs = tf.nn.embedding_lookup(embedding, tokens)

    if is_training and keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)

    # NOTE: zero_state is a tuple of LSTMStateTuples. But we may be able to do something simpler
    # with the state that we want to keep as a variable.
    print("Cell::zero_state", cell.zero_state(batch_size, dtype))
    with tf.name_scope("HiddenStateInitializer"):
        init_state = []
        reinit_ops = []
        for i, state_tuple in enumerate(cell.zero_state(batch_size, dtype)):
            c = tf.get_variable("init_c_%d" % i, initializer=state_tuple.c, trainable=False)
            h = tf.get_variable("init_h_%d" % i, initializer=state_tuple.h, trainable=False)
            init_state.append(tf.contrib.rnn.LSTMStateTuple(c,h))
            reinit_ops.extend([tf.assign(c, tf.zeros_like(c)), tf.assign(h, tf.zeros_like(h))])

        maybe_reinit_hidden_state = tf.cond(tf.equal(tf.mod(tf.train.get_or_create_global_step(),
                                                            iters_per_epoch), 0),
                                            lambda: tf.group(*reinit_ops),  # Once per epoch
                                            tf.no_op)
        init_state = tuple(init_state)

    table = None
    # Predict will currently draw a sample from the generative model. We likely need other
    # functionality as well, such as, tell me the likelihood of this character sequence, or provide
    # a probability distribution over the next character. But we'll come back to that later.
    if mode == tf.estimator.ModeKeys.PREDICT:
        mapping_string = tf.constant(params['vocab'])
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
                    mapping_string, default_value="<unk>")
        # We need to feed an input at each step from the output of the previous
        # step.
        temperature = features['temperature']
        helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
          embedding=embedding,
          start_tokens=tf.tile([char_vocab.GO_ID], [batch_size]),
          end_token=char_vocab.EOS_ID,
          softmax_temperature=temperature)
    else:
        # NOTE(kjchavez): By default, the inputs to TrainingHelper are assumed
        # to be batch major. Use time_major=True if you care to flip it.
        helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=tf.tile([unroll_length], [batch_size]))

    output_layer = tf.layers.Dense(vocab_size, name="fully_connected",
                                   activation=None)

    # NOTE(kjchavez): The |output_layer| is applied prior to storing the result OR SAMPLING. This
    # last part is important. If you forget, and the LSTM hidden dim is smaller than the vocab, you
    # will be out of luck.
    with tf.control_dependencies([maybe_reinit_hidden_state]):
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=init_state,
            output_layer=output_layer)

        outputs, final_state, seq_lens = tf.contrib.seq2seq.dynamic_decode(
           decoder=decoder,
           output_time_major=False,
           impute_finished=True,
           maximum_iterations=params.get('max_sample_length',
               DEFAULT_MAX_SAMPLE_LENGTH))

    tf.summary.histogram("lstm_activations", outputs.rnn_output)

    predictions = {}
    with tf.name_scope("Prediction"):
        output_token_ids = outputs.sample_id
        tf.summary.histogram("token_id", output_token_ids)
        predictions['token_ids'] = output_token_ids
        if table is not None:
            predictions['tokens'] = table.lookup(tf.to_int64(outputs.sample_id))

        logits = outputs.rnn_output
        tf.summary.histogram("logits", logits)
        token_probability = tf.nn.softmax(logits)

    loss = None
    train_op = None
    metric_ops = {}
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                tf.ones_like(targets, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
        tf.summary.scalar('loss', loss)

        # NOTE: All elements are the same length in this model_fn, therefore it's okay to take this
        # mean of means. Otherwise, we'd want to avoid averaging_across_x in the loss function above
        # and instead use the individual losses in this metric.
        metric_ops["mean_loss"] = tf.metrics.mean(loss)

        # NOTE(kjchavez): It's okay if we add this train op to the EstimatorSpec in EVAL mode. It
        # will be ignored. Notice that we also pass the 'mode' itself.
        tvars = tf.trainable_variables()
        opt_map = {
            'sgd': tf.train.GradientDescentOptimizer,
            'adagrad': tf.train.AdagradOptimizer,
            'adam': tf.train.AdamOptimizer,
            'rmsprop': tf.train.RMSPropOptimizer
        }
        if params['optimizer'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            optimizer = opt_map[params['optimizer']](learning_rate)

        # NOTE: We may want to ensure that if we with the <eos> token, that the state is cleared to
        # zeros.
        train_op = clipped_train_op(loss, tvars, optimizer)
        with tf.control_dependencies([train_op]):
            assign_op = assign_lstm_state(init_state, final_state)

        train_op = tf.group(maybe_reinit_hidden_state, train_op, assign_op)

    return tf.estimator.EstimatorSpec(
        eval_metric_ops=metric_ops,
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)
