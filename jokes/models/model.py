import tensorflow as tf

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
    l2_reg = params['l2_reg']
    dtype = tf.float32
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN

    tokens = features['tokens']
    targets = labels

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
          embedding_dim, forget_bias=0.0, state_is_tuple=True,
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

    output_layer = tf.layers.Dense(vocab_size,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=cell.zero_state(batch_size, dtype),
        output_layer=output_layer)

    outputs, state, seq_lens = tf.contrib.seq2seq.dynamic_decode(
       decoder=decoder,
       output_time_major=False,
       impute_finished=True,
       maximum_iterations=params.get('max_sample_length',
           DEFAULT_MAX_SAMPLE_LENGTH))

    predictions = {}

    output_token_ids = outputs.sample_id
    predictions['token_ids'] = output_token_ids
    if table is not None:
        predictions['tokens'] = table.lookup(tf.to_int64(outputs.sample_id))

    logits = outputs.rnn_output
    token_probability = tf.nn.softmax(logits)
    loss = None
    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                tf.ones_like(targets, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)
        tf.summary.scalar('loss', loss)

        tvars = tf.trainable_variables()
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = clipped_train_op(loss, tvars, optimizer)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)
