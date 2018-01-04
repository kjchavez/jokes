import tensorflow as tf
import numpy as np

from jokes.data import char_vocab


class BatchedDataset(object):
    def __init__(self, raw_data, batch_size, num_steps, forever=False):
        raw_data = np.array(raw_data)
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        self.data = np.reshape(raw_data[0 : batch_size * batch_len],
                               [batch_size, batch_len])
        self.epoch_size = (batch_len - 1) // num_steps
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.forever = forever

    def generator(self):
        i = 0
        while True:
            x = self.data[:, i*self.num_steps:(i+1)*self.num_steps]
            y = self.data[:, (i*self.num_steps + 1):((i+1)*self.num_steps+1)]
            yield x, y
            i += 1
            if i == self.epoch_size:
                if self.forever:
                    i = 0
                else:
                    break

def file_sequence_generator(filename, batch_size, num_steps, forever=False):
    with open(filename) as fp:
        data = [int(x) for x in fp.readlines()]

    batched_dset = BatchedDataset(data, batch_size, num_steps, forever=forever)
    return batched_dset.generator

def create_tf_dataset(filename, batch_size, num_steps, forever=False):
    return tf.data.Dataset.from_generator(file_sequence_generator(filename, batch_size, num_steps,
                                                                  forever=forever),
                                          (tf.int64, tf.int64),
                                          (tf.TensorShape([batch_size, num_steps]),
                                           tf.TensorShape([batch_size, num_steps])))

def _parse_function(example_proto):
  features = {"tokens": tf.VarLenFeature(tf.int64),
              "length": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return tf.sparse_tensor_to_dense(parsed_features["tokens"]), parsed_features["length"]


def create_joke_aligned_dataset(filename, batch_size):
    return tf.data.TFRecordDataset([filename]).map(_parse_function) \
             .padded_batch(batch_size, ((None,), ()),
                           padding_values=(np.int64(char_vocab.PAD_ID),
                                           np.int64(0)))


def oneshot_input_fn(dset):
    iterator = dset.make_one_shot_iterator()
    x, y = iterator.get_next()
    return {'tokens': x}, y


if __name__ == "__main__":
    # ds = create_tf_dataset("data/processed/jokes.dat", batch_size=32, num_steps=50, forever=False)
    ds = create_joke_aligned_dataset("data/processed/jokes.tfrecord", 32)
    value = ds.make_one_shot_iterator().get_next()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        x = sess.run(value)
        print(x)

