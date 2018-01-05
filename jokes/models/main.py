import tensorflow as tf
import click
import json
import math
import os
import shutil
import numpy as np
import itertools

import jokes.models.input as model_input
import jokes.models.model as model
from jokes.data import char_vocab

def default_hparams(vocab_file):
    transform = char_vocab.Transform(vocab_file)
    HP = tf.contrib.training.HParams(
        batch_size=32,
        unroll_length=100,
        embedding_dim=200,
        learning_rate=0.01,
        num_layers=2,
        keep_prob=0.9,
        l2_reg=0.001,
        vocab_size=len(transform.chars),
        vocab=transform.chars
    )
    return HP

@click.group()
def cli():
    tf.logging.set_verbosity(tf.logging.INFO)

@cli.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("vocab_file", type=click.Path(exists=True))
@click.option("--model_dir", type=click.Path(), default="models/baseline")
@click.option("--sanity_check", is_flag=True)
@click.option("--steps", type=int, default=None)
@click.option("--hparams", type=str)
def train(data_file, vocab_file, model_dir, hparams=None, steps=None, sanity_check=False):
    HP = default_hparams(vocab_file)
    if hparams:
        HP.parse(hparams)
    else:
        print("No hparams specified. Using defaults.")

    print("="*80)
    print("Hyperparameters:")
    print(json.dumps(HP.values(), indent=2))
    print("="*80)
    print("Expected initial loss: %0.3f" % math.log(HP.vocab_size))

    ds = model_input.create_tf_dataset(data_file, batch_size=HP.batch_size,
                                       num_steps=HP.unroll_length)
    if sanity_check:
        # Take a small slice of the dataset.
        ds = ds.take(10)

    # Repeat indefinitely.
    ds.repeat()

    print("Saving vocab file to model directory.")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    shutil.copyfile(vocab_file, os.path.join(model_dir, "vocab.txt"))

    train_input_fn = lambda: model_input.oneshot_input_fn(ds)
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=200
    )

    estimator = tf.estimator.Estimator(model.model_fn, model_dir=model_dir, params=HP.values(),
                                       config=config)

    # Exit gracefully if we hit the end of the Dataset. This shouldn't happen since we are using
    # Dataset.repeat(count=None) above. It will just happily train until we hit ctrl-c or some other
    # hook exits training.
    try:
        estimator.train(train_input_fn, steps=steps)
    except tf.errors.OutOfRangeError:
        pass


def reconstitute(tokens):
    """ Takes a numpy array of bytes and creates a single string of all characters up to the <eos>
    token. """
    chars = [str(x, 'utf8') for x in tokens]
    if char_vocab.EOS in chars:
        idx = chars.index(char_vocab.EOS)
        chars = chars[0:idx]

    return ''.join(chars).replace('<newline>', '\n').replace('<tab>', '\t')

@cli.command()
@click.option("--model_dir", type=click.Path(), default="models/baseline")
@click.option("--temperature", type=float, default=1.0)
def generate(model_dir, temperature):
    vocab_file = os.path.join(model_dir, "vocab.txt")
    transform = char_vocab.Transform(vocab_file)
    HP = default_hparams(vocab_file)
    X = np.zeros((1, 1), dtype=np.int32)
    X[0, 0] = transform.GO_id()
    t = np.zeros((1, 1), dtype=np.float32)
    t[0, 0] = temperature
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"tokens": X, "temperature": t},
        num_epochs=1,
        shuffle=False)

    estimator = tf.estimator.Estimator(model.model_fn, model_dir=model_dir, params=HP.values())
    result = estimator.predict(input_fn=predict_input_fn)
    outputs = next(result)
    tokens = outputs['tokens']
    print("Sample:")
    print(reconstitute(tokens))

if __name__ == "__main__":
    cli()
