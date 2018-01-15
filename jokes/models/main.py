import tensorflow as tf
import click
import json
import math
import os
import sys
import shutil
import numpy as np
import itertools

import jokes.models.input as model_input
import jokes.models.model as model
from jokes.data import char_vocab

def default_hparams(vocab_file):
    transform = char_vocab.Transform(vocab_file)
    HP = tf.contrib.training.HParams(
        batch_size=64,
        unroll_length=20,
        embedding_dim=256,
        learning_rate=0.1,
        num_layers=2,
        keep_prob=0.9,
        optimizer='momentum',
        reset_after_eos=True,
        vocab_size=len(transform.chars),
        vocab=transform.chars
    )
    return HP

@click.group()
def cli():
    tf.logging.set_verbosity(tf.logging.INFO)

def get_dataset_size(ds):
    elem = ds.make_one_shot_iterator().get_next()
    num_iters = 0
    with tf.Session() as sess:
        while True:
            try:
                sess.run(elem)
                num_iters += 1
            except tf.errors.OutOfRangeError:
                break
    return num_iters

@cli.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("vocab_file", type=click.Path(exists=True))
def data_integrity_check(data_file, vocab_file):
    transform = char_vocab.Transform(vocab_file)
    ds = model_input.create_tf_dataset(data_file, batch_size=2,
                                       num_steps=20, forever=True)
    # Take a small slice of the dataset.
    ds = ds.take(10)
    iterator = ds.make_one_shot_iterator()
    x, y = iterator.get_next()
    with tf.Session() as sess:
        for iteration in range(10):
            x_eval, y_eval = sess.run((x,y))
            print("== Iteration #%d ==" % iteration)
            print('* ' + ''.join(transform.chars[i] for i in x_eval[0]))
            print('* ' + ''.join(transform.chars[i] for i in x_eval[1]))

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

    with tf.name_scope("TrainDataset"):
        ds = model_input.create_tf_dataset(data_file, batch_size=HP.batch_size,
                                           num_steps=HP.unroll_length)
        if sanity_check:
            # Take a small slice of the dataset.
            ds = ds.take(10)

        print("="*80)
        print("Determining number of iterations per epoch...")
        print("="*80)
        iters_per_epoch = get_dataset_size(ds)
        print("Done. iters_per_epoch = %d", iters_per_epoch)
        HP.add_hparam('iters_per_epoch', iters_per_epoch)

        # Repeat indefinitely.
        ds = ds.repeat()

    print("Saving vocab file to model directory.")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    shutil.copyfile(vocab_file, os.path.join(model_dir, "vocab.txt"))
    with open(os.path.join(model_dir, "hparams.json"), 'w') as fp:
        json.dump(HP.values(), fp)

    train_input_fn = lambda: model_input.oneshot_input_fn(ds)
    config = tf.estimator.RunConfig(
        save_checkpoints_steps=200
    )

    # Add some more frequent monitoring.
    if sanity_check:
        config = config.replace(save_summary_steps=1)

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
@click.option("--verbose", is_flag=True)
def generate(model_dir, temperature, verbose=False):
    vocab_file = os.path.join(model_dir, "vocab.txt")
    transform = char_vocab.Transform(vocab_file)

    hparam_file = os.path.join(model_dir, "hparams.json")
    with open(hparam_file) as fp:
        saved_params = json.load(fp)
    HP = tf.contrib.training.HParams(**saved_params)

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
    if verbose:
        print("Initial hidden state")
        print(outputs)

@cli.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--model_dir", type=click.Path(), default="models/baseline")
def eval(data_file, model_dir):
    vocab_file = os.path.join(model_dir, "vocab.txt")
    transform = char_vocab.Transform(vocab_file)

    hparam_file = os.path.join(model_dir, "hparams.json")
    with open(hparam_file) as fp:
        saved_params = json.load(fp)
    HP = tf.contrib.training.HParams(**saved_params)
    # Set batch size to 1 so we don't start in the middle sometimes?
    estimator = tf.estimator.Estimator(model.model_fn, model_dir=model_dir, params=HP.values())

    with tf.name_scope("EvalDataset"):
        ds = model_input.create_tf_dataset(data_file, batch_size=HP.batch_size,
                                           num_steps=HP.unroll_length)
        eval_input_fn = lambda: model_input.oneshot_input_fn(ds)

    metrics = estimator.evaluate(input_fn=eval_input_fn)
    perplexity = math.exp(metrics['mean_loss'])
    print(metrics)
    print("Per-character perplexity: %0.2f" % perplexity)

if __name__ == "__main__":
    cli()
