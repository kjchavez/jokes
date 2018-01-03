import os
import glob
import click
import logging
import json
import random
from jokes.data import char_vocab

def joke_iter(filepattern):
    filenames = glob.glob(filepattern)
    for fname in filenames:
        with open(fname) as fp:
            data = json.load(fp)
            random.shuffle(data)
            for elem in data:
                if 'body' not in elem:
                    continue
                if 'title' in elem:
                    yield elem['title'] + '\n' + elem['body']
                else:
                    yield elem['body']

def project_dir():
    return os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

def ensure_parent_dir_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_filepattern')
@click.argument('vocab_filename', type=click.Path())
def build_vocab(input_filepattern, vocab_filename):
    ensure_parent_dir_exists(vocab_filename)
    char_vocab.create_char_vocab(joke_iter(input_filepattern), vocab_filename)

@cli.command()
@click.argument('input_filepattern')
@click.argument('vocab_filename', type=click.Path(exists=True))
@click.argument('outfile', type=click.Path())
def build_dataset(input_filepattern, vocab_filename, outfile):
    ensure_parent_dir_exists(outfile)
    transform = char_vocab.Transform(vocab_filename)
    with open(outfile, 'w') as outfp:
        for joke in joke_iter(input_filepattern):
            indices = transform.apply(joke)
            print(transform.GO_id(), file=outfp)
            for i in indices:
                print(i, file=outfp)
            print(transform.EOS_id(), file=outfp)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    processed = os.path.join(project_dir(), 'data', 'processed')
    if not os.path.exists(processed):
        os.makedirs(processed)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    cli()
