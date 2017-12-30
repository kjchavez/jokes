import os
import glob
import click
import logging
import json

# We want to preserve newline characters since the structure often adds to the joke
# so we use a different delimiter to mark the end of the joke.
EOJ = '#'

def clean(text):
    # remove any EOJ characters from the text
    return text.replace(EOJ, '')

@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    base_path = os.path.join(project_dir, "data", "external", "joke-dataset-master")
    filenames = glob.glob(os.path.join(base_path, '*.json'))

    outfile = os.path.join(project_dir, "data", "processed", 'jokes.dat')
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    with open(outfile, 'w') as outfp:
        for fname in filenames:
            with open(fname) as fp:
                data = json.load(fp)
                for elem in data:
                    if 'body' in elem:
                        outfp.write(clean(elem['body']))
                        outfp.write(EOJ)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
